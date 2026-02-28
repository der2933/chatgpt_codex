import hashlib
import json
import numpy as np
import os
import re
import torch
import torch.nn as nn

from tqdm import tqdm
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy,
)
from train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
)
from train.eval_utils import get_scores
from transformers import GenerationConfig


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param




class QwenTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(QwenTrainer, self).__init__(*args, **kwargs)
        self._grad_logging_enabled = bool(getattr(self.args, "enable_gradient_logging", False))
        self._grad_log_every_n_steps = max(1, int(getattr(self.args, "gradient_log_every_n_steps", 50)))
        self._grad_log_max_examples_per_step = max(1, int(getattr(self.args, "gradient_log_max_examples_per_step", 1)))
        grad_log_path = getattr(self.args, "gradient_log_path", None)
        self._grad_log_path = grad_log_path or os.path.join(self.args.output_dir, "gradient_logs.jsonl")
        self._adapter_grad_params = None
        self._all_trainable_grad_params = None
        self._adapter_grad_indices = None
        self._grad_log_save_full_grad = bool(getattr(self.args, "gradient_log_save_full_grad", False))
        self._grad_log_full_grad_max_params = int(getattr(self.args, "gradient_log_full_grad_max_params", 8))
        full_grad_dir = getattr(self.args, "gradient_log_full_grad_dir", None)
        self._grad_log_full_grad_dir = full_grad_dir or os.path.join(self.args.output_dir, "gradient_vectors")

        self._hidden_state_logging_enabled = bool(getattr(self.args, "enable_hidden_state_logging", False))
        self._hidden_state_log_every_n_steps = max(1, int(getattr(self.args, "hidden_state_log_every_n_steps", 200)))
        self._hidden_state_log_max_tokens_per_modality = max(1, int(getattr(self.args, "hidden_state_log_max_tokens_per_modality", 64)))
        hidden_state_log_path = getattr(self.args, "hidden_state_log_path", None)
        self._hidden_state_log_path = hidden_state_log_path or os.path.join(self.args.output_dir, "hidden_state_logs.jsonl")
        hidden_state_vector_dir = getattr(self.args, "hidden_state_vector_dir", None)
        self._hidden_state_vector_dir = hidden_state_vector_dir or os.path.join(self.args.output_dir, "hidden_state_vectors")

    def _is_transformer_block_adapter_param(self, name: str) -> bool:
        if "model.layers." not in name:
            return False
        adapter_keywords = ("custom_lora", "lora_", "adapter", "kradapter")
        return any(keyword in name.lower() for keyword in adapter_keywords)

    def _extract_layer_depth(self, name: str) -> int:
        match = re.search(r"model\.layers\.(\d+)", name)
        return int(match.group(1)) if match else -1

    def _extract_adapter_type(self, name: str) -> str:
        lower_name = name.lower()
        if "custom_lora" in lower_name:
            return "custom_lora"
        if "kradapter" in lower_name:
            return "kradapter"
        if "lora_a" in lower_name:
            return "lora_A"
        if "lora_b" in lower_name:
            return "lora_B"
        if "adapter" in lower_name:
            return "adapter"
        return "other"

    def _get_adapter_grad_params(self):
        if self._adapter_grad_params is not None:
            return self._adapter_grad_params
        self._adapter_grad_params = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad and self._is_transformer_block_adapter_param(name)
        ]
        if self._grad_logging_enabled and self.is_world_process_zero():
            logger.info(
                f"Gradient logging enabled. Tracking {len(self._adapter_grad_params)} transformer-block adapter parameters."
            )
        return self._adapter_grad_params

    def _get_all_trainable_grad_params(self):
        if self._all_trainable_grad_params is not None:
            return self._all_trainable_grad_params
        self._all_trainable_grad_params = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        return self._all_trainable_grad_params

    def _get_adapter_grad_indices(self):
        if self._adapter_grad_indices is not None:
            return self._adapter_grad_indices

        all_params = self._get_all_trainable_grad_params()
        adapter_param_ids = {id(param) for _, param in self._get_adapter_grad_params()}
        self._adapter_grad_indices = [
            idx
            for idx, (_, param) in enumerate(all_params)
            if id(param) in adapter_param_ids
        ]
        return self._adapter_grad_indices

    def _should_log_gradients(self) -> bool:
        if not self._grad_logging_enabled:
            return False
        if not self.is_world_process_zero():
            return False
        next_step = self.state.global_step + 1
        return next_step % self._grad_log_every_n_steps == 0

    def _strip_auxiliary_inputs(self, inputs):
        model_inputs = dict(inputs)
        model_inputs.pop("modality_type", None)
        model_inputs.pop("image_grid_counts", None)
        model_inputs.pop("video_grid_counts", None)
        return model_inputs

    def _get_ignore_index(self) -> int:
        if self.label_smoother is not None and hasattr(self.label_smoother, "ignore_index"):
            return int(self.label_smoother.ignore_index)
        return -100

    def _summarize_supervised_tokens(self, inputs):
        labels = inputs.get("labels")
        token_modality_type = inputs.get("token_modality_type")
        if not isinstance(labels, torch.Tensor) or not isinstance(token_modality_type, torch.Tensor):
            return {
                "total_supervised_tokens": None,
                "text_supervised_tokens": None,
                "image_supervised_tokens": None,
                "text_token_ratio": None,
                "image_token_ratio": None,
            }

        ignore_index = self._get_ignore_index()
        supervised_mask = labels.ne(ignore_index)
        image_mask = supervised_mask & token_modality_type.eq(1)
        text_mask = supervised_mask & token_modality_type.ne(1)

        total_tokens = int(supervised_mask.sum().item())
        text_tokens = int(text_mask.sum().item())
        image_tokens = int(image_mask.sum().item())
        denom = max(1, total_tokens)
        return {
            "total_supervised_tokens": total_tokens,
            "text_supervised_tokens": text_tokens,
            "image_supervised_tokens": image_tokens,
            "text_token_ratio": float(text_tokens / denom),
            "image_token_ratio": float(image_tokens / denom),
        }

    def _select_full_grad_dump_indices(self, adapter_grad_indices, all_trainable_params, all_grads):
        if not self._grad_log_save_full_grad or len(adapter_grad_indices) == 0:
            return set()

        scored = []
        for grad_idx in adapter_grad_indices:
            _, param = all_trainable_params[grad_idx]
            grad = all_grads[grad_idx]
            if grad is None:
                norm_val = 0.0
            else:
                norm_val = float(grad.detach().float().norm(p=2).item())
            scored.append((norm_val, grad_idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        if self._grad_log_full_grad_max_params <= 0:
            return {idx for _, idx in scored}
        return {idx for _, idx in scored[: self._grad_log_full_grad_max_params]}

    def _dump_full_grad_vector(self, step, grad_partition, param_name, grad_cpu):
        os.makedirs(self._grad_log_full_grad_dir, exist_ok=True)
        safe_name = hashlib.md5(param_name.encode("utf-8")).hexdigest()[:16]
        filename = f"step{step:08d}_{grad_partition}_{safe_name}.npy"
        path = os.path.join(self._grad_log_full_grad_dir, filename)
        np.save(path, grad_cpu.numpy())
        return path

    def _should_log_hidden_states(self) -> bool:
        if not self._hidden_state_logging_enabled:
            return False
        if not self.is_world_process_zero():
            return False
        next_step = self.state.global_step + 1
        return next_step % self._hidden_state_log_every_n_steps == 0

    def _dump_hidden_vectors(self, step, layer_depth, sample_idx, modality, vector):
        os.makedirs(self._hidden_state_vector_dir, exist_ok=True)
        file_name = f"step{step:08d}_layer{layer_depth:03d}_sample{sample_idx:04d}_{modality}.npy"
        path = os.path.join(self._hidden_state_vector_dir, file_name)
        np.save(path, vector)
        return path

    def _log_token_hidden_states(self, model, inputs):
        if not self._should_log_hidden_states():
            return

        model_inputs = self._strip_auxiliary_inputs(inputs)
        if "token_modality_type" not in model_inputs or "labels" not in model_inputs:
            return

        hs_log_dir = os.path.dirname(self._hidden_state_log_path)
        if hs_log_dir:
            os.makedirs(hs_log_dir, exist_ok=True)

        step = int(self.state.global_step + 1)
        ignore_index = self._get_ignore_index()

        with torch.no_grad():
            outputs = model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            return

        labels = model_inputs["labels"]
        token_modality_type = model_inputs["token_modality_type"]
        supervised_mask = labels.ne(ignore_index)
        modality_ids = model_inputs.get("modality_type")

        batch_size = int(labels.size(0))
        with open(self._hidden_state_log_path, "a", encoding="utf-8") as f:
            for layer_idx, layer_hidden in enumerate(hidden_states[1:]):
                layer_depth = layer_idx
                layer_hidden = layer_hidden.detach().float().cpu()

                for sample_idx in range(batch_size):
                    sample_hidden = layer_hidden[sample_idx]
                    sample_supervised = supervised_mask[sample_idx]
                    sample_token_modality = token_modality_type[sample_idx]

                    sample_image_mask = sample_supervised & sample_token_modality.eq(1)
                    sample_text_mask = sample_supervised & sample_token_modality.ne(1)

                    for modality, mask in (("image", sample_image_mask), ("text", sample_text_mask)):
                        token_count = int(mask.sum().item())
                        if token_count == 0:
                            continue

                        vectors = sample_hidden[mask]
                        mean_vector = vectors.mean(dim=0)
                        vec_np = mean_vector.numpy()
                        vec_path = self._dump_hidden_vectors(step, layer_depth, sample_idx, modality, vec_np)
                        record = {
                            "step": step,
                            "layer_depth": int(layer_depth),
                            "sample_index": int(sample_idx),
                            "sample_modality_type": (
                                int(modality_ids[sample_idx].item())
                                if isinstance(modality_ids, torch.Tensor)
                                else None
                            ),
                            "modality": modality,
                            "token_count": token_count,
                            "hidden_dim": int(vec_np.shape[0]),
                            "hidden_path": vec_path,
                            "hidden_norm": float(np.linalg.norm(vec_np)),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_example_level_gradients(self, model, inputs):
        if not self._should_log_gradients():
            return

        all_trainable_params = self._get_all_trainable_grad_params()
        adapter_grad_indices = self._get_adapter_grad_indices()
        if len(adapter_grad_indices) == 0:
            return

        step = int(self.state.global_step + 1)

        grad_log_dir = os.path.dirname(self._grad_log_path)
        if grad_log_dir:
            os.makedirs(grad_log_dir, exist_ok=True)
        with open(self._grad_log_path, "a", encoding="utf-8") as f:
            modality_ids = inputs.get("modality_type")
            if isinstance(modality_ids, torch.Tensor):
                modality_ids = modality_ids.detach().cpu().tolist()
            else:
                modality_ids = []
            token_summary = self._summarize_supervised_tokens(inputs)

            supervised_token_count = token_summary["total_supervised_tokens"]
            loss = self.compute_loss(model, self._strip_auxiliary_inputs(inputs))
            all_grads = torch.autograd.grad(
                loss,
                [param for _, param in all_trainable_params],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            full_grad_dump_indices = self._select_full_grad_dump_indices(
                adapter_grad_indices,
                all_trainable_params,
                all_grads,
            )

            for grad_idx in adapter_grad_indices:
                name, param = all_trainable_params[grad_idx]
                grad = all_grads[grad_idx]
                grad_is_none = grad is None
                if grad_is_none:
                    grad = torch.zeros_like(param, memory_format=torch.preserve_format)
                grad_cpu = grad.detach().float().cpu()
                record = {
                    "step": step,
                    "example_index": -1,
                    "modality_ids": modality_ids,
                    "modality_type": "batch",
                    "grad_partition": "all",
                    "supervised_token_count": supervised_token_count,
                    "total_supervised_tokens": token_summary["total_supervised_tokens"],
                    "text_supervised_tokens": token_summary["text_supervised_tokens"],
                    "image_supervised_tokens": token_summary["image_supervised_tokens"],
                    "text_token_ratio": token_summary["text_token_ratio"],
                    "image_token_ratio": token_summary["image_token_ratio"],
                    "partition_token_ratio": 1.0 if supervised_token_count is not None else None,
                    "adapter_type": self._extract_adapter_type(name),
                    "param_name": name,
                    "layer_depth": self._extract_layer_depth(name),
                    "grad_norm": float(grad_cpu.norm(p=2).item()),
                    "grad_norm_per_token": (
                        float(grad_cpu.norm(p=2).item() / max(1, supervised_token_count))
                        if supervised_token_count is not None
                        else None
                    ),
                    "grad_mean": float(grad_cpu.mean().item()),
                    "grad_std": float(grad_cpu.std(unbiased=False).item()),
                    "grad_abs_mean": float(grad_cpu.abs().mean().item()),
                    "numel": int(grad_cpu.numel()),
                    "grad_was_none": bool(grad_is_none),
                    "grad_path": None,
                }
                if grad_idx in full_grad_dump_indices:
                    record["grad_path"] = self._dump_full_grad_vector(
                        step,
                        "all",
                        name,
                        grad_cpu,
                    )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def training_step(self, model, inputs, *args, **kwargs):
        if self._grad_logging_enabled:
            with torch.enable_grad():
                self._log_example_level_gradients(model, inputs)

        if self._hidden_state_logging_enabled:
            self._log_token_hidden_states(model, inputs)

        model_inputs = self._strip_auxiliary_inputs(inputs)
        return super().training_step(model, model_inputs, *args, **kwargs)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "visual" in name and "merger" not in name
                ]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [
                    name for name, _ in opt_model.named_parameters() if "merger" in name
                ]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        if not self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters(), require_grad_only=False
            )
            torch.save(
                non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin")
            )

            if (
                self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH]
                and self.state.best_global_step
            ):
                best_checkpoint_folder = (
                    f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                )
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                if os.path.exists(best_checkpoint_dir):
                    self.state.best_model_checkpoint = best_checkpoint_dir

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                self._save_scaler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb
                    for cb in self.callback_handler.callbacks + [self.control]
                    if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
        else:
            super(QwenTrainer, self)._save_checkpoint(model, trial)

    # def training_step(self, model, inputs):
    #     for name, param in model.named_parameters():
    #         if 'visual' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
    #
    #     return super().training_step(model, inputs)

    def generate_eval_save(self, test_dataset, processor, test_data_path, output_json_path):
        logger.info("start generate test dataset answer")
        dataloader = self.get_eval_dataloader(test_dataset)
        test_qids = test_dataset.get_qids()
        batch_size = self.args.per_device_eval_batch_size

        generation_config = GenerationConfig(
            max_new_tokens=16,
            do_sample=False,
            bos_token_id=151643,
            eos_token_id=[151645, 151643],
        )

        results_ans = {}
        results_rationale = {}
        results_reference = {}

        preds_list = []
        targets_list = []

        num_batches = len(dataloader)
        pbar = tqdm(total=num_batches)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(dataloader):
                # for k, v in batch.items():
                #     print(batch)
                    # if isinstance(v, torch.Tensor):
                    #     batch[k] = v.to(self.model.device)
                labels = batch.pop("labels")

                labels = torch.where(
                    labels == -100, torch.tensor(processor.tokenizer.pad_token_id), labels
                )

                with torch.no_grad():
                    preds = self.model.generate(
                        **batch, generation_config=generation_config
                    )

                prompts = processor.batch_decode(
                    batch["input_ids"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                preds = processor.batch_decode(
                    preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for j in range(len(prompts)):
                    preds[j] = preds[j][len(prompts[j]):].strip()


                targets = processor.batch_decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                

                actual_batch_size = len(preds)
                for idx, qid in enumerate(
                    range(
                        batch_idx * batch_size, batch_idx * batch_size + actual_batch_size
                    )
                ):
                    # ALE
                    actual_qid = test_qids[qid]
                    pred = preds[idx]
                    ref = targets[idx]

                    extract_pred = pred[0] if pred is not None and len(pred) > 0 else ""

                    results_ans[actual_qid] = extract_pred
                    results_rationale[actual_qid] = pred
                    results_reference[actual_qid] = ref

                    preds_list.append(pred)
                    targets_list.append(ref)
                torch.cuda.empty_cache()
                pbar.update()

        # scores = get_scores(
        #     results_ans,
        #     results_rationale,
        #     results_reference,
        #     test_data_path,
        # )
        # print(scores)
        # for key, value in scores.items():
        #     print(f"\n{key}:")
        #     for sub_key, sub_value in value.items():
        #         print(f"  {sub_key}: {sub_value}")

        output_data = dict(
            results_ans=results_ans,
            results_rationale=results_rationale,
            results_reference=results_reference,
            preds=preds_list,
            labels=targets_list,
        )

        with open(output_json_path, "w") as f:
            json.dump(output_data, f, indent=4)
