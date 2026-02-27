import json
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

    def _build_partitioned_inputs(self, single_inputs):
        labels = single_inputs.get("labels")
        token_modality_type = single_inputs.get("token_modality_type")
        if not isinstance(labels, torch.Tensor) or not isinstance(token_modality_type, torch.Tensor):
            return [("all", single_inputs, None)]

        ignore_index = -100
        if self.label_smoother is not None and hasattr(self.label_smoother, "ignore_index"):
            ignore_index = self.label_smoother.ignore_index

        supervised_mask = labels.ne(ignore_index)
        image_mask = supervised_mask & token_modality_type.eq(1)
        text_mask = supervised_mask & token_modality_type.ne(1)

        partitions = [("all", single_inputs, int(supervised_mask.sum().item()))]
        for partition_name, partition_mask in (("image_only", image_mask), ("text_only", text_mask)):
            token_count = int(partition_mask.sum().item())
            if token_count == 0:
                continue
            partition_inputs = dict(single_inputs)
            partition_labels = labels.clone()
            partition_labels[~partition_mask] = ignore_index
            partition_inputs["labels"] = partition_labels
            partitions.append((partition_name, partition_inputs, token_count))

        return partitions

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

            for grad_partition, partition_inputs, supervised_token_count in self._build_partitioned_inputs(inputs):
                loss = self.compute_loss(model, self._strip_auxiliary_inputs(partition_inputs))
                all_grads = torch.autograd.grad(
                    loss,
                    [param for _, param in all_trainable_params],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
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
                        "grad_partition": grad_partition,
                        "supervised_token_count": supervised_token_count,
                        "adapter_type": self._extract_adapter_type(name),
                        "param_name": name,
                        "layer_depth": self._extract_layer_depth(name),
                        "grad_norm": float(grad_cpu.norm(p=2).item()),
                        "grad_mean": float(grad_cpu.mean().item()),
                        "grad_std": float(grad_cpu.std(unbiased=False).item()),
                        "grad_abs_mean": float(grad_cpu.abs().mean().item()),
                        "numel": int(grad_cpu.numel()),
                        "grad_was_none": bool(grad_is_none),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def training_step(self, model, inputs, *args, **kwargs):
        if self._grad_logging_enabled:
            with torch.enable_grad():
                self._log_example_level_gradients(model, inputs)

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
