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

    def _should_log_gradients(self) -> bool:
        if not self._grad_logging_enabled:
            return False
        if not self.is_world_process_zero():
            return False
        next_step = self.state.global_step + 1
        return next_step % self._grad_log_every_n_steps == 0

    def _slice_example_from_inputs(self, inputs, example_idx: int):
        single_inputs = {}
        batch_size = int(inputs["input_ids"].size(0))

        image_counts = inputs.get("image_grid_counts")
        video_counts = inputs.get("video_grid_counts")
        image_prefix = None
        video_prefix = None
        if isinstance(image_counts, torch.Tensor) and image_counts.numel() == batch_size:
            image_prefix = torch.nn.functional.pad(image_counts.to(dtype=torch.long).cumsum(dim=0), (1, 0), value=0)
        if isinstance(video_counts, torch.Tensor) and video_counts.numel() == batch_size:
            video_prefix = torch.nn.functional.pad(video_counts.to(dtype=torch.long).cumsum(dim=0), (1, 0), value=0)

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dim() > 0 and value.size(0) == batch_size:
                    single_inputs[key] = value[example_idx : example_idx + 1]
                else:
                    single_inputs[key] = value
            elif isinstance(value, list):
                if len(value) == batch_size:
                    single_inputs[key] = [value[example_idx]]
                else:
                    single_inputs[key] = value
            else:
                single_inputs[key] = value

        if image_prefix is not None:
            start = int(image_prefix[example_idx].item())
            end = int(image_prefix[example_idx + 1].item())
            if "pixel_values" in inputs:
                single_inputs["pixel_values"] = inputs["pixel_values"][start:end]
            if "image_grid_thw" in inputs:
                single_inputs["image_grid_thw"] = inputs["image_grid_thw"][start:end]

        if video_prefix is not None:
            start = int(video_prefix[example_idx].item())
            end = int(video_prefix[example_idx + 1].item())
            if "pixel_values_videos" in inputs:
                single_inputs["pixel_values_videos"] = inputs["pixel_values_videos"][start:end]
            if "video_grid_thw" in inputs:
                single_inputs["video_grid_thw"] = inputs["video_grid_thw"][start:end]
            if "second_per_grid_ts" in inputs and isinstance(inputs["second_per_grid_ts"], list):
                single_inputs["second_per_grid_ts"] = inputs["second_per_grid_ts"][start:end]

        return single_inputs

    def _strip_auxiliary_inputs(self, inputs):
        model_inputs = dict(inputs)
        model_inputs.pop("modality_type", None)
        model_inputs.pop("image_grid_counts", None)
        model_inputs.pop("video_grid_counts", None)
        return model_inputs

    def _log_example_level_gradients(self, model, inputs):
        if not self._should_log_gradients():
            return

        adapter_params = self._get_adapter_grad_params()
        if len(adapter_params) == 0:
            return

        batch_size = int(inputs["input_ids"].size(0))
        n_examples = min(batch_size, self._grad_log_max_examples_per_step)
        step = int(self.state.global_step + 1)

        grad_log_dir = os.path.dirname(self._grad_log_path)
        if grad_log_dir:
            os.makedirs(grad_log_dir, exist_ok=True)
        with open(self._grad_log_path, "a", encoding="utf-8") as f:
            for example_idx in range(n_examples):
                single_inputs = self._slice_example_from_inputs(inputs, example_idx)
                loss = self.compute_loss(model, self._strip_auxiliary_inputs(single_inputs))
                grads = torch.autograd.grad(
                    loss,
                    [param for _, param in adapter_params],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

                modality_id = int(single_inputs.get("modality_type", torch.tensor([0], device=loss.device))[0].item())
                modality_map = {0: "text_only", 1: "image_text", 2: "video_text"}
                modality_type = modality_map.get(modality_id, "unknown")

                for (name, _), grad in zip(adapter_params, grads):
                    if grad is None:
                        continue
                    grad_cpu = grad.detach().float().cpu()
                    record = {
                        "step": step,
                        "example_index": int(example_idx),
                        "modality_id": modality_id,
                        "modality_type": modality_type,
                        "adapter_type": self._extract_adapter_type(name),
                        "param_name": name,
                        "layer_depth": self._extract_layer_depth(name),
                        "grad_norm": float(grad_cpu.norm(p=2).item()),
                        "grad_mean": float(grad_cpu.mean().item()),
                        "grad_std": float(grad_cpu.std(unbiased=False).item()),
                        "grad_abs_mean": float(grad_cpu.abs().mean().item()),
                        "numel": int(grad_cpu.numel()),
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
