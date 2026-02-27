import os
import re
import json
from typing import Any

from peft.tuners.boft.config import BOFTConfig
from peft.tuners.prefix_tuning.config import PrefixTuningConfig
import torch
from peft import LoraConfig, LoHaConfig, get_peft_model, PromptEncoderConfig
import ast
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    HfArgumentParser,
    # Qwen2_5_VLForConditionalGeneration,
)
from models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from train.trainer import QwenTrainer
from train.data import (
    EvaluationDataset,
    SupervisedDataset,
    make_supervised_data_module,
    DataCollatorForSupervisedDataset,
)
from train.params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)
from train.eval_utils import MetricsCalculator
import pathlib
from liger_kernel.transformers import (
    apply_liger_kernel_to_qwen2_vl,
    apply_liger_kernel_to_qwen2_5_vl,
)
from monkey_patch_forward import (
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen_2_with_mixed_modality_forward,
)


local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)


def find_target_linear_names(
    model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True
):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)


def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    use_liger = training_args.use_liger
    if "Qwen2.5" in model_args.model_id:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen2_5_with_mixed_modality_forward(use_liger=use_liger)
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            # apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
            from models import modeling_qwen2_5_vl
            from liger_kernel.transformers.qwen2vl_mrope import (
                liger_multimodal_rotary_pos_emb,
            )
            from liger_kernel.transformers.rms_norm import LigerRMSNorm
            from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

            modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = (
                liger_multimodal_rotary_pos_emb
            )
            modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
            modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP
    else:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen_2_with_mixed_modality_forward(use_liger=use_liger)
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False)

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, (
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        )

    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError(
            "If `vision_lora` is True, `freeze_vision_tower` must also be True."
        )

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(
                training_args.lora_namespan_exclude
            )
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    if training_args.modules_to_save is not None:
        training_args.modules_to_save = ast.literal_eval(training_args.modules_to_save)

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["visual", "lm_head"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    if "Qwen2.5" in model_args.model_id:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2"
            if not training_args.disable_flash_attn2
            else "sdpa",
            **bnb_model_from_pretrained_args,
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2"
            if not training_args.disable_flash_attn2
            else "sdpa",
            **bnb_model_from_pretrained_args,
        )

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(
        model_to_configure, training_args, compute_dtype, training_args.device
    )

    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if (
        training_args.lora_method != 'none'
        and training_args.modules_to_save
        and "custom_lora" in training_args.modules_to_save
    ):
        # lora_config = {
        #     "lora_method": training_args.lora_method,
        #     "hidden_size": 2048,
        #     "n_routed_experts": 4,
        #     "n_activated_experts": 2,
        #     "n_shared_experts": 2,
        #     "score_func": "sigmoid",
        #     "route_scale": 1.0,
        #     "use_lfb": True,
        #     "r": 128,
        #     "alpha": 256,
        #     "dropout": 0.1,
        # }
        
        # mamoe
        lora_config = {
            "lora_method": training_args.lora_method,
            "hidden_size": 2048,
            "n_routed_experts": 2,
            "n_activated_experts": 1,
            "n_shared_experts": 1,
            "score_func": "sigmoid",
            "route_scale": 1.0,
            "use_lfb": True,
            "r": 64,
            "alpha": 128,
            "dropout": 0.0,
            "num_modality": 2,
            "use_MEGate": True
        }

        # fusionlora
        # lora_config = {
        #     "lora_method": training_args.lora_method,
        #     "hidden_size": 2048,
        #     "r": 128,
        #     "alpha": 128,
        #     "dropout": 0.1,
        #     "num_modality": 2,
        # }

        if training_args.lora_method == "kradapter":
            import torch.nn as nn
            import math

            class KRAdapterLinearLayer(nn.Module):
                def __init__(self, original_layer, scaling=2):
                    super().__init__()
                    in_features, out_features = original_layer.in_features, original_layer.out_features
                    min_shape, max_shape = min(in_features, out_features), max(out_features, in_features)
                    self.s = (out_features, in_features)
                    self.r = int(math.sqrt(min_shape))
                    self.scaling = scaling
                    self.original_layer = original_layer
                    self.merged = False

                    self.d = 0
                    while self.r * (self.r + self.d) < max_shape:
                        self.d += 1
                    
                    self.min_shape = min_shape
                    self.weight = nn.Parameter(torch.zeros(min_shape, self.r))
                    self.v_weight = nn.Parameter(torch.zeros(min_shape, self.r + self.d))

                    nn.init.kaiming_uniform_(self.v_weight, a=5)

                def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                    if self.merged:
                        return self.original_layer(x)

                    update = self.get_update()
                    output = x @ update * self.scaling + self.original_layer(x)
                    return output

                def get_update(self):
                    update = (self.weight.unsqueeze(1) * self.v_weight.unsqueeze(-1)).view(self.weight.shape[0], -1)
                    if torch.argmin(torch.tensor(self.s)) != torch.argmin(torch.tensor(update.shape)):
                        update = update.T
                    update = update[:self.s[0], :self.s[1]]
                    return update.T

                @torch.no_grad
                def merge(self):
                    if not self.merged:
                        self.original_layer.weight.data += self.get_update().data * self.scaling
                        self.merged = True

            def replace_linear_with_kradapter(model, scaling=2):
                """
                Recursively replace all `torch.nn.Linear` layers in the model with `KRAdapterLinearLayer`.
                """
                targe_module_name = find_target_linear_names(
                    model,
                    lora_namespan_exclude=training_args.lora_namespan_exclude,
                    num_lora_modules=training_args.num_lora_modules,
                )
                for name, module in model.named_modules():
                    if name in targe_module_name and isinstance(module, torch.nn.Linear):
                        kradapter_layer = KRAdapterLinearLayer(module, scaling=scaling)
                        # Replace the module in the model
                        parent_module = model
                        name_parts = name.split('.')
                        for part in name_parts[:-1]:
                            parent_module = getattr(parent_module, part)
                        setattr(parent_module, name_parts[-1], kradapter_layer)
                        rank0_print(f"Replaced Linear layer '{name}' with KRAdapterLinearLayer.")


            replace_linear_with_kradapter(model, scaling=2)
        else:
            for layer in model.model.layers:
                layer.post_init_lora(lora_config)
        
        if training_args.output_dir:
            # save custom lora config to output directory
            pathlib.Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
            with open(
                os.path.join(training_args.output_dir, "custom_lora_config.json"), "w"
            ) as f:
                json.dump(lora_config, f, indent=4)
                
    print(model)
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                lora_namespan_exclude=lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules,
            ),
            modules_to_save=training_args.modules_to_save,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            use_dora=training_args.use_dora,
        )

        # peft_config = BOFTConfig(
        #    boft_block_size=256,
        #    boft_n_butterfly_factor=3,
        #    target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
        #    boft_dropout=0.1,
        #    bias="boft_only",
        # )
        # peft_config = LoHaConfig(
        #     r=training_args.lora_rank,
        #     alpha=training_args.lora_alpha,
        #     target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
        #     rank_dropout=training_args.lora_dropout,
        #     modules_to_save=training_args.modules_to_save,
        # )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model")
        model = get_peft_model(model, peft_config)

        # Peft maodel makes vision tower and merger freezed again.
        # Configuring fuction could be called here, but sometimes it does not work properly.
        # So I just made it this way.
        # Need to be fixed in the future.

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rank0_print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)

            if "lm_head" in name or "embed_token" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(
        model_id=model_args.model_id, processor=processor, data_args=data_args
    )

    compute_metrics_acc = MetricsCalculator(processor, False)
    print(model)
    trainer = QwenTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=compute_metrics_acc,
        **data_module,
    )

    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    # if training_args.lora_enable:
    #     state_dict = get_peft_state_maybe_zero_3(
    #         model.named_parameters(), training_args.lora_bias
    #     )

    #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
    #         model.named_parameters(), require_grad_only=True
    #     )

    #     if local_rank == 0 or local_rank == -1:
    #         model.config.save_pretrained(training_args.output_dir)
    #         model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    #         torch.save(
    #             non_lora_state_dict,
    #             os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
    #         )
    # else:
    #     safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

    if training_args.do_final_eval:
        trainer.args.per_device_eval_batch_size = 8
        test_dataset = EvaluationDataset(
            data_path=data_args.test_data_path,
            processor=processor,
            data_args=data_args,
            model_id=model_args.model_id,
        )
        trainer.data_collator = DataCollatorForSupervisedDataset(
            pad_token_id=processor.tokenizer.pad_token_id,
            image_token_id=151655,
            # image_token_id=processor.tokenizer.encode(processor.image_token)[0]
            padding_side="left",
        )
        result = trainer.generate_eval_save(
            test_dataset=test_dataset,
            processor=processor,
            test_data_path=data_args.test_data_path,
            output_json_path=os.path.join(
                training_args.output_dir, "testset_result.json"
            ),
        )
        rank0_print("Final evaluation result: ", result)


if __name__ == "__main__":
    train()
