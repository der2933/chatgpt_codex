from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments
from trl import DPOConfig as DPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_method: str = "none"
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    modules_to_save: str = field(default=None, metadata={"help": "List of modules to train and save"})
    num_lora_modules: int = -1
    use_liger: bool = True
    do_final_eval: bool = False
    enable_gradient_logging: bool = field(
        default=False,
        metadata={"help": "Enable example-level gradient logging for transformer block adapters."},
    )
    gradient_log_every_n_steps: int = field(
        default=50,
        metadata={"help": "Log gradients every N training steps."},
    )
    gradient_log_max_examples_per_step: int = field(
        default=1,
        metadata={"help": "Maximum number of examples logged per selected step."},
    )
    gradient_log_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to write gradient jsonl records. Defaults to output_dir/gradient_logs.jsonl."},
    )
    gradient_log_save_full_grad: bool = field(
        default=False,
        metadata={"help": "Whether to dump full gradient vectors to .npy files for selected adapter parameters."},
    )
    gradient_log_full_grad_max_params: int = field(
        default=8,
        metadata={"help": "Maximum number of adapter parameters per partition/step to dump full gradient vectors for. <=0 means all."},
    )
    gradient_log_full_grad_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store dumped full gradient vectors (.npy). Defaults to output_dir/gradient_vectors."},
    )
    enable_hidden_state_logging: bool = field(
        default=False,
        metadata={"help": "Enable token hidden-state logging by layer for modality-space analysis."},
    )
    hidden_state_log_every_n_steps: int = field(
        default=200,
        metadata={"help": "Log token hidden states every N training steps."},
    )
    hidden_state_log_max_tokens_per_modality: int = field(
        default=64,
        metadata={"help": "Maximum sampled tokens per modality (text/image) per layer when hidden-state logging is enabled."},
    )
    hidden_state_log_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to write hidden-state jsonl records. Defaults to output_dir/hidden_state_logs.jsonl."},
    )
    hidden_state_vector_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store hidden-state token vectors (.npy). Defaults to output_dir/hidden_state_vectors."},
    )

@dataclass
class DPOArguments(DPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger: bool = True
    beta: float = field(
        default=0.1,
        metadata={"help": "The beta value for DPO."}
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={"help": "Whether to precompute the reference log probabilities."}
    )
    dpo_loss:str = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    test_data_path: str = field(
        default=None, metadata={"help": "Path to the test data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: float = 1.0
