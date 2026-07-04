import os
import torch
import transformers

# ========== 关键：解除 Intel XPU 4GB 单次分配限制 ==========
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
os.environ["UR_L0_ENABLE_SYSMAN_ENV_DEFAULT"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
# ============================================================================

# ========== 绕过 accelerate device_map 训练检测 ==========
os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"
# ============================================================================

# ========== 禁用 transformers 缓存预热 ==========
if hasattr(transformers.modeling_utils, 'caching_allocator_warmup'):
    transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
# ============================================================================

# ========== Patch Intel Arc A770 ==========
def _patched_mem_get_info(device=None):
    total = 17179869184
    free = 15032385536  # 14GB，让 accelerate 尽量把模型往 GPU 塞
    return (free, total)

torch.xpu.memory.mem_get_info = _patched_mem_get_info
torch.xpu.mem_get_info = _patched_mem_get_info

_orig_get_device_properties = torch.xpu.get_device_properties
def _patched_get_device_properties(device=None):
    try:
        return _orig_get_device_properties(device)
    except Exception:
        class FakeProps:
            name = 'Intel(R) Arc(TM) A770 Graphics'
            total_memory = 17179869184
            max_compute_units = 512
            gpu_eu_count = 512
            has_fp16 = 1
            has_fp64 = 0
        return FakeProps()

torch.xpu.get_device_properties = _patched_get_device_properties
# ============================================================================

CONFIG = {
    "model_path": "/home/hornstudio/models/Qwen3-VL-8B-Instruct",
    "dataset_path": "/home/hornstudio/data/xiaoh_qa_dataset_10k_expanded.json",
    "output_dir": "/home/hornstudio/unsloth_train/outputs",
    "max_seq_length": 512,
    "lora_r": 16,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "batch_size": 1,
    "grad_accum": 8,
    "max_steps": 3000,
    "warmup_steps": 5,
}

os.environ["TRITON_CACHE_DIR"] = "/home/hornstudio/triton_cache"
os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
os.makedirs(CONFIG["output_dir"], exist_ok=True)

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ========== 禁用 Unsloth fix_untrained_tokens（必须在 import unsloth 之后） ==========
import unsloth_zoo.tokenizer_utils
unsloth_zoo.tokenizer_utils.fix_untrained_tokens = lambda *args, **kwargs: None
# ============================================================================

print(f">>> PyTorch: {torch.__version__}")
print(f">>> XPU: {torch.xpu.is_available()}")

print("\n>>> 正在加载模型...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=CONFIG["model_path"],
    max_seq_length=CONFIG["max_seq_length"],
    dtype=torch.bfloat16,
    load_in_4bit=False,
    local_files_only=True,
    trust_remote_code=True,
)

# 不强制 to("xpu")，让 accelerate 保持 CPU/GPU 混合调度

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=CONFIG["lora_r"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print(">>> LoRA 配置完成（视觉层已冻结）")

print(">>> 正在加载数据集...")
dataset = load_dataset("json", data_files=CONFIG["dataset_path"], split="train")

alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples.get("instruction", examples.get("text", []))
    inputs = examples.get("input", [""] * len(instructions))
    outputs = examples.get("output", [""] * len(instructions))
    texts = []
    for instruction, inp, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, inp if inp else "", output)
        text += tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)
print(f">>> 数据集加载完成，共 {len(dataset)} 条")

print("\n>>> 开始训练...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=CONFIG["max_seq_length"],
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        warmup_steps=CONFIG["warmup_steps"],
        max_steps=CONFIG["max_steps"],
        learning_rate=CONFIG["learning_rate"],
        fp16=False,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=CONFIG["output_dir"],
        save_steps=CONFIG["max_steps"],
        report_to="none",
    ),
)

trainer.train()
print("\n>>> 训练完成！")
model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f">>> 模型已保存到: {CONFIG['output_dir']}")
