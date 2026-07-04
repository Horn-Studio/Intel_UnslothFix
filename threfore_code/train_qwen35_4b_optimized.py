import os
import torch
import transformers

# ========== 关键环境变量：Intel XPU 性能优化 ==========
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
os.environ["UR_L0_ENABLE_SYSMAN_ENV_DEFAULT"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
# IPEX_XPU_ONEDNN_LAYOUT=1 是 Intel Arc A770 训练速度的关键优化
# 设置后内存吞吐量大幅提升，训练速度接近 Titan RTX
os.environ["IPEX_XPU_ONEDNN_LAYOUT"] = "1"
# ============================================================================

# ========== 禁用 transformers 缓存预热 ==========
if hasattr(transformers.modeling_utils, 'caching_allocator_warmup'):
    transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
# ============================================================================

# ========== Patch Intel Arc A770 ==========
def _patched_mem_get_info(device=None):
    total = 17179869184
    free = 16106127360  # 15GB，Qwen3.5-4B bf16 约 8GB，全进 GPU
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
    "model_path": "/home/hornstudio/models/Qwen3.5-4B",
    "dataset_path": "/home/hornstudio/data/xiaoh_qa_dataset_10k_expanded.json",
    "output_dir": "/home/hornstudio/unsloth_train/outputs",
    "max_seq_length": 1024,
    "lora_r": 16,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "grad_accum": 2,
    "max_steps": 3000,
    "warmup_steps": 5,
}

os.environ["TRITON_CACHE_DIR"] = "/home/hornstudio/triton_cache"
os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
os.makedirs(CONFIG["output_dir"], exist_ok=True)

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

print(f">>> PyTorch: {torch.__version__}")
print(f">>> XPU: {torch.xpu.is_available()}")

print("\n>>> 正在加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["model_path"],
    max_seq_length=CONFIG["max_seq_length"],
    dtype=torch.bfloat16,
    load_in_4bit=False,
    local_files_only=True,
    trust_remote_code=True,
    device_map="xpu",              # 强制单设备到 XPU
    low_cpu_mem_usage=True,       # 减少 CPU 内存峰值，避免双份加载
)

print(f">>> 模型设备: {next(model.parameters()).device}")
print(f">>> 显存占用: {torch.xpu.memory_allocated() / 1e9:.2f} GB")

model = FastLanguageModel.get_peft_model(
    model,
    r=CONFIG["lora_r"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print(">>> LoRA 配置完成")

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
print(">>> 注意：第一步可能极慢（Triton JIT 编译 SPIR-V），请耐心等待...")
print(">>> 第一步完成后，后续步骤会快很多（编译结果已缓存）")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=CONFIG["max_seq_length"],
    dataset_num_proc=4,
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
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
    ),
)

trainer.train()
print("\n>>> 训练完成！")
model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f">>> 模型已保存到: {CONFIG['output_dir']}")
