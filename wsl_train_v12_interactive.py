#!/usr/bin/env python3
"""
Intel Arc A770 + Unsloth 交互式微调脚本 v12
支持方向键选择模型/数据集，交互式修改参数
"""
import os
import sys
import glob
import curses

# ========== 扫描目录 ==========
MODELS_DIR = "/home/hornstudio/models/"
DATA_DIR = "/home/hornstudio/data/"
BASE_OUTPUT_DIR = "/home/hornstudio/unsloth_train/outputs"


def scan_models():
    """扫描模型目录"""
    if not os.path.exists(MODELS_DIR):
        return []
    dirs = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    return sorted(dirs)


def scan_datasets():
    """扫描数据集目录"""
    if not os.path.exists(DATA_DIR):
        return []
    files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    return sorted([os.path.basename(f) for f in files])


def select_item(stdscr, title, items, hint=""):
    """方向键选择列表项"""
    if not items:
        return None
    current = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, title, curses.A_BOLD | curses.color_pair(1))
        if hint:
            stdscr.addstr(1, 0, hint, curses.A_DIM)

        for i, item in enumerate(items):
            y = 3 + i
            if y >= h - 2:
                break
            prefix = "> " if i == current else "  "
            attr = curses.A_REVERSE if i == current else curses.A_NORMAL
            stdscr.addstr(y, 2, prefix + item, attr)

        stdscr.addstr(h-1, 0, "↑↓选择 | 回车确认 | ESC取消", curses.A_DIM)
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP and current > 0:
            current -= 1
        elif key == curses.KEY_DOWN and current < len(items) - 1:
            current += 1
        elif key in (10, 13):
            return items[current]
        elif key == 27:
            return None


def edit_params(stdscr):
    """交互式编辑参数"""
    params = {
        "learning_rate": 2e-4,
        "batch_size": 4,
        "max_steps": 3000,
        "max_seq_length": 1024,
    }
    keys = list(params.keys())
    current = 0
    editing = False
    buffer = ""

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, "参数设置", curses.A_BOLD | curses.color_pair(1))
        stdscr.addstr(1, 0, "* 其他参数（LoRA rank、grad_accum等）请修改脚本代码", curses.A_DIM)
        stdscr.addstr(2, 0, "回车编辑 | ↑↓切换 | q开始训练 | ESC取消", curses.A_DIM)

        for i, k in enumerate(keys):
            y = 4 + i
            val = params[k]
            if i == current and not editing:
                stdscr.addstr(y, 2, f"> {k}: {val}", curses.A_REVERSE)
            elif i == current and editing:
                stdscr.addstr(y, 2, f"> {k}: {buffer}", curses.A_REVERSE | curses.color_pair(2))
            else:
                stdscr.addstr(y, 2, f"  {k}: {val}")

        stdscr.refresh()

        if editing:
            key = stdscr.getch()
            if key in (10, 13):
                try:
                    old = params[keys[current]]
                    if isinstance(old, int):
                        params[keys[current]] = int(buffer)
                    elif isinstance(old, float):
                        params[keys[current]] = float(buffer)
                except:
                    pass
                editing = False
                buffer = ""
            elif key in (127, 8, 263):
                buffer = buffer[:-1]
            elif 32 <= key <= 126:
                buffer += chr(key)
        else:
            key = stdscr.getch()
            if key == curses.KEY_UP and current > 0:
                current -= 1
            elif key == curses.KEY_DOWN and current < len(keys) - 1:
                current += 1
            elif key in (10, 13):
                editing = True
                buffer = str(params[keys[current]])
            elif key in (ord('q'), ord('Q')):
                return params
            elif key == 27:
                return None


def confirm_screen(stdscr, model, dataset, output, params):
    """确认界面"""
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "配置确认", curses.A_BOLD | curses.color_pair(1))
        stdscr.addstr(2, 0, f"模型: {model}")
        stdscr.addstr(3, 0, f"数据集: {dataset}")
        stdscr.addstr(4, 0, f"输出: {output}")
        stdscr.addstr(5, 0, f"learning_rate: {params['learning_rate']}")
        stdscr.addstr(6, 0, f"batch_size: {params['batch_size']}")
        stdscr.addstr(7, 0, f"max_steps: {params['max_steps']}")
        stdscr.addstr(8, 0, f"max_seq_length: {params['max_seq_length']}")
        stdscr.addstr(10, 0, "按回车开始训练，按 ESC 返回修改...", curses.A_DIM)
        stdscr.refresh()
        key = stdscr.getch()
        if key in (10, 13):
            return True
        elif key == 27:
            return False


def interactive_ui(stdscr):
    """主交互界面"""
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)

    # 1. 选择模型
    models = scan_models()
    if not models:
        stdscr.addstr(0, 0, f"错误: {MODELS_DIR} 下没有模型文件夹", curses.A_BOLD)
        stdscr.refresh()
        stdscr.getch()
        return None
    selected_model = select_item(stdscr, "选择模型文件夹", models, f"扫描目录: {MODELS_DIR}")
    if selected_model is None:
        return None
    model_path = os.path.join(MODELS_DIR, selected_model)

    # 2. 选择数据集
    datasets = scan_datasets()
    if not datasets:
        stdscr.addstr(0, 0, f"错误: {DATA_DIR} 下没有 .json 文件", curses.A_BOLD)
        stdscr.refresh()
        stdscr.getch()
        return None
    selected_dataset = select_item(stdscr, "选择数据集 (.json)", datasets, f"扫描目录: {DATA_DIR}")
    if selected_dataset is None:
        return None
    dataset_path = os.path.join(DATA_DIR, selected_dataset)

    # 3. 编辑参数
    params = edit_params(stdscr)
    if params is None:
        return None

    # 4. 确认
    output_dir = os.path.join(BASE_OUTPUT_DIR, selected_model)
    if not confirm_screen(stdscr, model_path, dataset_path, output_dir, params):
        return None

    return {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "params": params,
        "model_name": selected_model,
    }


def run_training(config):
    """执行训练（v11 核心代码）"""
    import torch
    import transformers

    model_path = config["model_path"]
    dataset_path = config["dataset_path"]
    output_dir = config["output_dir"]
    p = config["params"]

    # ========== 环境变量 ==========
    os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
    os.environ["UR_L0_ENABLE_SYSMAN_ENV_DEFAULT"] = "0"
    os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["IPEX_XPU_ONEDNN_LAYOUT"] = "1"

    # ========== 禁用 transformers 缓存预热 ==========
    if hasattr(transformers.modeling_utils, 'caching_allocator_warmup'):
        transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

    # ========== Patch Intel Arc A770 ==========
    def _patched_mem_get_info(device=None):
        total = 17179869184
        free = 16106127360
        return (free, total)
    torch.xpu.memory.mem_get_info = _patched_mem_get_info
    torch.xpu.mem_get_info = _patched_mem_get_info

    _orig = torch.xpu.get_device_properties
    def _patched(device=None):
        try:
            return _orig(device)
        except Exception:
            class FakeProps:
                name = 'Intel(R) Arc(TM) A770 Graphics'
                total_memory = 17179869184
                max_compute_units = 512
                gpu_eu_count = 512
                has_fp16 = 1
                has_fp64 = 0
            return FakeProps()
    torch.xpu.get_device_properties = _patched

    # ========== 目录准备 ==========
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("/home/hornstudio/triton_cache", exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = "/home/hornstudio/triton_cache"
    os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    print(f">>> PyTorch: {torch.__version__}")
    print(f">>> XPU: {torch.xpu.is_available()}")
    print(f">>> 模型: {model_path}")
    print(f">>> 数据集: {dataset_path}")
    print(f">>> 输出: {output_dir}")

    print("\n>>> 正在加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=p["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
        local_files_only=True,
        trust_remote_code=True,
        device_map="xpu",
        low_cpu_mem_usage=True,
    )

    print(f">>> 模型设备: {next(model.parameters()).device}")
    print(f">>> 显存占用: {torch.xpu.memory_allocated() / 1e9:.2f} GB")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print(">>> LoRA 配置完成")

    print(">>> 正在加载数据集...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

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
        max_seq_length=p["max_seq_length"],
        dataset_num_proc=4,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=p["batch_size"],
            gradient_accumulation_steps=2,
            warmup_steps=5,
            max_steps=p["max_steps"],
            learning_rate=p["learning_rate"],
            fp16=False,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_steps=p["max_steps"],
            report_to="none",
            dataloader_num_workers=2,
            dataloader_pin_memory=False,
        ),
    )

    trainer.train()
    print("\n>>> 训练完成！")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f">>> 模型已保存到: {output_dir}")


if __name__ == "__main__":
    config = curses.wrapper(interactive_ui)
    if config:
        run_training(config)
    else:
        print(">>> 已取消")
