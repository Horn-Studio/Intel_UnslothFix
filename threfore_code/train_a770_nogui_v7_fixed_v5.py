import os
import sys
import subprocess
import traceback
import glob
from pathlib import Path

# ==================== Triton 编译参数修复（必须在 import triton 之前） ====================
# Triton Intel XPU backend 在 Windows 上生成 GCC 风格命令行传给 MSVC，导致大量兼容性问题。
# 这里直接替换 triton.runtime.build._build 在 Windows 上的实现，构造正确的 MSVC 命令行。
import triton.runtime.build as triton_build
import tempfile

_orig_build = triton_build._build

def _patched_build(name, src, tmpdir, library_dirs, include_dirs, libraries, ccflags=None):
    if sys.platform != 'win32':
        return _orig_build(name, src, tmpdir, library_dirs, include_dirs, libraries, ccflags)

    cc = os.environ.get('CC', 'cl.exe')

    # 构造 MSVC 风格命令行
    cmd = [cc, src]

    # 基础编译选项：/Zc:__cplusplus 让 MSVC 正确设置 __cplusplus 宏（SYCL 头文件需要）
    cmd.extend(['/nologo', '/O2', '/std:c++17', '/Zc:__cplusplus', '/MD', '/EHsc'])

    # include dirs
    for inc in include_dirs:
        cmd.append(f'/I{inc}')

    # 处理 ccflags：把 GCC 参数转换为 MSVC 参数
    compile_opts = []
    link_opts = []

    if ccflags:
        if isinstance(ccflags, str):
            ccflags = ccflags.split()
        for f in ccflags:
            if f in ('-O3', '-O2', '-O1', '-O0', '-shared'):
                continue
            elif f.startswith('-D'):
                compile_opts.append('/D' + f[2:])
            elif f.startswith('-I'):
                compile_opts.append('/I' + f[2:])
            elif f.startswith('-L'):
                link_opts.append('/LIBPATH:' + f[2:])
            elif f.startswith('-l'):
                link_opts.append(f[2:] + '.lib')
            elif f == '-fPIC':
                pass
            elif f.startswith('-W') or f.startswith('-f'):
                pass
            elif f in ('-Wno-psabi', '-Wno-deprecated-declarations', '-Wno-unknown-pragmas'):
                pass
            elif f.startswith('/LIBPATH:'):
                link_opts.append(f)
            else:
                compile_opts.append(f)

    cmd.extend(compile_opts)

    # 输出文件：Triton 期望 .pyd 文件名格式
    output = os.path.join(tmpdir, f'{name}.cp{sys.version_info.major}{sys.version_info.minor}-win_amd64.pyd')
    cmd.append(f'/Fe{output}')

    # 链接器选项（必须放在 /link 之后）
    cmd.append('/link')
    for lib_dir in library_dirs:
        cmd.append(f'/LIBPATH:{lib_dir}')
    for lib in libraries:
        cmd.append(f'{lib}.lib')
    cmd.extend(link_opts)

    # 执行编译
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stdout) from e

    return output

triton_build._build = _patched_build

# 同时 patch Intel backend 的 compile_module_from_src，确保它使用我们的 _build
try:
    import triton.backends.intel.driver as intel_driver
    if hasattr(intel_driver, '_build'):
        intel_driver._build = _patched_build
except Exception as e:
    print(f">>> Triton Intel backend _build patch 警告: {e}")
# ================================================================================

# ==================== 编码修复（必须在任何其他 import 之前） ====================
_orig_read_text = Path.read_text

def _patched_read_text(self, encoding=None, errors=None, newline=None):
    if self.name == "driver.c" and "triton\\backends\\intel" in str(self):
        encoding = "utf-8"
    return _orig_read_text(self, encoding, errors, newline)

Path.read_text = _patched_read_text
# ================================================================================

# ==================== 统一配置区域 ====================
CONFIG = {
    "model_path": r"H:/Qwen3-8B-unsloth-bnb-4bit",
    "dataset_path": r"D:/xiaoh_qa_dataset_10k_expanded.json",
    "output_dir": r"H:/unsloth_train/outputs",
    "max_seq_length": 1024,
    "lora_r": 16,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "batch_size": 1,
    "grad_accum": 4,
    "max_steps": 100,
    "warmup_steps": 5,
    "trust_remote_code": False,
    "oneapi_setvars_path": r"C:\Program Files (x86)\Intel\oneAPI\setvars.bat",
    "ze_path": r"C:\level-zero-win-sdk-1.30.0",
    "unsloth_cache_dir": r"H:/unsloth_train/cache",
}
# =====================================================


def find_cl_exe():
    possible_patterns = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
    ]
    for pattern in possible_patterns:
        matches = glob.glob(pattern)
        if matches:
            matches.sort()
            return matches[-1]
    return None


def init_msvc_env():
    possible_vcv = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    ]

    for vcv in possible_vcv:
        if os.path.exists(vcv):
            print(f">>> 正在加载 MSVC 环境: {vcv}")
            result = subprocess.run(
                f'cmd /v /c "{vcv}" && echo ===ENV_START=== && set',
                capture_output=True, text=True, shell=True
            )
            in_env = False
            for line in result.stdout.splitlines():
                if "===ENV_START===" in line:
                    in_env = True
                    continue
                if in_env and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ[k] = v
            print(">>> MSVC 环境加载完成")
            break
    else:
        print(">>> 警告: 未找到 vcvars64.bat")

    cl_path = find_cl_exe()
    if cl_path:
        os.environ["CC"] = cl_path
        os.environ["CXX"] = cl_path
        print(f">>> CC = {cl_path}")
    else:
        print(">>> ⚠️ 未找到 cl.exe")
        return False

    cl_dir = os.path.dirname(cl_path)
    if cl_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cl_dir + os.pathsep + os.environ.get("PATH", "")

    return True


def init_intel_env():
    bat_path = CONFIG["oneapi_setvars_path"]

    if os.path.exists(bat_path):
        print(f">>> 正在加载 Intel oneAPI 环境: {bat_path}")
        result = subprocess.run(
            f'cmd /v /c "{bat_path}" && echo ===ENV_START=== && set',
            capture_output=True, text=True, shell=True
        )
        in_env = False
        for line in result.stdout.splitlines():
            if "===ENV_START===" in line:
                in_env = True
                continue
            if in_env and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k] = v
        print(">>> Intel oneAPI 环境加载完成")
    else:
        print(f">>> 警告: 未找到 oneAPI setvars.bat: {bat_path}")

    os.environ["ZE_PATH"] = CONFIG["ze_path"]
    os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "0"

    cache_dir = CONFIG["unsloth_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["UNSLOTH_CACHE_DIR"] = cache_dir

    has_msvc = init_msvc_env()

    print(f">>> ZE_PATH = {CONFIG['ze_path']}")
    print(f">>> ONEDNN_PRIMITIVE_CACHE_CAPACITY = 0")
    print(f">>> UNSLOTH_CACHE_DIR = {cache_dir}")

    # 自动修复：确保 INCLUDE/LIB 包含 MSVC C++ 标准库路径
    if has_msvc and sys.platform == 'win32':
        cl_path = os.environ.get("CC", "")
        if cl_path and os.path.exists(cl_path):
            try:
                # 从 cl.exe 路径推断 MSVC 工具链根目录
                # 例如: .../VC/Tools/MSVC/14.39.33519/bin/Hostx64/x64/cl.exe
                norm_path = os.path.normpath(cl_path)
                parts = norm_path.split(os.sep)
                if 'MSVC' in parts:
                    msvc_idx = parts.index('MSVC')
                    msvc_root = os.sep.join(parts[:msvc_idx + 2])

                    include_paths = []
                    msvc_include = os.path.join(msvc_root, 'include')
                    if os.path.exists(msvc_include):
                        include_paths.append(msvc_include)

                    # 推断 VS 安装根目录，查找 Windows SDK
                    if 'VC' in parts:
                        vc_idx = parts.index('VC')
                        vs_root = os.sep.join(parts[:vc_idx])

                        # 查找 Windows SDK
                        for sdk_base in [
                            r"C:\Program Files (x86)\Windows Kits\10\Include",
                            r"C:\Program Files\Windows Kits\10\Include",
                        ]:
                            if os.path.exists(sdk_base):
                                versions = [d for d in os.listdir(sdk_base) 
                                            if os.path.isdir(os.path.join(sdk_base, d)) and d.startswith('10.0')]
                                if versions:
                                    versions.sort(reverse=True)
                                    for v in versions:
                                        ucrt = os.path.join(sdk_base, v, 'ucrt')
                                        shared = os.path.join(sdk_base, v, 'shared')
                                        um = os.path.join(sdk_base, v, 'um')
                                        if os.path.exists(ucrt):
                                            include_paths.append(ucrt)
                                        if os.path.exists(shared):
                                            include_paths.append(shared)
                                        if os.path.exists(um):
                                            include_paths.append(um)
                                        break

                        # ATL/MFC
                        atl_include = os.path.join(vs_root, 'VC', 'Tools', 'MSVC', parts[msvc_idx + 1], 'atlmfc', 'include')
                        if os.path.exists(atl_include):
                            include_paths.append(atl_include)

                    # 合并到 INCLUDE
                    current_include = os.environ.get('INCLUDE', '')
                    current_paths = [p.strip() for p in current_include.split(os.pathsep) if p.strip()]
                    for p in include_paths:
                        if p not in current_paths:
                            current_paths.append(p)
                    os.environ['INCLUDE'] = os.pathsep.join(current_paths)
                    print(f">>> 已自动修复 INCLUDE 路径，共 {len(current_paths)} 个目录")

                    # 同样修复 LIB 路径
                    lib_paths = []
                    msvc_lib = os.path.join(msvc_root, 'lib', 'x64')
                    if os.path.exists(msvc_lib):
                        lib_paths.append(msvc_lib)

                    # Windows SDK LIB
                    for sdk_base in [
                        r"C:\Program Files (x86)\Windows Kits\10\Lib",
                        r"C:\Program Files\Windows Kits\10\Lib",
                    ]:
                        if os.path.exists(sdk_base):
                            versions = [d for d in os.listdir(sdk_base) 
                                        if os.path.isdir(os.path.join(sdk_base, d)) and d.startswith('10.0')]
                            if versions:
                                versions.sort(reverse=True)
                                for v in versions:
                                    ucrt_lib = os.path.join(sdk_base, v, 'ucrt', 'x64')
                                    um_lib = os.path.join(sdk_base, v, 'um', 'x64')
                                    if os.path.exists(ucrt_lib):
                                        lib_paths.append(ucrt_lib)
                                    if os.path.exists(um_lib):
                                        lib_paths.append(um_lib)
                                    break

                    current_lib = os.environ.get('LIB', '')
                    current_lib_paths = [p.strip() for p in current_lib.split(os.pathsep) if p.strip()]
                    for p in lib_paths:
                        if p not in current_lib_paths:
                            current_lib_paths.append(p)
                    os.environ['LIB'] = os.pathsep.join(current_lib_paths)
                    print(f">>> 已自动修复 LIB 路径，共 {len(current_lib_paths)} 个目录")
            except Exception as e:
                print(f">>> 自动修复 MSVC 路径时出错: {e}")

    if not has_msvc:
        print("\n>>> ❌ 错误: 找不到 C 编译器，无法继续")
        print(">>> 请安装 Visual Studio 2022，勾选 '使用 C++ 的桌面开发'")
        input("按回车退出...")
        sys.exit(1)


def print_config():
    print("\n>>> ===== 训练配置 =====")
    for k, v in CONFIG.items():
        print(f">>> {k}: {v}")
    print(">>> =====================\n")


def main():
    init_intel_env()
    print_config()

    import transformers
    if hasattr(transformers.modeling_utils, 'caching_allocator_warmup'):
        transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        print(">>> 已禁用 transformers caching_allocator_warmup\n")

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    out_dir = CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f">>> 正在加载模型: {CONFIG['model_path']}")
    load_kwargs = {
        "model_name": CONFIG["model_path"],
        "max_seq_length": CONFIG["max_seq_length"],
        "dtype": None,
        "load_in_4bit": True,
    }
    if CONFIG.get("trust_remote_code", False):
        load_kwargs["trust_remote_code"] = True
        print(">>> 已启用 trust_remote_code")

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    print(">>> 模型加载完成")

    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print(">>> LoRA 配置完成")

    ds_path = CONFIG["dataset_path"]
    print(f">>> 正在加载数据集: {ds_path}")

    if os.path.exists(ds_path):
        ext = Path(ds_path).suffix.lower()
        if ext in ('.json', '.jsonl'):
            dataset = load_dataset("json", data_files=ds_path, split="train")
        else:
            dataset = load_dataset(ds_path, split="train")
    else:
        dataset = load_dataset(ds_path, split="train")

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

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names
    )
    print(f">>> 数据集加载完成，共 {len(dataset)} 条\n")

    print(">>> 开始训练...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        dataset_num_proc=1,
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
            output_dir=out_dir,
            save_steps=CONFIG["max_steps"],
            report_to="none",
        ),
    )

    trainer.train()
    print("\n>>> ✅ 训练完成！")

    print("\n>>> ===== 请选择导出方式 =====")
    print("1. 仅保留 LoRA")
    print("2. 导出 16bit 完整权重")
    print("3. 导出 GGUF q4_k_m")
    print("4. 全部导出")
    print("0. 不保存")

    choice = input("\n请输入选项 (0/1/2/3/4): ").strip()

    if choice == "1":
        save_path = os.path.join(out_dir, "lora_adapter")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f">>> ✅ LoRA: {save_path}")

    elif choice == "2":
        save_path = os.path.join(out_dir, "merged_16bit")
        model.save_pretrained_merged(save_path, tokenizer)
        print(f">>> ✅ 16bit: {save_path}")

    elif choice == "3":
        save_path = os.path.join(out_dir, "gguf")
        model.save_pretrained_gguf(save_path, tokenizer, quantization_method="q4_k_m")
        print(f">>> ✅ GGUF: {save_path}")

    elif choice == "4":
        p1 = os.path.join(out_dir, "lora_adapter")
        model.save_pretrained(p1)
        tokenizer.save_pretrained(p1)
        print(f">>> ✅ LoRA: {p1}")

        p2 = os.path.join(out_dir, "merged_16bit")
        model.save_pretrained_merged(p2, tokenizer)
        print(f">>> ✅ 16bit: {p2}")

        p3 = os.path.join(out_dir, "gguf")
        model.save_pretrained_gguf(p3, tokenizer, quantization_method="q4_k_m")
        print(f">>> ✅ GGUF: {p3}")

    else:
        print(">>> 用户选择不保存")

    print("\n>>> 全部完成！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n>>> ❌ 训练出错:")
        traceback.print_exc()
        input("\n按回车键退出...")
        sys.exit(1)
