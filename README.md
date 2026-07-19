<div align="center">

# Intel XPU设备运行Unsloth修复

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch XPU](https://img.shields.io/badge/PyTorch-2.7.1%2Bxpu-red.svg)](https://pytorch.org/)
[![Unsloth](https://img.shields.io/badge/Unsloth-2026.6.9-green.svg)](https://github.com/unslothai/unsloth)

**LoRA fine-tuning on Intel Arc | Windows & WSL2**

[简体中文](#简体中文) | [繁體中文](#繁體中文) | [English](#english)

[ Windows 脚本](#windows) | [ WSL2 脚本](#wsl2)

</div>

---

## 📁 仓库文件说明

### Windows 端（技术验证，基本不可用作生产力）

| 文件 | 状态 | 说明 |
|------|------|------|
| `windows_v7_final.py` | ✅ 成功 | Windows 最终可用版，修复 Triton GCC→MSVC 参数转换、INCLUDE/LIB 路径、C++17、python313.lib、/LD 等 6 个关键 Bug |
| `train_a770_nogui_v7_fixed_v3.py` | ❌ 失败 | 修复 MSVC 头文件路径缺失 |
| `train_a770_nogui_v7_fixed_v4.py` | ❌ 失败 | 修复 SYCL C++17 要求 |
| `train_a770_nogui_v7_fixed_v5.py` | ❌ 失败 | 修复 python313.lib 链接 |
| `train_a770_nogui_v7_fixed_v6.py` | ❌ 失败 | 修复 python313.lib 路径 |
| `train_a770_nogui_v7_fixed_v7.py` | ❌ 失败 | 添加 /LD 创建 DLL |
| `train_a770_nogui_v7_fixed_v8.py` | ✅ 成功 | 设置 TRITON_CACHE_DIR，但没有效果 |

> ⚠️ Windows 原生训练速度 **1523s/it**，GPU无法被利用，**不可实际训练**。

### WSL2 / Linux 端（推荐实际训练）

| 文件 | 状态 | 说明 |
|------|------|------|
| `train_wsl2_final.py` | ⚠️ 旧版 | 早期 WSL2 路径版本，使用 4-bit |
| `train_wsl2_linux_paths.py` | ❌ 失败 | Linux 路径适配，存在 device_map 冲突 |
| `train_wsl2_linux_paths_fixed.py` | ❌ 失败 | 添加显存分配修复，仍有问题 |
| `train_qwen35_4b.py` | ✅ 成功 | Qwen3.5-4B bf16 训练，基础可用版 |
| `train_qwen35_4b_optimized.py` | ✅ 成功 | 可以以正常速度运行的第一个定向脚本 |
| `train_qwen3vl_text_only.py` ~ `v7.py` | ❌ 失败 | Qwen3-VL 文本微调各版本，device_map / meta tensor 冲突 |
| `train_qwen3vl_text_only_v8.py` ~ `v10.py` | ✅ 成功 | 清除 `hf_device_map`、禁用 `fix_untrained_tokens`，最终可用 |
| `wsl_train_v12_interactive.py` | ✅ 成功 | **v12 交互式终端 UI**，方向键选择模型/数据集，推荐日常使用 |

> 🚀 WSL2 训练速度 **11-18s/it**，GPU 成功被利用。

---

## <a id="简体中文"></a>简体中文

<div align="center">

**[Windows 教程](#windows-zh) | [WSL2 教程](#wsl2-zh)**

</div>

### <a id="windows-zh"></a>🖥️ Windows 原生

**适用**：Windows 11 + Intel Arc A770 + 4-bit 量化模型（如 Qwen3-8B-BNB）

**环境要求**：Python 3.13、PyTorch 2.12.1+xpu、VS2022（MSVC v143）、oneAPI 2025.2、Level Zero SDK 1.30

**核心修复**：

- Triton GCC 参数透传 MSVC → 拦截 `_build` 转换参数
- MSVC 找不到 `cstddef` → 自动补全 `INCLUDE`/`LIB`
- SYCL 要求 C++17 → 添加 `/std:c++17` + `/Zc:__cplusplus`
- 找不到 `python313.lib` → 自动添加 `Python313/libs`
- 链接器要求入口点 → 添加 `/LD`
- JIT 编译极慢 → `TRITON_CACHE_DIR` + `TRITON_DISABLE_AUTOTUNE=1`

**运行**：

```powershell
python windows_v7_final.py
```

**配置**（修改脚本顶部 `CONFIG`）：

```python
CONFIG = {
    "model_path": r"H:/Qwen3-8B-unsloth-bnb-4bit",
    "dataset_path": r"D:/dataset.json",
    "output_dir": r"H:/unsloth_train/outputs",
    "max_seq_length": 1024,
    "lora_r": 16, "lora_alpha": 16,
    "learning_rate": 2e-4,
    "batch_size": 1, "grad_accum": 4,
    "max_steps": 3000,
}
```

---

### <a id="wsl2-zh"></a>🐧 WSL2 (Ubuntu 24.04)

**适用**：WSL2 + Intel Arc A770 + 非量化模型 < 16GB（如 Qwen3.5-4B）

**安装**：

```bash
# 1. 系统依赖
sudo apt update
sudo apt install -y libze-dev intel-opencl-icd build-essential python3.12-dev

# 2. 虚拟环境
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# 3. PyTorch XPU（自带 pytorch-triton-xpu，不要单独装 triton）
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu     intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt     pytorch-triton-xpu tcmlib umf intel-pti     --index-url https://download.pytorch.org/whl/xpu

# 4. Unsloth（无依赖安装，跳过 xformers/triton）
pip install --no-deps unsloth unsloth-zoo
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0     peft accelerate bitsandbytes huggingface-hub tokenizers     numpy scipy tqdm regex sentencepiece safetensors psutil packaging
```

> ⚠️ **不要** `pip install triton` / `xformers` / `intel_extension_for_pytorch`
> ⚠️ **不要** 安装完整 oneAPI，如果存在 `/etc/profile.d/oneapi.sh` 请删除

**运行**：

```bash
# 交互式（推荐）
python wsl_train_v12_interactive.py

# 或手动配置
python train_qwen35_4b_optimized.py
```

**关键环境变量**（脚本已内置）：

```bash
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export PYTORCH_XPU_ALLOC_CONF=expandable_segments:True
export IPEX_XPU_ONEDNN_LAYOUT=1
export TRITON_CACHE_DIR=/home/hornstudio/triton_cache
```

**性能对比**：

| 环境 | 模型 | 速度 | GPU 利用率 |
|------|------|------|-----------|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3.5-4B bf16 | **11-15s/it** | **70-85%** |

---

## <a id="繁體中文"></a>繁體中文

<div align="center">

**[Windows 教學](#windows-tw) | [WSL2 教學](#wsl2-tw)**

</div>

### <a id="windows-tw"></a>🖥️ Windows 原生

**適用**：Windows 11 + Intel Arc A770 + 4-bit 量化模型

**環境要求**：Python 3.13、PyTorch 2.12.1+xpu、VS2022（MSVC v143）、oneAPI 2025.2、Level Zero SDK 1.30

**核心修復**：

- Triton GCC 參數透傳 MSVC → 攔截 `_build` 轉換參數
- MSVC 找不到 `cstddef` → 自動補全 `INCLUDE`/`LIB`
- SYCL 要求 C++17 → 新增 `/std:c++17` + `/Zc:__cplusplus`
- 找不到 `python313.lib` → 自動新增 `Python313/libs`
- 連結器要求進入點 → 新增 `/LD`
- JIT 編譯極慢 → `TRITON_CACHE_DIR` + `TRITON_DISABLE_AUTOTUNE=1`

**執行**：

```powershell
python windows_v7_final.py
```

---

### <a id="wsl2-tw"></a>🐧 WSL2 (Ubuntu 24.04)

**適用**：WSL2 + Intel Arc A770 + 非量化模型 < 16GB

**安裝**：

```bash
# 1. 系統依賴
sudo apt update
sudo apt install -y libze-dev intel-opencl-icd build-essential python3.12-dev

# 2. 虛擬環境
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# 3. PyTorch XPU（自帶 pytorch-triton-xpu，不要單獨裝 triton）
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu     intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt     pytorch-triton-xpu tcmlib umf intel-pti     --index-url https://download.pytorch.org/whl/xpu

# 4. Unsloth（無依賴安裝，跳過 xformers/triton）
pip install --no-deps unsloth unsloth-zoo
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0     peft accelerate bitsandbytes huggingface-hub tokenizers     numpy scipy tqdm regex sentencepiece safetensors psutil packaging
```

> ⚠️ **不要** `pip install triton` / `xformers` / `intel_extension_for_pytorch`
> ⚠️ **不要** 安裝完整 oneAPI，如果存在 `/etc/profile.d/oneapi.sh` 請刪除

**執行**：

```bash
# 交互式（推薦）
python wsl_train_v12_interactive.py

# 或手動配置
python train_qwen35_4b_optimized.py
```

**效能比較**：

| 環境 | 模型 | 速度 | GPU 利用率 |
|------|------|------|-----------|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3.5-4B bf16 | **11-15s/it** | **70-85%** |

---

## <a id="english"></a>English

<div align="center">

**[Windows Guide](#windows-en) | [WSL2 Guide](#wsl2-en)**

</div>

### <a id="windows-en"></a>🖥️ Windows Native

**For**: Windows 11 + Intel Arc A770 + 4-bit quantized models

**Requirements**: Python 3.13, PyTorch 2.12.1+xpu, VS2022 (MSVC v143), oneAPI 2025.2, Level Zero SDK 1.30

**Key Fixes**: GCC→MSVC arg translation, `INCLUDE`/`LIB` auto-repair, C++17, `python313.lib`, `/LD` flag, Triton JIT cache

**Run**:

```powershell
python windows_v7_final.py
```

---

### <a id="wsl2-en"></a>🐧 WSL2 (Ubuntu 24.04)

**For**: WSL2 + Intel Arc A770 + non-quantized models < 16GB

**Install**:

```bash
# System deps
sudo apt update
sudo apt install -y libze-dev intel-opencl-icd build-essential python3.12-dev

# Virtual env
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# PyTorch XPU (includes pytorch-triton-xpu, do NOT install triton separately)
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu     intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt     pytorch-triton-xpu tcmlib umf intel-pti     --index-url https://download.pytorch.org/whl/xpu

# Unsloth (no-deps, skip xformers/triton)
pip install --no-deps unsloth unsloth-zoo
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0     peft accelerate bitsandbytes huggingface-hub tokenizers     numpy scipy tqdm regex sentencepiece safetensors psutil packaging
```

> ⚠️ **Do NOT** `pip install triton` / `xformers` / `intel_extension_for_pytorch`
> ⚠️ **Do NOT** install full oneAPI. Remove `/etc/profile.d/oneapi.sh` if exists.

**Run**:

```bash
# Interactive (recommended)
python wsl_train_v12_interactive.py

# Or manual config
python train_qwen35_4b_optimized.py
```

**Performance**:

| Environment | Model | Speed | GPU Utilization |
|-------------|-------|-------|-----------------|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3.5-4B bf16 | **11-15s/it** | **70-85%** |

---

<div align="center">

**Good luck with training! 🎉**

*For actual training, use WSL2/Linux. Windows is for technical verification only.*

</div>
