<div align="center">

**[简体中文](#简体中文) | [繁體中文](#繁體中文) | [English](#english) | [日本語](#日本語) | [Русский](#русский)**

</div>

---

# <a id="简体中文"></a>简体中文

# Intel Arc A770 + Unsloth 双端微调指南

> **Windows 端**：Windows 11 上使用 Intel Arc A770 16GB 对 4bit 量化模型（全量模型大概率也可以，此处选择 Qwen3-8b-bnb-4bit）进行 LoRA 微调。由于 Triton 的 Intel XPU backend 在 Windows 上未经完整测试，直接运行会产生大量 MSVC/GCC 兼容性问题。本仓库的一部分旨在提供一个解决方案。
>
> **WSL2/Linux 端**：原权重 < 16GB 的非量化模型（如 Qwen3.5-4B、Qwen3-1.7B 等），本指南基于 Intel Arc A770 16GB 作为设备，Qwen3.5-4B 作为模型。

---

# Intel Arc A770 + Unsloth + Windows 微调脚本

> **适用场景**：Windows 11 上使用 Intel Arc A770 16GB 对 4bit 量化模型（全量模型大概率也可以，此处选择 Qwen3-8b-bnb-4bit） 进行 LoRA 微调
> 
> 由于 Triton 的 Intel XPU backend 在 Windows 上未经完整测试，直接运行会产生大量 MSVC/GCC 兼容性问题。本仓库的一部分旨在提供一个解决方案

---

## 一、功能

- 在 **Windows 11** 上使用 **Intel Arc A770** 进行 **Qwen3-8B-BNB（仅为示例，其他模型自行测试）** 模型的 Unsloth LoRA 微调
- 自动检测并加载 Intel oneAPI + MSVC 编译环境
- 自动修复 Triton 在 Windows MSVC 下的 GCC 参数兼容性问题
- 自动修复 MSVC 头文件/库文件路径缺失问题
- 支持训练完成后导出 LoRA / 16bit 完整权重 / GGUF（由于没有跑完过所以最后是否能够转完整权重和 ggud 还未知，自行有耐心测试）

---

## 二、前提安装

### 硬件
- **GPU**: Intel Arc 独显（核心显卡理论可行未测试，B系列理论可行未测试）
- **系统**: Windows 11 

### 软件

| 组件 | 版本/要求 | 用途 |
|------|----------|------|
| Intel Arc 显卡驱动 | 最新版 (31.0.101.xxx+) | GPU 计算 |
| Intel oneAPI Base Toolkit 和 DeepLearning Toolkit | 2025.2 和最新版 | SYCL / Level Zero 运行时 |
| Level Zero SDK | 1.28.x - 1.30.x | Triton XPU backend |
| Visual Studio 2022 | Community/Professional/Enterprise | MSVC C++ 编译器 |
| Python | 3.13 (Windows 版) | 运行环境 |
| PyTorch | 2.12.1+xpu (Intel 官方 wheel) | XPU 深度学习框架 |
| Unsloth | 2026.6.9 | 快速微调框架 |

### VS 2022 必须安装的工作负载
- **"使用 C++ 的桌面开发"**
- **MSVC v143 - VS 2022 C++ x64/x86 生成工具**
- **Windows 11 SDK**

---

## 三、适用模型

| 模型 | 格式 | 状态 |
|------|------|------|
| Qwen3-8B-BNB | 4-bit BNB (unsloth 预量化) | ✅ 已验证可行但速度极慢 |

> 其他模型未经测试。理论上只要是 Unsloth 支持的、通过 BNB 4-bit 加载的模型均可使用，但可能需要额外调整。

---

## 四、为 Intel Arc 修复的 Bug

### 1. Triton GCC 参数透传给 MSVC 导致 D8021 错误
**现象**: `cl: 命令行 error D8021 :无效的数值参数"/Wno-psabi"`  
**原因**: Triton Intel XPU backend 按 GCC 风格生成编译命令，直接传给 `cl.exe`  
**修复**: 拦截 `triton.runtime.build._build`，过滤 `-Wno-psabi`、`-Wno-deprecated-declarations`、`-fPIC` 等 GCC 参数，并将 `-D`/`-I`/`-L`/`-l`/`-shared` 转换为 MSVC 风格 `/D`/`-I`/`-LIBPATH:`/`lib`/`/LD`

### 2. MSVC 找不到 C++ 标准库头文件（`cstddef` 等）
**现象**: `fatal error C1083: 无法打开包括文件: "cstddef"`  
**原因**: `vcvars64.bat` 只设置了 `PATH`，没设置 `INCLUDE` 和 `LIB` 环境变量  
**修复**: 从 `cl.exe` 路径自动推断 MSVC 工具链根目录，补全 `INCLUDE`（MSVC include + Windows SDK ucrt/shared/um + ATL/MFC）和 `LIB`（MSVC lib/x64 + Windows SDK lib）

### 3. SYCL 头文件要求 C++17
**现象**: `error C2338: static_assert failed: 'DPCPP does not support C++ version earlier than C++17.'`  
**原因**: MSVC 默认 C++14，SYCL 头文件用 `__cplusplus` 宏检查版本  
**修复**: 编译命令添加 `/std:c++17` 和 `/Zc:__cplusplus`（后者让 MSVC 正确设置 `__cplusplus` 宏为 `201703L`）

### 4. 链接时找不到 `python313.lib`
**现象**: `LINK : fatal error LNK1104: 无法打开文件"python313.lib"`  
**原因**: Triton 编译 Python 扩展（`.pyd`）时，`library_dirs` 只包含 `Library/bin` 和 `Library/lib`，没有 `libs`  
**修复**: 自动检测 `Python313/libs` 目录并添加到 `/LIBPATH`

### 5. 链接器要求入口点（缺少 `/LD`）
**现象**: `LINK : fatal error LNK1561: 必须定义入口点`  
**原因**: `.pyd` 本质是 DLL，需要 `/LD` 标志，但编译命令中没有  
**修复**: 在编译命令中添加 `/LD`（创建 DLL）

### 6. Triton JIT 编译极慢（缓存不生效）
**现象**: 训练第一步耗时 20+ 分钟，GPU 利用率接近 0%  
**原因**: Windows 上 Triton Intel XPU backend 的 JIT kernel 缓存机制有问题，每次 step 都可能重新编译 SPIR-V  
**缓解**: 设置 `TRITON_CACHE_DIR` 和 `TRITON_DISABLE_AUTOTUNE=1` 减少重复编译开销

---

## 五、仍存问题

- **训练速度极慢**: 即使修复了编译问题，Windows 上 Triton XPU backend 的 JIT kernel 执行效率远低于 Linux，GPU 利用率长期低于 10%，单步训练仍需数分钟至数十分钟
- **Triton 缓存不完全可靠**: `TRITON_CACHE_DIR` 有时无法命中，导致同一 kernel 多次重新编译
- **Level Zero SDK 版本不一致**: 环境变量 `ZE_PATH` 指向 1.30.0，但 Triton 编译命令中可能出现 1.28.2 路径，需手动统一
- **xformers 不支持**: Intel XPU 无法使用 xformers（仅 CUDA），Unsloth 的部分 FlashAttention 优化失效
- **mem_get_info 跨平台差异**: `torch.xpu.memory.mem_get_info()` 在 Windows Intel Arc 驱动上可用，但在 WSL2/Linux 上不可用（跨平台脚本需注意）

---

## 六、版本说明（Release）

第八个版本才是能够开始跑起来的模型，前七个版本都没有完整修复 bug 使其正常开始调试

---

## 七、快速开始

```powershell
# 1. 确保已安装所有前提软件（见上文）
# 2. 下载 Release 并解压
# 3. 修改脚本顶部的 CONFIG 区域（模型路径、数据集路径等）
# 4. 运行
```

---

## 八、配置说明

```python
CONFIG = {
    "model_path": r"H:/Qwen3-8B-unsloth-bnb-4bit",      # 模型路径
    "dataset_path": r"D:/dataset.json",                  # 数据集路径
    "output_dir": r"H:/unsloth_train/outputs",          # 输出目录
    "max_seq_length": 1024,
    "lora_r": 16,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "batch_size": 1,        # 根据显存调整
    "grad_accum": 4,        # 总 batch = batch_size * grad_accum
    "max_steps": 3000,
    "warmup_steps": 5,
}
```

---

## 九、微小故障排查

| 出现的问题 | 现象 | 解决方案 |
|---|---|---|
| **Triton GCC 参数报错** | `D8021 :无效的数值参数"/Wno-psabi"` | 使用 v2+ 版本，已过滤 GCC 参数 |
| **找不到 C++ 头文件** | `fatal error C1083: "cstddef"` | 使用 v4+ 版本，自动修复 INCLUDE |
| **SYCL 要求 C++17** | `DPCPP does not support C++ version earlier than C++17` | 使用 v5+ 版本，已添加 `/std:c++17` |
| **找不到 python313.lib** | `LNK1104: 无法打开文件"python313.lib"` | 使用 v6+ 版本，自动添加 Python libs |
| **必须定义入口点** | `LNK1561: 必须定义入口点` | 使用 v7+ 版本，已添加 `/LD` |
| **训练第一步极慢** | 20+ 分钟，GPU 利用率 0% | 使用 v8 版本，设置 `TRITON_CACHE_DIR`；若仍极慢，建议迁移到 WSL2/Linux |
| **Windows 原生训练不可接受** | 1523s/it，GPU 利用率 6% | **必须迁移到 WSL2/Linux**，Windows 上 Triton XPU backend 未经优化 |
| **Level Zero 版本不一致** | 编译命令中出现不同版本路径 | 统一环境变量 `ZE_PATH` 与实际安装的 SDK 版本 |

---

## 十、性能对比

| 环境 | 模型 | 速度 | GPU 利用率 |
|---|---|---|---|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3-8B bnb-4bit | 11-15s/it | 70-85% |

> **结论**: Windows 上只能解决"能跑"的问题，无法解决"快"的问题。如需实际训练，强烈建议迁移到 WSL2/Linux。

---

## 十一、一键重建脚本（Windows 环境检查）

```powershell
# 检查必要环境变量
$env:ZE_PATH
$env:CC

# 检查 VS 2022 安装
Test-Path "C:\Program Files\Microsoft Visual Studio2\Community\VC\Auxiliary\Buildcvars64.bat"

# 检查 Python 版本
python --version  # 应为 3.13

# 检查 PyTorch XPU
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"

# 检查 Triton
python -c "import triton; print(triton.__version__)"
```

---

## 十二、未来预计更新的功能

1. **添加终端版图形化**：提升易用性
2. **最后导出 merged 与 gguf**：导出待测试
3. **Windows 速度优化**：等待 Intel/Triton 官方修复 XPU backend 在 Windows 上的性能或尝试移植包

---

> **最后**：祝训练胜利！如需实际训练，请移步 WSL2/Linux

---

---

# Intel Arc A770 + Unsloth + WSL2 微调模型脚本

> **适用场景**：原权重 < 16GB 的非量化模型（如 Qwen3.5-4B、Qwen3-1.7B 等），本指南基于 Intel Arc A770 16GB 作为设备，Qwen3.5-4B 作为模型
> 

---

## 一、硬件/环境要求

- **GPU**: Intel Arc 独显设备（核心显卡为测试请自查，B系列理论可行但未测试）
- **OS**: Windows 11 21H2+，开启 WSL2
- **WSL2 发行版**: Ubuntu 24.04 (Noble) ，22.04 的 Intel GPU 驱动包名和仓库路径不同，26.04 的 Python 版本过高不适用
- **Python**: 3.12


---

## 二、WSL2 Ubuntu 24.04 安装

在 Windows PowerShell（管理员）中执行：

```powershell
# 更新 WSL
wsl --update

# 安装 Ubuntu 24.04
wsl --install Ubuntu-24.04
wsl --set-default Ubuntu-24.04
#如果提示找不到那就是微软服务器间歇抽风，请通过 store rg 下载 Ubuntu 安装包自行安装

# 验证版本
wsl --list --verbose
# 应显示 Ubuntu-24.04 Running 版本 2
```

---

## 三、Intel GPU 驱动和运行时配置

进入 WSL2 Ubuntu 24.04 终端，执行（如果出现缺失包等问题请自行 sudo apt 安装或者询问 AI）：

```bash
# 1. 更新系统
sudo apt update && sudo apt upgrade -y

# 2. 安装基础工具
sudo apt install -y gpg-agent wget build-essential python3.12-dev

# 3. 添加 Intel GPU 仓库（Noble 版本）
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified' | \
  sudo tee /etc/apt/sources.list.d/intel.gpu.noble.list

sudo apt update

# 4. 安装 Intel GPU 运行时（关键包）
sudo apt install -y libze-dev intel-opencl-icd intel-media-va-driver-non-free \
  libmfx1 libvpl2 libegl-mesa0 libegl1-mesa-dev libgbm1 libgl1-mesa-dev \
  libgl1-mesa-dri libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 \
  libxatracker2 mesa-va-drivers mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all

# 5. 将用户加入 render 组（GPU 访问权限）
sudo gpasswd -a ${USER} render
newgrp render

# 6. 验证 GPU 可见性
ls /dev/dri
# 应看到 renderD128 和 card0

clinfo | grep "Device Name"
# 应显示 Intel(R) Arc(TM) A770 Graphics 或者 0x5860 之类
```

> **⚠️绝对注意**：
> - 不要装完整版 oneAPI Base Toolkit（会污染 LD_LIBRARY_PATH，导致 PyTorch 库冲突）
> - 如果之前装过 oneAPI 并配置了 /etc/profile.d/oneapi.sh，**务必删除**：
>   ```bash
>   sudo rm /etc/profile.d/oneapi.sh
>   ```
> - 如果 sycl-ls 后来因版本冲突坏了，**不影响 PyTorch 训练**，不用管。

---

## 四、PyTorch XPU 环境安装（虚拟环境名为 unsloth_env）

```bash
# 1. 创建虚拟环境
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# 2. 升级 pip
pip install --upgrade pip setuptools wheel

# 3. 安装 PyTorch XPU 完整栈（自带 pytorch-triton-xpu，不要单独装 triton）
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu \
    intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt \
    pytorch-triton-xpu tcmlib umf intel-pti \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple

# 4. 验证 PyTorch XPU
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.is_available())"

# 5. 验证 Triton XPU（正确的验证方式）
python -c "
import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

x = torch.rand(128, device='xpu')
out = torch.empty_like(x)
n_elements = x.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
test_kernel[grid](x, out, n_elements, BLOCK_SIZE=128)
print('Triton XPU test passed!')
"
```

> **⚠️注意事项**：
> - **不要** `pip install triton`（会覆盖 pytorch-triton-xpu，导致 Intel XPU backend 丢失，且安装的是通用版 triton 不包含 xpu 算子支持）
> - **不要** `pip install xformers`（只支持 CUDA，会连带安装 NVIDIA 驱动）
> - **不要** `pip install intel_extension_for_pytorch`（PyTorch 2.7.1+xpu 已有原生 XPU 支持，IPEX 会引入版本冲突）

---

## 五、Unsloth 安装

```bash
source ~/unsloth_env/bin/activate

# 1. 安装 Unsloth（必须保持无依赖安装）
pip install --no-deps unsloth unsloth-zoo

# 2. 手动安装 Unsloth 的其他依赖（跳过 xformers 和 triton）
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0 \
    cut_cross_entropy hf_transfer msgspec torchao tyro diffusers \
    nest-asyncio pydantic peft accelerate bitsandbytes \
    huggingface-hub tokenizers protobuf numpy scipy tqdm regex \
    sentencepiece safetensors psutil packaging
```

---

## 六、该脚本做出的修复

### torch.xpu.memory.mem_get_info() 不支持
PyTorch issue #164057，Arc A770 WSL2/Linux 驱动未实现此 API。
**修复**：monkey-patch 返回固定值。

### torch.xpu.get_device_properties() 可能崩溃
**修复**：异常时返回 FakeProps。

### Intel XPU 在 WSL2 下缺失显存分配函数
**修复**：设置环境变量 `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` 和 `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True`。

### transformers caching_allocator_warmup 导致 OOM
**修复**：在 `import unsloth` 之前禁用。

### Triton JIT 编译慢（Intel XPU 通病）
**修复**：设置 `TRITON_CACHE_DIR` 缓存编译结果，设置 `IPEX_XPU_ONEDNN_LAYOUT=1` 加速内存吞吐量。

### Unsloth fix_untrained_tokens 与 meta tensor 冲突
**修复**：禁用该函数。

---

## 七、完整训练代码（v11 优化版）

已提供于 release 或仓库中，请自行查找，最终可用版本为 v12，前 11 个版本都没有完整修复上述问题

---

## 八、一键重建脚本

如果你搞炸了环境，直接运行这个脚本重建：

```bash
set -e

echo ">>> 开始重建环境..."

# 1. 删除旧环境
rm -rf ~/unsloth_env

# 2. 创建新环境
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# 3. 升级 pip
pip install --upgrade pip setuptools wheel

# 4. 安装 PyTorch XPU 
echo ">>> 安装 PyTorch XPU..."
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu \
    intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt \
    pytorch-triton-xpu tcmlib umf intel-pti \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple

# 5. 验证 PyTorch XPU
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.is_available())"

# 6. 验证 Triton XPU
echo ">>> 验证 Triton XPU..."
python -c "
import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

x = torch.rand(128, device='xpu')
out = torch.empty_like(x)
n_elements = x.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
test_kernel[grid](x, out, n_elements, BLOCK_SIZE=128)
print('Triton XPU test passed!')
"

# 7. 安装 Unsloth（不覆盖 PyTorch）
echo ">>> 安装 Unsloth..."
pip install --no-deps unsloth unsloth-zoo

# 8. 手动安装其他依赖（跳过 xformers 和 triton）
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0 \
    cut_cross_entropy hf_transfer msgspec torchao tyro diffusers \
    nest-asyncio pydantic peft accelerate bitsandbytes \
    huggingface-hub tokenizers protobuf numpy scipy tqdm regex \
    sentencepiece safetensors psutil packaging
```

---

## 九、运行训练

**预期输出**：
- 模型加载到 `xpu:0`，显存占用约 8-10GB
- 第一步可能较慢（Triton JIT 编译），约 10-20 秒
- 第二步起稳定约 **11-15 秒/步**
- GPU 利用率 70-85%

---

## 十、微小故障排查

| 出现的问题 | 现象 | 解决方案 |
|---|---|---|
| **Ubuntu 22.04 驱动包名不对** | `libze1` 找不到，`sycl-ls` 报错 | 换 **24.04 (noble)**，包名是 `libze-dev` |
| **oneAPI 污染 LD_LIBRARY_PATH** | PyTorch 报 `libur_loader.so` 版本冲突 | 删除 `/etc/profile.d/oneapi.sh`，不加载 oneAPI 环境变量 |
| **通用 triton 覆盖 xpu 版** | `0 active drivers` 或 `cannot import intel` | **不要** `pip install triton`，只用 `pytorch-triton-xpu` |
| **bitsandbytes 4-bit 不支持 XPU** | `cdequantize_blockwise_fp32` 报错 | 改用 bf16 加载，不用 4-bit |
| **accelerate device_map 训练冲突** | `Can't train model loaded with device_map='auto'` | 模型全进 GPU（`device_map="xpu"` + `low_cpu_mem_usage=True`） |
| **meta tensor backward 报错** | `Cannot copy out of meta tensor` | 确保模型全在 GPU，不 offload 到 CPU |
| **Triton JIT 编译极慢** | 第一步 10-20 分钟 | 正常现象，设置 `TRITON_CACHE_DIR` 缓存，后续重启会快 |
| **Windows 原生训练极慢** | 1523s/it，GPU 利用率 6% | **必须迁移到 WSL2**，Windows 上 Triton XPU backend 未经优化 |
| **HuggingFace 联网超时** | `Timed out after 120s` | `local_files_only=True` 强制离线加载 |
| **模型内存双份** | CPU 内存和 GPU 显存各一份 | `low_cpu_mem_usage=True` + `device_map="xpu"` |

---

## 十一、性能对比

| 环境 | 模型 | 速度 | GPU 利用率 |
|---|---|---|---|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3.5-4B bf16 | **11-15s/it** | **70-85%** |

---

## 十二、未来预计更新的功能

1. **添加终端版图形化**：提升易用性
2. **最后导出 merged 与 gguf**：`自动下载 llama.cpp 导出
3. **4bit 模型支持**：Intel XPU bitsandbytes 4bit 支持懒烂得要命等我去修

---

> **最后**：祝训练胜利！

---

---

# <a id="繁體中文"></a>繁體中文

# Intel Arc A770 + Unsloth 雙端微調指南

> **Windows 端**：Windows 11 上使用 Intel Arc A770 16GB 對 4bit 量化模型（全量模型大概率也可以，此處選擇 Qwen3-8b-bnb-4bit）進行 LoRA 微調。由於 Triton 的 Intel XPU backend 在 Windows 上未經完整測試，直接執行會產生大量 MSVC/GCC 相容性問題。本倉庫的一部分旨在提供一個解決方案。
>
> **WSL2/Linux 端**：原始權重 < 16GB 的非量化模型（如 Qwen3.5-4B、Qwen3-1.7B 等），本指南基於 Intel Arc A770 16GB 作為裝置，Qwen3.5-4B 作為模型。

---

# Intel Arc A770 + Unsloth + Windows 微調腳本

> **適用場景**：Windows 11 上使用 Intel Arc A770 16GB 對 4bit 量化模型（全量模型大概率也可以，此處選擇 Qwen3-8b-bnb-4bit）進行 LoRA 微調
> 
> 由於 Triton 的 Intel XPU backend 在 Windows 上未經完整測試，直接執行會產生大量 MSVC/GCC 相容性問題。本倉庫的一部分旨在提供一個解決方案

---

## 一、功能

- 在 **Windows 11** 上使用 **Intel Arc A770** 進行 **Qwen3-8B-BNB（僅為範例，其他模型自行測試）** 模型的 Unsloth LoRA 微調
- 自動偵測並載入 Intel oneAPI + MSVC 編譯環境
- 自動修復 Triton 在 Windows MSVC 下的 GCC 參數相容性問題
- 自動修復 MSVC 標頭檔/函式庫路徑缺失問題
- 支援訓練完成後匯出 LoRA / 16bit 完整權重 / GGUF（由於沒有跑完過所以最後是否能夠轉完整權重和 gguf 還未知，自行有耐心測試）

---

## 二、前提安裝

### 硬體
- **GPU**: Intel Arc 獨顯（核心顯示卡理論可行未測試，B系列理論可行未測試）
- **系統**: Windows 11 

### 軟體

| 元件 | 版本/要求 | 用途 |
|------|----------|------|
| Intel Arc 顯示卡驅動 | 最新版 (31.0.101.xxx+) | GPU 計算 |
| Intel oneAPI Base Toolkit 和 DeepLearning Toolkit | 2025.2 和最新版 | SYCL / Level Zero 執行階段 |
| Level Zero SDK | 1.28.x - 1.30.x | Triton XPU backend |
| Visual Studio 2022 | Community/Professional/Enterprise | MSVC C++ 編譯器 |
| Python | 3.13 (Windows 版) | 執行環境 |
| PyTorch | 2.12.1+xpu (Intel 官方 wheel) | XPU 深度學習框架 |
| Unsloth | 2026.6.9 | 快速微調框架 |

### VS 2022 必須安裝的工作負載
- **"使用 C++ 的桌面開發"**
- **MSVC v143 - VS 2022 C++ x64/x86 生成工具**
- **Windows 11 SDK**

---

## 三、適用模型

| 模型 | 格式 | 狀態 |
|------|------|------|
| Qwen3-8B-BNB | 4-bit BNB (unsloth 預量化) | ✅ 已驗證可行但速度極慢 |

> 其他模型未經測試。理論上只要是 Unsloth 支援的、透過 BNB 4-bit 載入的模型均可使用，但可能需要額外調整。

---

## 四、為 Intel Arc 修復的 Bug

### 1. Triton GCC 參數透傳給 MSVC 導致 D8021 錯誤
**現象**: `cl: 命令列 error D8021 :無效的數值參數"/Wno-psabi"`  
**原因**: Triton Intel XPU backend 按 GCC 風格產生編譯命令，直接傳給 `cl.exe`  
**修復**: 攔截 `triton.runtime.build._build`，過濾 `-Wno-psabi`、`-Wno-deprecated-declarations`、`-fPIC` 等 GCC 參數，並將 `-D`/`-I`/`-L`/`-l`/`-shared` 轉換為 MSVC 風格 `/D`/`-I`/`-LIBPATH:`/`lib`/`/LD`

### 2. MSVC 找不到 C++ 標準函式庫標頭檔（`cstddef` 等）
**現象**: `fatal error C1083: 無法開啟包含檔案: "cstddef"`  
**原因**: `vcvars64.bat` 只設定了 `PATH`，沒設定 `INCLUDE` 和 `LIB` 環境變數  
**修復**: 從 `cl.exe` 路徑自動推斷 MSVC 工具鏈根目錄，補全 `INCLUDE`（MSVC include + Windows SDK ucrt/shared/um + ATL/MFC）和 `LIB`（MSVC lib/x64 + Windows SDK lib）

### 3. SYCL 標頭檔要求 C++17
**現象**: `error C2338: static_assert failed: 'DPCPP does not support C++ version earlier than C++17.'`  
**原因**: MSVC 預設 C++14，SYCL 標頭檔用 `__cplusplus` 巨集檢查版本  
**修復**: 編譯命令新增 `/std:c++17` 和 `/Zc:__cplusplus`（後者讓 MSVC 正確設定 `__cplusplus` 巨集為 `201703L`）

### 4. 連結時找不到 `python313.lib`
**現象**: `LINK : fatal error LNK1104: 無法開啟檔案"python313.lib"`  
**原因**: Triton 編譯 Python 擴充（`.pyd`）時，`library_dirs` 只包含 `Library/bin` 和 `Library/lib`，沒有 `libs`  
**修復**: 自動偵測 `Python313/libs` 目錄並新增到 `/LIBPATH`

### 5. 連結器要求進入點（缺少 `/LD`）
**現象**: `LINK : fatal error LNK1561: 必須定義進入點`  
**原因**: `.pyd` 本質是 DLL，需要 `/LD` 旗標，但編譯命令中沒有  
**修復**: 在編譯命令中新增 `/LD`（建立 DLL）

### 6. Triton JIT 編譯極慢（快取不生效）
**現象**: 訓練第一步耗時 20+ 分鐘，GPU 利用率接近 0%  
**原因**: Windows 上 Triton Intel XPU backend 的 JIT kernel 快取機制有問題，每次 step 都可能重新編譯 SPIR-V  
**緩解**: 設定 `TRITON_CACHE_DIR` 和 `TRITON_DISABLE_AUTOTUNE=1` 減少重複編譯開銷

---

## 五、仍存問題

- **訓練速度極慢**: 即使修復了編譯問題，Windows 上 Triton XPU backend 的 JIT kernel 執行效率遠低於 Linux，GPU 利用率長期低於 10%，單步訓練仍需數分鐘至數十分鐘
- **Triton 快取不完全可靠**: `TRITON_CACHE_DIR` 有時無法命中，導致同一 kernel 多次重新編譯
- **Level Zero SDK 版本不一致**: 環境變數 `ZE_PATH` 指向 1.30.0，但 Triton 編譯命令中可能出現 1.28.2 路徑，需手動統一
- **xformers 不支援**: Intel XPU 無法使用 xformers（僅 CUDA），Unsloth 的部分 FlashAttention 最佳化失效
- **mem_get_info 跨平台差異**: `torch.xpu.memory.mem_get_info()` 在 Windows Intel Arc 驅動上可用，但在 WSL2/Linux 上不可用（跨平台腳本需注意）

---

## 六、版本說明（Release）

第八個版本才是能夠開始跑起來的模型，前七個版本都沒有完整修復 bug 使其正常開始偵錯

---

## 七、快速開始

```powershell
# 1. 確保已安裝所有前提軟體（見上文）
# 2. 下載 Release 並解壓縮
# 3. 修改腳本頂部的 CONFIG 區域（模型路徑、資料集路徑等）
# 4. 執行
```

---

## 八、配置說明

```python
CONFIG = {
    "model_path": r"H:/Qwen3-8B-unsloth-bnb-4bit",      # 模型路徑
    "dataset_path": r"D:/dataset.json",                  # 資料集路徑
    "output_dir": r"H:/unsloth_train/outputs",          # 輸出目錄
    "max_seq_length": 1024,
    "lora_r": 16,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "batch_size": 1,        # 根據顯存調整
    "grad_accum": 4,        # 總 batch = batch_size * grad_accum
    "max_steps": 3000,
    "warmup_steps": 5,
}
```

---

## 九、微小故障排查

| 出現的問題 | 現象 | 解決方案 |
|---|---|---|
| **Triton GCC 參數報錯** | `D8021 :無效的數值參數"/Wno-psabi"` | 使用 v2+ 版本，已過濾 GCC 參數 |
| **找不到 C++ 標頭檔** | `fatal error C1083: "cstddef"` | 使用 v4+ 版本，自動修復 INCLUDE |
| **SYCL 要求 C++17** | `DPCPP does not support C++ version earlier than C++17` | 使用 v5+ 版本，已新增 `/std:c++17` |
| **找不到 python313.lib** | `LNK1104: 無法開啟檔案"python313.lib"` | 使用 v6+ 版本，自動新增 Python libs |
| **必須定義進入點** | `LNK1561: 必須定義進入點` | 使用 v7+ 版本，已新增 `/LD` |
| **訓練第一步極慢** | 20+ 分鐘，GPU 利用率 0% | 使用 v8 版本，設定 `TRITON_CACHE_DIR`；若仍極慢，建議遷移到 WSL2/Linux |
| **Windows 原生訓練不可接受** | 1523s/it，GPU 利用率 6% | **必須遷移到 WSL2/Linux**，Windows 上 Triton XPU backend 未經最佳化 |
| **Level Zero 版本不一致** | 編譯命令中出現不同版本路徑 | 統一環境變數 `ZE_PATH` 與實際安裝的 SDK 版本 |

---

## 十、效能比較

| 環境 | 模型 | 速度 | GPU 利用率 |
|---|---|---|---|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3-8B bnb-4bit | 11-15s/it | 70-85% |

> **結論**: Windows 上只能解決"能跑"的問題，無法解決"快"的問題。如需實際訓練，強烈建議遷移到 WSL2/Linux。

---

## 十一、一鍵重建腳本（Windows 環境檢查）

```powershell
# 檢查必要環境變數
$env:ZE_PATH
$env:CC

# 檢查 VS 2022 安裝
Test-Path "C:\Program Files\Microsoft Visual Studio2\Community\VC\Auxiliary\Buildcvars64.bat"

# 檢查 Python 版本
python --version  # 應為 3.13

# 檢查 PyTorch XPU
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"

# 檢查 Triton
python -c "import triton; print(triton.__version__)"
```

---

## 十二、未來預計更新的功能

1. **新增終端機版圖形化**：提升易用性
2. **最後匯出 merged 與 gguf**：匯出待測試
3. **Windows 速度最佳化**：等待 Intel/Triton 官方修復 XPU backend 在 Windows 上的效能或嘗試移植套件

---

> **最後**：祝訓練勝利！如需實際訓練，請移步 WSL2/Linux

---

---

# Intel Arc A770 + Unsloth + WSL2 微調模型腳本

> **適用場景**：原始權重 < 16GB 的非量化模型（如 Qwen3.5-4B、Qwen3-1.7B 等），本指南基於 Intel Arc A770 16GB 作為裝置，Qwen3.5-4B 作為模型
> 

---

## 一、硬體/環境需求

- **GPU**: Intel Arc 獨顯裝置（核心顯示卡請自行測試，B系列理論可行但未測試）
- **OS**: Windows 11 21H2+，開啟 WSL2
- **WSL2 發行版**: Ubuntu 24.04 (Noble)，22.04 的 Intel GPU 驅動套件名稱和倉庫路徑不同，26.04 的 Python 版本過高不適用
- **Python**: 3.12


---

## 二、WSL2 Ubuntu 24.04 安裝

在 Windows PowerShell（管理員）中執行：

```powershell
# 更新 WSL
wsl --update

# 安裝 Ubuntu 24.04
wsl --install Ubuntu-24.04
wsl --set-default Ubuntu-24.04
#如果提示找不到那就是微軟伺服器間歇性故障，請透過 store 或自行下載 Ubuntu 安裝包進行安裝

# 驗證版本
wsl --list --verbose
# 應顯示 Ubuntu-24.04 Running 版本 2
```

---

## 三、Intel GPU 驅動和執行階段配置

進入 WSL2 Ubuntu 24.04 終端機，執行（如果出現缺失套件等問題請自行 sudo apt 安裝或者詢問 AI）：

```bash
# 1. 更新系統
sudo apt update && sudo apt upgrade -y

# 2. 安裝基礎工具
sudo apt install -y gpg-agent wget build-essential python3.12-dev

# 3. 新增 Intel GPU 倉庫（Noble 版本）
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified' | \
  sudo tee /etc/apt/sources.list.d/intel.gpu.noble.list

sudo apt update

# 4. 安裝 Intel GPU 執行階段（關鍵套件）
sudo apt install -y libze-dev intel-opencl-icd intel-media-va-driver-non-free \
  libmfx1 libvpl2 libegl-mesa0 libegl1-mesa-dev libgbm1 libgl1-mesa-dev \
  libgl1-mesa-dri libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 \
  libxatracker2 mesa-va-drivers mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all

# 5. 將使用者加入 render 群組（GPU 存取權限）
sudo gpasswd -a ${USER} render
newgrp render

# 6. 驗證 GPU 可見性
ls /dev/dri
# 應看到 renderD128 和 card0

clinfo | grep "Device Name"
# 應顯示 Intel(R) Arc(TM) A770 Graphics 或者 0x5860 之類
```

> **⚠️絕對注意**：
> - 不要安裝完整版 oneAPI Base Toolkit（會污染 LD_LIBRARY_PATH，導致 PyTorch 函式庫衝突）
> - 如果之前安裝過 oneAPI 並設定了 /etc/profile.d/oneapi.sh，**務必刪除**：
>   ```bash
>   sudo rm /etc/profile.d/oneapi.sh
>   ```
> - 如果 sycl-ls 後來因版本衝突損壞，**不影響 PyTorch 訓練**，不用管。

---

## 四、PyTorch XPU 環境安裝（虛擬環境名為 unsloth_env）

```bash
# 1. 建立虛擬環境
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# 2. 升級 pip
pip install --upgrade pip setuptools wheel

# 3. 安裝 PyTorch XPU 完整堆疊（內建 pytorch-triton-xpu，不要單獨安裝 triton）
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu \
    intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt \
    pytorch-triton-xpu tcmlib umf intel-pti \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple

# 4. 驗證 PyTorch XPU
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.is_available())"

# 5. 驗證 Triton XPU（正確的驗證方式）
python -c "
import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

x = torch.rand(128, device='xpu')
out = torch.empty_like(x)
n_elements = x.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
test_kernel[grid](x, out, n_elements, BLOCK_SIZE=128)
print('Triton XPU test passed!')
"
```

> **⚠️注意事項**：
> - **不要** `pip install triton`（會覆蓋 pytorch-triton-xpu，導致 Intel XPU backend 遺失，且安裝的是通用版 triton 不包含 xpu 算子支援）
> - **不要** `pip install xformers`（只支援 CUDA，會連帶安裝 NVIDIA 驅動）
> - **不要** `pip install intel_extension_for_pytorch`（PyTorch 2.7.1+xpu 已有原生 XPU 支援，IPEX 會引入版本衝突）

---

## 五、Unsloth 安裝

```bash
source ~/unsloth_env/bin/activate

# 1. 安裝 Unsloth（必須保持無依賴安裝）
pip install --no-deps unsloth unsloth-zoo

# 2. 手動安裝 Unsloth 的其他相依套件（跳過 xformers 和 triton）
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0 \
    cut_cross_entropy hf_transfer msgspec torchao tyro diffusers \
    nest-asyncio pydantic peft accelerate bitsandbytes \
    huggingface-hub tokenizers protobuf numpy scipy tqdm regex \
    sentencepiece safetensors psutil packaging
```

---

## 六、此腳本做出的修復

### torch.xpu.memory.mem_get_info() 不支援
PyTorch issue #164057，Arc A770 WSL2/Linux 驅動未實作此 API。
**修復**：monkey-patch 回傳固定值。

### torch.xpu.get_device_properties() 可能當機
**修復**：異常時回傳 FakeProps。

### Intel XPU 在 WSL2 下缺失顯存分配函數
**修復**：設定環境變數 `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` 和 `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True`。

### transformers caching_allocator_warmup 導致 OOM
**修復**：在 `import unsloth` 之前停用。

### Triton JIT 編譯慢（Intel XPU 通病）
**修復**：設定 `TRITON_CACHE_DIR` 快取編譯結果，設定 `IPEX_XPU_ONEDNN_LAYOUT=1` 加速記憶體吞吐量。

### Unsloth fix_untrained_tokens 與 meta tensor 衝突
**修復**：停用此函數。

---

## 七、完整訓練程式碼（v11 最佳化版）

已提供於 release 或倉庫中，請自行尋找，最終可用版本為 v12，前 11 個版本都沒有完整修復上述問題

---

## 八、一鍵重建腳本

如果你搞壞了環境，直接執行此腳本重建：

```bash
set -e

echo ">>> 開始重建環境..."

# 1. 刪除舊環境
rm -rf ~/unsloth_env

# 2. 建立新環境
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# 3. 升級 pip
pip install --upgrade pip setuptools wheel

# 4. 安裝 PyTorch XPU 
echo ">>> 安裝 PyTorch XPU..."
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu \
    intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt \
    pytorch-triton-xpu tcmlib umf intel-pti \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple

# 5. 驗證 PyTorch XPU
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.is_available())"

# 6. 驗證 Triton XPU
echo ">>> 驗證 Triton XPU..."
python -c "
import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

x = torch.rand(128, device='xpu')
out = torch.empty_like(x)
n_elements = x.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
test_kernel[grid](x, out, n_elements, BLOCK_SIZE=128)
print('Triton XPU test passed!')
"

# 7. 安裝 Unsloth（不覆蓋 PyTorch）
echo ">>> 安裝 Unsloth..."
pip install --no-deps unsloth unsloth-zoo

# 8. 手動安裝其他相依套件（跳過 xformers 和 triton）
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0 \
    cut_cross_entropy hf_transfer msgspec torchao tyro diffusers \
    nest-asyncio pydantic peft accelerate bitsandbytes \
    huggingface-hub tokenizers protobuf numpy scipy tqdm regex \
    sentencepiece safetensors psutil packaging
```

---

## 九、執行訓練

**預期輸出**：
- 模型載入到 `xpu:0`，顯存佔用約 8-10GB
- 第一步可能較慢（Triton JIT 編譯），約 10-20 秒
- 第二步起穩定約 **11-15 秒/步**
- GPU 利用率 70-85%

---

## 十、微小故障排查

| 出現的問題 | 現象 | 解決方案 |
|---|---|---|
| **Ubuntu 22.04 驅動套件名稱不對** | `libze1` 找不到，`sycl-ls` 報錯 | 換 **24.04 (noble)**，套件名稱是 `libze-dev` |
| **oneAPI 污染 LD_LIBRARY_PATH** | PyTorch 報 `libur_loader.so` 版本衝突 | 刪除 `/etc/profile.d/oneapi.sh`，不載入 oneAPI 環境變數 |
| **通用 triton 覆蓋 xpu 版** | `0 active drivers` 或 `cannot import intel` | **不要** `pip install triton`，只用 `pytorch-triton-xpu` |
| **bitsandbytes 4-bit 不支援 XPU** | `cdequantize_blockwise_fp32` 報錯 | 改用 bf16 載入，不用 4-bit |
| **accelerate device_map 訓練衝突** | `Can't train model loaded with device_map='auto'` | 模型全進 GPU（`device_map="xpu"` + `low_cpu_mem_usage=True`） |
| **meta tensor backward 報錯** | `Cannot copy out of meta tensor` | 確保模型全在 GPU，不 offload 到 CPU |
| **Triton JIT 編譯極慢** | 第一步 10-20 分鐘 | 正常現象，設定 `TRITON_CACHE_DIR` 快取，後續重啟會快 |
| **Windows 原生訓練極慢** | 1523s/it，GPU 利用率 6% | **必須遷移到 WSL2**，Windows 上 Triton XPU backend 未經最佳化 |
| **HuggingFace 連線逾時** | `Timed out after 120s` | `local_files_only=True` 強制離線載入 |
| **模型記憶體雙份** | CPU 記憶體和 GPU 顯存各一份 | `low_cpu_mem_usage=True` + `device_map="xpu"` |

---

## 十一、效能比較

| 環境 | 模型 | 速度 | GPU 利用率 |
|---|---|---|---|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3.5-4B bf16 | **11-15s/it** | **70-85%** |

---

## 十二、未來預計更新的功能

1. **新增終端機版圖形化**：提升易用性
2. **最後匯出 merged 與 gguf**：`自動下載 llama.cpp 匯出
3. **4bit 模型支援**：Intel XPU bitsandbytes 4bit 支援爛得要命等我去修

---

> **最後**：祝訓練勝利！

---

---

# <a id="english"></a>English

# Intel Arc A770 + Unsloth Dual-Platform Fine-Tuning Guide

> **Windows**: Use Intel Arc A770 16GB on Windows 11 for LoRA fine-tuning of 4-bit quantized models (full-precision models likely work too; Qwen3-8b-bnb-4bit is used here as an example). Since Triton's Intel XPU backend has not been fully tested on Windows, running it directly produces numerous MSVC/GCC compatibility issues. Part of this repository aims to provide a solution.
>
> **WSL2/Linux**: Non-quantized models with original weights < 16GB (e.g., Qwen3.5-4B, Qwen3-1.7B, etc.). This guide uses Intel Arc A770 16GB as the device and Qwen3.5-4B as the model.

---

# Intel Arc A770 + Unsloth + Windows Fine-Tuning Script

> **Use Case**: LoRA fine-tuning of 4-bit quantized models (full-precision models likely work too; Qwen3-8b-bnb-4bit is used here as an example) using Intel Arc A770 16GB on Windows 11.
>
> Since Triton's Intel XPU backend has not been fully tested on Windows, running it directly produces numerous MSVC/GCC compatibility issues. Part of this repository aims to provide a solution.

---

## I. Features

- **Qwen3-8B-BNB (example only; test other models yourself)** model Unsloth LoRA fine-tuning on **Windows 11** using **Intel Arc A770**
- Auto-detect and load Intel oneAPI + MSVC compilation environment
- Auto-fix Triton GCC parameter compatibility issues under Windows MSVC
- Auto-fix missing MSVC header/library path issues
- Support exporting LoRA / 16-bit full weights / GGUF after training (since a full run has not been completed, whether final conversion to full weights and GGUF works is unknown; test patiently at your own discretion)

---

## II. Prerequisites

### Hardware
- **GPU**: Intel Arc discrete GPU (integrated graphics theoretically work but untested; B-series theoretically work but untested)
- **OS**: Windows 11

### Software

| Component | Version/Requirement | Purpose |
|-----------|---------------------|---------|
| Intel Arc Graphics Driver | Latest (31.0.101.xxx+) | GPU compute |
| Intel oneAPI Base Toolkit & DeepLearning Toolkit | 2025.2 & latest | SYCL / Level Zero runtime |
| Level Zero SDK | 1.28.x - 1.30.x | Triton XPU backend |
| Visual Studio 2022 | Community/Professional/Enterprise | MSVC C++ compiler |
| Python | 3.13 (Windows version) | Runtime environment |
| PyTorch | 2.12.1+xpu (Intel official wheel) | XPU deep learning framework |
| Unsloth | 2026.6.9 | Fast fine-tuning framework |

### VS 2022 Required Workloads
- **"Desktop development with C++"**
- **MSVC v143 - VS 2022 C++ x64/x86 build tools**
- **Windows 11 SDK**

---

## III. Supported Models

| Model | Format | Status |
|-------|--------|--------|
| Qwen3-8B-BNB | 4-bit BNB (unsloth pre-quantized) | ✅ Verified working but extremely slow |

> Other models are untested. In theory, any model supported by Unsloth and loadable via BNB 4-bit should work, but additional adjustments may be needed.

---

## IV. Bugs Fixed for Intel Arc

### 1. Triton GCC parameters passed through to MSVC causing D8021 error
**Symptom**: `cl: Command line error D8021 : invalid numeric argument "/Wno-psabi"`  
**Cause**: Triton Intel XPU backend generates compilation commands in GCC style and passes them directly to `cl.exe`  
**Fix**: Intercept `triton.runtime.build._build`, filter out GCC parameters such as `-Wno-psabi`, `-Wno-deprecated-declarations`, `-fPIC`, and convert `-D`/`-I`/`-L`/`-l`/`-shared` to MSVC style `/D`/`-I`/`-LIBPATH:`/`lib`/`/LD`

### 2. MSVC cannot find C++ standard library headers (`cstddef`, etc.)
**Symptom**: `fatal error C1083: Cannot open include file: "cstddef"`  
**Cause**: `vcvars64.bat` only sets `PATH`, not `INCLUDE` and `LIB` environment variables  
**Fix**: Automatically infer the MSVC toolchain root directory from the `cl.exe` path, and supplement `INCLUDE` (MSVC include + Windows SDK ucrt/shared/um + ATL/MFC) and `LIB` (MSVC lib/x64 + Windows SDK lib)

### 3. SYCL headers require C++17
**Symptom**: `error C2338: static_assert failed: 'DPCPP does not support C++ version earlier than C++17.'`  
**Cause**: MSVC defaults to C++14, and SYCL headers check version via the `__cplusplus` macro  
**Fix**: Add `/std:c++17` and `/Zc:__cplusplus` to compilation commands (the latter makes MSVC correctly set the `__cplusplus` macro to `201703L`)

### 4. Linker cannot find `python313.lib`
**Symptom**: `LINK : fatal error LNK1104: cannot open file "python313.lib"`  
**Cause**: When Triton compiles Python extensions (`.pyd`), `library_dirs` only contains `Library/bin` and `Library/lib`, not `libs`  
**Fix**: Automatically detect the `Python313/libs` directory and add it to `/LIBPATH`

### 5. Linker requires entry point (missing `/LD`)
**Symptom**: `LINK : fatal error LNK1561: entry point must be defined`  
**Cause**: `.pyd` is essentially a DLL and requires the `/LD` flag, which is missing from the compilation command  
**Fix**: Add `/LD` (create DLL) to the compilation command

### 6. Triton JIT compilation extremely slow (cache not effective)
**Symptom**: First training step takes 20+ minutes, GPU utilization near 0%  
**Cause**: On Windows, Triton Intel XPU backend's JIT kernel caching mechanism has issues, potentially recompiling SPIR-V on every step  
**Mitigation**: Set `TRITON_CACHE_DIR` and `TRITON_DISABLE_AUTOTUNE=1` to reduce repeated compilation overhead

---

## V. Remaining Issues

- **Extremely slow training speed**: Even after fixing compilation issues, Triton XPU backend JIT kernel execution efficiency on Windows is far lower than on Linux. GPU utilization remains below 10% for extended periods, and a single training step still takes several minutes to tens of minutes.
- **Triton cache not fully reliable**: `TRITON_CACHE_DIR` sometimes fails to hit, causing the same kernel to be recompiled multiple times.
- **Level Zero SDK version inconsistency**: Environment variable `ZE_PATH` points to 1.30.0, but Triton compilation commands may reference 1.28.2 paths; manual unification required.
- **xformers not supported**: Intel XPU cannot use xformers (CUDA only), so some of Unsloth's FlashAttention optimizations are ineffective.
- **mem_get_info cross-platform differences**: `torch.xpu.memory.mem_get_info()` works on Windows Intel Arc drivers but is unavailable on WSL2/Linux (cross-platform scripts should take note).

---

## VI. Version Notes (Release)

The eighth version is the first one that can actually start running the model; the first seven versions did not fully fix bugs to enable normal debugging.

---

## VII. Quick Start

```powershell
# 1. Ensure all prerequisite software is installed (see above)
# 2. Download the Release and extract it
# 3. Modify the CONFIG section at the top of the script (model path, dataset path, etc.)
# 4. Run
```

---

## VIII. Configuration

```python
CONFIG = {
    "model_path": r"H:/Qwen3-8B-unsloth-bnb-4bit",      # Model path
    "dataset_path": r"D:/dataset.json",                  # Dataset path
    "output_dir": r"H:/unsloth_train/outputs",          # Output directory
    "max_seq_length": 1024,
    "lora_r": 16,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "batch_size": 1,        # Adjust according to VRAM
    "grad_accum": 4,        # Total batch = batch_size * grad_accum
    "max_steps": 3000,
    "warmup_steps": 5,
}
```

---

## IX. Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Triton GCC parameter error** | `D8021 : invalid numeric argument "/Wno-psabi"` | Use v2+; GCC parameters are filtered |
| **Cannot find C++ headers** | `fatal error C1083: "cstddef"` | Use v4+; INCLUDE is auto-fixed |
| **SYCL requires C++17** | `DPCPP does not support C++ version earlier than C++17` | Use v5+; `/std:c++17` is added |
| **Cannot find python313.lib** | `LNK1104: cannot open file "python313.lib"` | Use v6+; Python libs are auto-added |
| **Entry point must be defined** | `LNK1561: entry point must be defined` | Use v7+; `/LD` is added |
| **First training step extremely slow** | 20+ minutes, GPU utilization 0% | Use v8; set `TRITON_CACHE_DIR`; if still extremely slow, migrate to WSL2/Linux |
| **Windows native training unacceptable** | 1523s/it, GPU utilization 6% | **Must migrate to WSL2/Linux**; Triton XPU backend is unoptimized on Windows |
| **Level Zero version inconsistency** | Different version paths appear in compilation commands | Unify environment variable `ZE_PATH` with the actually installed SDK version |

---

## X. Performance Comparison

| Environment | Model | Speed | GPU Utilization |
|-------------|-------|-------|-----------------|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3-8B bnb-4bit | 11-15s/it | 70-85% |

> **Conclusion**: On Windows, we can only solve the "can it run" problem, not the "can it run fast" problem. For actual training, strongly recommend migrating to WSL2/Linux.

---

## XI. One-Click Rebuild Script (Windows Environment Check)

```powershell
# Check required environment variables
$env:ZE_PATH
$env:CC

# Check VS 2022 installation
Test-Path "C:\Program Files\Microsoft Visual Studio2\Community\VC\Auxiliary\Buildcvars64.bat"

# Check Python version
python --version  # Should be 3.13

# Check PyTorch XPU
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"

# Check Triton
python -c "import triton; print(triton.__version__)"
```

---

## XII. Future Planned Updates

1. **Add terminal GUI**: Improve usability
2. **Final export merged & gguf**: Export pending testing
3. **Windows speed optimization**: Wait for Intel/Triton official fixes for XPU backend performance on Windows, or attempt porting packages

---

> **Finally**: Good luck with training! For actual training, please move to WSL2/Linux.

---

---

# Intel Arc A770 + Unsloth + WSL2 Fine-Tuning Script

> **Use Case**: Non-quantized models with original weights < 16GB (e.g., Qwen3.5-4B, Qwen3-1.7B, etc.). This guide uses Intel Arc A770 16GB as the device and Qwen3.5-4B as the model.

---

## I. Hardware / Environment Requirements

- **GPU**: Intel Arc discrete GPU (integrated graphics untested; B-series theoretically work but untested)
- **OS**: Windows 11 21H2+, WSL2 enabled
- **WSL2 Distro**: Ubuntu 24.04 (Noble). 22.04 has different Intel GPU driver package names and repository paths; 26.04 has a Python version that is too high and not suitable.
- **Python**: 3.12

---

## II. WSL2 Ubuntu 24.04 Installation

Execute in Windows PowerShell (Administrator):

```powershell
# Update WSL
wsl --update

# Install Ubuntu 24.04
wsl --install Ubuntu-24.04
wsl --set-default Ubuntu-24.04
# If it says not found, Microsoft servers are intermittently down; download the Ubuntu installer from the Store yourself

# Verify version
wsl --list --verbose
# Should show Ubuntu-24.04 Running version 2
```

---

## III. Intel GPU Driver and Runtime Configuration

Enter the WSL2 Ubuntu 24.04 terminal and execute (if missing packages occur, install with sudo apt yourself or ask an AI):

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install basic tools
sudo apt install -y gpg-agent wget build-essential python3.12-dev

# 3. Add Intel GPU repository (Noble version)
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified' | \
  sudo tee /etc/apt/sources.list.d/intel.gpu.noble.list

sudo apt update

# 4. Install Intel GPU runtime (critical packages)
sudo apt install -y libze-dev intel-opencl-icd intel-media-va-driver-non-free \
  libmfx1 libvpl2 libegl-mesa0 libegl1-mesa-dev libgbm1 libgl1-mesa-dev \
  libgl1-mesa-dri libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 \
  libxatracker2 mesa-va-drivers mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all

# 5. Add user to render group (GPU access permission)
sudo gpasswd -a ${USER} render
newgrp render

# 6. Verify GPU visibility
ls /dev/dri
# Should see renderD128 and card0

clinfo | grep "Device Name"
# Should show Intel(R) Arc(TM) A770 Graphics or something like 0x5860
```

> **⚠️ Absolute Attention**:
> - Do NOT install the full oneAPI Base Toolkit (it will pollute LD_LIBRARY_PATH and cause PyTorch library conflicts)
> - If you previously installed oneAPI and configured /etc/profile.d/oneapi.sh, **you MUST delete it**:
>   ```bash
>   sudo rm /etc/profile.d/oneapi.sh
>   ```
> - If sycl-ls breaks later due to version conflicts, **it does not affect PyTorch training**; ignore it.

---

## IV. PyTorch XPU Environment Installation (virtual environment name: unsloth_env)

```bash
# 1. Create virtual environment
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch XPU full stack (includes pytorch-triton-xpu; do NOT install triton separately)
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu \
    intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt \
    pytorch-triton-xpu tcmlib umf intel-pti \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple

# 4. Verify PyTorch XPU
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.is_available())"

# 5. Verify Triton XPU (correct verification method)
python -c "
import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

x = torch.rand(128, device='xpu')
out = torch.empty_like(x)
n_elements = x.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
test_kernel[grid](x, out, n_elements, BLOCK_SIZE=128)
print('Triton XPU test passed!')
"
```

> **⚠️ Notes**:
> - **Do NOT** `pip install triton` (it will overwrite pytorch-triton-xpu, causing the Intel XPU backend to be lost, and the generic triton installed does not include XPU operator support)
> - **Do NOT** `pip install xformers` (CUDA only; will pull in NVIDIA drivers)
> - **Do NOT** `pip install intel_extension_for_pytorch` (PyTorch 2.7.1+xpu already has native XPU support; IPEX will introduce version conflicts)

---

## V. Unsloth Installation

```bash
source ~/unsloth_env/bin/activate

# 1. Install Unsloth (must keep no-deps)
pip install --no-deps unsloth unsloth-zoo

# 2. Manually install other Unsloth dependencies (skip xformers and triton)
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0 \
    cut_cross_entropy hf_transfer msgspec torchao tyro diffusers \
    nest-asyncio pydantic peft accelerate bitsandbytes \
    huggingface-hub tokenizers protobuf numpy scipy tqdm regex \
    sentencepiece safetensors psutil packaging
```

---

## VI. Fixes Applied by This Script

### torch.xpu.memory.mem_get_info() not supported
PyTorch issue #164057, Arc A770 WSL2/Linux driver has not implemented this API.
**Fix**: Monkey-patch to return fixed values.

### torch.xpu.get_device_properties() may crash
**Fix**: Return FakeProps on exception.

### Intel XPU missing VRAM allocation functions under WSL2
**Fix**: Set environment variables `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` and `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True`.

### transformers caching_allocator_warmup causes OOM
**Fix**: Disable before `import unsloth`.

### Triton JIT compilation slow (common Intel XPU issue)
**Fix**: Set `TRITON_CACHE_DIR` to cache compilation results; set `IPEX_XPU_ONEDNN_LAYOUT=1` to accelerate memory throughput.

### Unsloth fix_untrained_tokens conflicts with meta tensor
**Fix**: Disable this function.

---

## VII. Complete Training Code (v11 Optimized)

Provided in releases or the repository; please find it yourself. The final working version is v12; the first 11 versions did not fully fix the above issues.

---

## VIII. One-Click Rebuild Script

If you break your environment, run this script to rebuild:

```bash
set -e

echo ">>> Starting environment rebuild..."

# 1. Delete old environment
rm -rf ~/unsloth_env

# 2. Create new environment
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch XPU
echo ">>> Installing PyTorch XPU..."
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu \
    intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt \
    pytorch-triton-xpu tcmlib umf intel-pti \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple

# 5. Verify PyTorch XPU
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.is_available())"

# 6. Verify Triton XPU
echo ">>> Verifying Triton XPU..."
python -c "
import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

x = torch.rand(128, device='xpu')
out = torch.empty_like(x)
n_elements = x.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
test_kernel[grid](x, out, n_elements, BLOCK_SIZE=128)
print('Triton XPU test passed!')
"

# 7. Install Unsloth (do not overwrite PyTorch)
echo ">>> Installing Unsloth..."
pip install --no-deps unsloth unsloth-zoo

# 8. Manually install other dependencies (skip xformers and triton)
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0 \
    cut_cross_entropy hf_transfer msgspec torchao tyro diffusers \
    nest-asyncio pydantic peft accelerate bitsandbytes \
    huggingface-hub tokenizers protobuf numpy scipy tqdm regex \
    sentencepiece safetensors psutil packaging
```

---

## IX. Running Training

**Expected Output**:
- Model loaded to `xpu:0`, VRAM usage approximately 8-10GB
- First step may be slow (Triton JIT compilation), approximately 10-20 seconds
- From the second step onward, stable at approximately **11-15 seconds/step**
- GPU utilization 70-85%

---

## X. Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Ubuntu 22.04 driver package name wrong** | `libze1` not found, `sycl-ls` errors | Switch to **24.04 (noble)**; package name is `libze-dev` |
| **oneAPI pollutes LD_LIBRARY_PATH** | PyTorch reports `libur_loader.so` version conflict | Delete `/etc/profile.d/oneapi.sh`, do not load oneAPI environment variables |
| **Generic triton overwrites XPU version** | `0 active drivers` or `cannot import intel` | **Do NOT** `pip install triton`; only use `pytorch-triton-xpu` |
| **bitsandbytes 4-bit not supported on XPU** | `cdequantize_blockwise_fp32` error | Use bf16 loading instead of 4-bit |
| **accelerate device_map training conflict** | `Can't train model loaded with device_map='auto'` | Load entire model into GPU (`device_map="xpu"` + `low_cpu_mem_usage=True`) |
| **meta tensor backward error** | `Cannot copy out of meta tensor` | Ensure model is fully on GPU, no offload to CPU |
| **Triton JIT compilation extremely slow** | First step 10-20 minutes | Normal behavior; set `TRITON_CACHE_DIR` to cache; subsequent restarts will be faster |
| **Windows native training extremely slow** | 1523s/it, GPU utilization 6% | **Must migrate to WSL2**; Triton XPU backend is unoptimized on Windows |
| **HuggingFace network timeout** | `Timed out after 120s` | `local_files_only=True` to force offline loading |
| **Model memory duplicated** | CPU RAM and GPU VRAM each have a copy | `low_cpu_mem_usage=True` + `device_map="xpu"` |

---

## XI. Performance Comparison

| Environment | Model | Speed | GPU Utilization |
|-------------|-------|-------|-----------------|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3.5-4B bf16 | **11-15s/it** | **70-85%** |

---

## XII. Future Planned Updates

1. **Add terminal GUI**: Improve usability
2. **Final export merged & gguf**: `Auto-download llama.cpp for export
3. **4-bit model support**: Intel XPU bitsandbytes 4-bit support is terribly broken; I'll fix it when I get to it

---

> **Finally**: Good luck with training!

---

---

# <a id="日本語"></a>日本語

# Intel Arc A770 + Unsloth デュアルプラットフォームファインチューニングガイド

> **Windows 側**: Windows 11 上で Intel Arc A770 16GB を使用し、4bit 量子化モデル（フルプレシジョンモデルもおそらく動作しますが、ここでは Qwen3-8b-bnb-4bit を例として選択）の LoRA ファインチューニングを行います。Triton の Intel XPU バックエンドは Windows 上で完全にテストされていないため、直接実行すると多数の MSVC/GCC 互換性問題が発生します。このリポジトリの一部はその解決策を提供することを目的としています。
>
> **WSL2/Linux 側**: 元の重みが 16GB 未満の非量子化モデル（例：Qwen3.5-4B、Qwen3-1.7B など）。このガイドでは Intel Arc A770 16GB をデバイスとして、Qwen3.5-4B をモデルとして使用します。

---

# Intel Arc A770 + Unsloth + Windows ファインチューニングスクリプト

> **使用場面**: Windows 11 上で Intel Arc A770 16GB を使用し、4bit 量子化モデル（フルプレシジョンモデルもおそらく動作しますが、ここでは Qwen3-8b-bnb-4bit を例として選択）の LoRA ファインチューニング。
>
> Triton の Intel XPU バックエンドは Windows 上で完全にテストされていないため、直接実行すると多数の MSVC/GCC 互換性問題が発生します。このリポジトリの一部はその解決策を提供することを目的としています。

---

## 一、機能

- **Windows 11** 上で **Intel Arc A770** を使用した **Qwen3-8B-BNB（あくまで例であり、他のモデルは各自でテストしてください）** モデルの Unsloth LoRA ファインチューニング
- Intel oneAPI + MSVC コンパイル環境の自動検出と読み込み
- Windows MSVC 下での Triton GCC パラメータ互換性問題の自動修正
- MSVC ヘッダー/ライブラリパス欠落問題の自動修正
- トレーニング完了後の LoRA / 16bit フルウェイト / GGUF エクスポートをサポート（完全な実行が完了していないため、最終的にフルウェイトと GGUF への変換が可能かどうかは不明です。各自で忍耐強くテストしてください）

---

## 二、前提インストール

### ハードウェア
- **GPU**: Intel Arc ディスクリート GPU（統合グラフィックスは理論上動作する可能性がありますが未テスト、Bシリーズも理論上動作する可能性がありますが未テスト）
- **システム**: Windows 11

### ソフトウェア

| コンポーネント | バージョン/要件 | 用途 |
|-------------|--------------|------|
| Intel Arc グラフィックスドライバ | 最新版 (31.0.101.xxx+) | GPU 計算 |
| Intel oneAPI Base Toolkit & DeepLearning Toolkit | 2025.2 & 最新版 | SYCL / Level Zero ランタイム |
| Level Zero SDK | 1.28.x - 1.30.x | Triton XPU バックエンド |
| Visual Studio 2022 | Community/Professional/Enterprise | MSVC C++ コンパイラ |
| Python | 3.13 (Windows 版) | 実行環境 |
| PyTorch | 2.12.1+xpu (Intel 公式 wheel) | XPU ディープラーニングフレームワーク |
| Unsloth | 2026.6.9 | 高速ファインチューニングフレームワーク |

### VS 2022 必須ワークロード
- **"C++ によるデスクトップ開発"**
- **MSVC v143 - VS 2022 C++ x64/x86 ビルドツール**
- **Windows 11 SDK**

---

## 三、対応モデル

| モデル | フォーマット | 状態 |
|-------|------------|------|
| Qwen3-8B-BNB | 4-bit BNB (unsloth 事前量子化) | ✅ 動作確認済みだが極めて遅い |

> 他のモデルは未テストです。理論上、Unsloth がサポートし、BNB 4-bit で読み込み可能なモデルであれば動作するはずですが、追加の調整が必要になる場合があります。

---

## 四、Intel Arc 向けに修正したバグ

### 1. Triton GCC パラメータが MSVC に透過して D8021 エラーが発生
**現象**: `cl: コマンドライン エラー D8021 : 無効な数値引数 "/Wno-psabi"`  
**原因**: Triton Intel XPU バックエンドが GCC スタイルでコンパイルコマンドを生成し、`cl.exe` に直接渡す  
**修正**: `triton.runtime.build._build` をインターセプトし、`-Wno-psabi`、`-Wno-deprecated-declarations`、`-fPIC` などの GCC パラメータをフィルタリングし、`-D`/`-I`/`-L`/`-l`/`-shared` を MSVC スタイル `/D`/`-I`/`-LIBPATH:`/`lib`/`/LD` に変換

### 2. MSVC が C++ 標準ライブラリヘッダー（`cstddef` など）を見つけられない
**現象**: `fatal error C1083: インクルードファイルを開けません: "cstddef"`  
**原因**: `vcvars64.bat` は `PATH` のみ設定し、`INCLUDE` と `LIB` 環境変数は設定しない  
**修正**: `cl.exe` パスから MSVC ツールチェーンルートディレクトリを自動推定し、`INCLUDE`（MSVC include + Windows SDK ucrt/shared/um + ATL/MFC）と `LIB`（MSVC lib/x64 + Windows SDK lib）を補完

### 3. SYCL ヘッダーが C++17 を要求
**現象**: `error C2338: static_assert failed: 'DPCPP does not support C++ version earlier than C++17.'`  
**原因**: MSVC はデフォルトで C++14、SYCL ヘッダーは `__cplusplus` マクロでバージョンをチェック  
**修正**: コンパイルコマンドに `/std:c++17` と `/Zc:__cplusplus` を追加（後者は MSVC に `__cplusplus` マクロを `201703L` に正しく設定させる）

### 4. リンク時に `python313.lib` が見つからない
**現象**: `LINK : 致命的エラー LNK1104: ファイル "python313.lib" を開けません`  
**原因**: Triton が Python 拡張（`.pyd`）をコンパイルする際、`library_dirs` は `Library/bin` と `Library/lib` のみを含み、`libs` は含まない  
**修正**: `Python313/libs` ディレクトリを自動検出し、`/LIBPATH` に追加

### 5. リンカーがエントリポイントを要求（`/LD` 欠落）
**現象**: `LINK : 致命的エラー LNK1561: エントリポイントを定義する必要があります`  
**原因**: `.pyd` は本質的に DLL であり、`/LD` フラグが必要だが、コンパイルコマンドにない  
**修正**: コンパイルコマンドに `/LD`（DLL の作成）を追加

### 6. Triton JIT コンパイルが極めて遅い（キャッシュが効かない）
**現象**: トレーニングの最初のステップに 20 分以上かかり、GPU 利用率がほぼ 0%  
**原因**: Windows 上の Triton Intel XPU バックエンドの JIT カーネルキャッシュメカニズムに問題があり、毎ステップで SPIR-V を再コンパイルする可能性がある  
**緩和**: `TRITON_CACHE_DIR` と `TRITON_DISABLE_AUTOTUNE=1` を設定し、繰り返しコンパイルのオーバーヘッドを削減

---

## 五、残存問題

- **トレーニング速度が極めて遅い**: コンパイル問題を修正しても、Windows 上の Triton XPU バックエンド JIT カーネル実行効率は Linux よりはるかに低い。GPU 利用率は長期間 10% を下回り、単一ステップのトレーニングにも数分から数十分かかる。
- **Triton キャッシュが完全に信頼できない**: `TRITON_CACHE_DIR` がヒットしない場合があり、同じカーネルが複数回再コンパイルされる。
- **Level Zero SDK バージョン不一致**: 環境変数 `ZE_PATH` は 1.30.0 を指すが、Triton コンパイルコマンドでは 1.28.2 のパスが参照されることがあり、手動での統一が必要。
- **xformers 非対応**: Intel XPU は xformers を使用できない（CUDA のみ）、Unsloth の一部の FlashAttention 最適化が無効になる。
- **mem_get_info クロスプラットフォーム差異**: `torch.xpu.memory.mem_get_info()` は Windows Intel Arc ドライバーでは動作するが、WSL2/Linux では使用不可（クロスプラットフォームスクリプトは注意が必要）。

---

## 六、バージョン説明（Release）

8 番目のバージョンが初めてモデルを実際に実行できるようになったものです。最初の 7 つのバージョンは、正常なデバッグを可能にするためのバグを完全に修正していませんでした。

---

## 七、クイックスタート

```powershell
# 1. すべての前提ソフトウェアがインストールされていることを確認（上記参照）
# 2. Release をダウンロードして展開
# 3. スクリプト上部の CONFIG セクションを修正（モデルパス、データセットパスなど）
# 4. 実行
```

---

## 八、設定

```python
CONFIG = {
    "model_path": r"H:/Qwen3-8B-unsloth-bnb-4bit",      # モデルパス
    "dataset_path": r"D:/dataset.json",                  # データセットパス
    "output_dir": r"H:/unsloth_train/outputs",          # 出力ディレクトリ
    "max_seq_length": 1024,
    "lora_r": 16,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "batch_size": 1,        # VRAM に応じて調整
    "grad_accum": 4,        # 総 batch = batch_size * grad_accum
    "max_steps": 3000,
    "warmup_steps": 5,
}
```

---

## 九、トラブルシューティング

| 問題 | 現象 | 解決策 |
|------|------|--------|
| **Triton GCC パラメータエラー** | `D8021 : 無効な数値引数 "/Wno-psabi"` | v2+ を使用；GCC パラメータはフィルタリング済み |
| **C++ ヘッダーが見つからない** | `fatal error C1083: "cstddef"` | v4+ を使用；INCLUDE は自動修正 |
| **SYCL が C++17 を要求** | `DPCPP does not support C++ version earlier than C++17` | v5+ を使用；`/std:c++17` を追加 |
| **python313.lib が見つからない** | `LNK1104: ファイル "python313.lib" を開けません` | v6+ を使用；Python libs は自動追加 |
| **エントリポイントを定義する必要がある** | `LNK1561: エントリポイントを定義する必要があります` | v7+ を使用；`/LD` を追加 |
| **トレーニング最初のステップが極めて遅い** | 20+ 分、GPU 利用率 0% | v8 を使用；`TRITON_CACHE_DIR` を設定；それでも極めて遅い場合は WSL2/Linux へ移行 |
| **Windows ネイティブトレーニングが許容できない** | 1523s/it、GPU 利用率 6% | **WSL2/Linux へ移行が必須**；Windows 上の Triton XPU バックエンドは最適化されていない |
| **Level Zero バージョン不一致** | コンパイルコマンドに異なるバージョンパスが出現 | 環境変数 `ZE_PATH` を実際にインストールされた SDK バージョンと統一 |

---

## 十、パフォーマンス比較

| 環境 | モデル | 速度 | GPU 利用率 |
|------|--------|------|-----------|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3-8B bnb-4bit | 11-15s/it | 70-85% |

> **結論**: Windows では「動作するか」という問題のみ解決でき、「速いか」という問題は解決できません。実際のトレーニングには、WSL2/Linux への移行を強く推奨します。

---

## 十一、ワンクリックリビルドスクリプト（Windows 環境チェック）

```powershell
# 必要な環境変数を確認
$env:ZE_PATH
$env:CC

# VS 2022 インストールを確認
Test-Path "C:\Program Files\Microsoft Visual Studio2\Community\VC\Auxiliary\Buildcvars64.bat"

# Python バージョンを確認
python --version  # 3.13 であるべき

# PyTorch XPU を確認
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"

# Triton を確認
python -c "import triton; print(triton.__version__)"
```

---

## 十二、今後の予定アップデート

1. **ターミナル GUI の追加**: 使いやすさの向上
2. **最終エクスポート merged & gguf**: エクスポートはテスト待ち
3. **Windows 速度最適化**: Intel/Triton による Windows 上の XPU バックエンドパフォーマンスの公式修正を待つ、またはパッケージの移植を試みる

---

> **最後に**: トレーニングが成功しますように！実際のトレーニングには WSL2/Linux をご利用ください。

---

---

# Intel Arc A770 + Unsloth + WSL2 ファインチューニングスクリプト

> **使用場面**: 元の重みが 16GB 未満の非量子化モデル（例：Qwen3.5-4B、Qwen3-1.7B など）。このガイドでは Intel Arc A770 16GB をデバイスとして、Qwen3.5-4B をモデルとして使用します。

---

## 一、ハードウェア / 環境要件

- **GPU**: Intel Arc ディスクリート GPU（統合グラフィックスは未テスト；Bシリーズは理論上動作する可能性がありますが未テスト）
- **OS**: Windows 11 21H2+、WSL2 有効化
- **WSL2 ディストリビューション**: Ubuntu 24.04 (Noble)。22.04 は Intel GPU ドライバーのパッケージ名とリポジトリパスが異なる；26.04 は Python バージョンが高すぎて適さない。
- **Python**: 3.12

---

## 二、WSL2 Ubuntu 24.04 インストール

Windows PowerShell（管理者）で実行：

```powershell
# WSL を更新
wsl --update

# Ubuntu 24.04 をインストール
wsl --install Ubuntu-24.04
wsl --set-default Ubuntu-24.04
# 見つからないと表示された場合、Microsoft サーバーが一時的にダウンしています；ストアから Ubuntu インストーラーを自分でダウンロードしてください

# バージョンを確認
wsl --list --verbose
# Ubuntu-24.04 Running version 2 と表示されるべき
```

---

## 三、Intel GPU ドライバーとランタイム設定

WSL2 Ubuntu 24.04 ターミナルに入り、実行（パッケージ欠落などの問題が発生した場合は、自分で sudo apt でインストールするか AI に尋ねてください）：

```bash
# 1. システムを更新
sudo apt update && sudo apt upgrade -y

# 2. 基本ツールをインストール
sudo apt install -y gpg-agent wget build-essential python3.12-dev

# 3. Intel GPU リポジトリを追加（Noble バージョン）
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified' | \
  sudo tee /etc/apt/sources.list.d/intel.gpu.noble.list

sudo apt update

# 4. Intel GPU ランタイムをインストール（重要パッケージ）
sudo apt install -y libze-dev intel-opencl-icd intel-media-va-driver-non-free \
  libmfx1 libvpl2 libegl-mesa0 libegl1-mesa-dev libgbm1 libgl1-mesa-dev \
  libgl1-mesa-dri libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 \
  libxatracker2 mesa-va-drivers mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all

# 5. ユーザーを render グループに追加（GPU アクセス権限）
sudo gpasswd -a ${USER} render
newgrp render

# 6. GPU 可視性を確認
ls /dev/dri
# renderD128 と card0 が見えるべき

clinfo | grep "Device Name"
# Intel(R) Arc(TM) A770 Graphics または 0x5860 などが表示されるべき
```

> **⚠️絶対注意**:
> - フル版の oneAPI Base Toolkit はインストールしないでください（LD_LIBRARY_PATH を汚染し、PyTorch ライブラリの競合を引き起こします）
> - 以前に oneAPI をインストールし /etc/profile.d/oneapi.sh を設定した場合、**必ず削除してください**:
>   ```bash
>   sudo rm /etc/profile.d/oneapi.sh
>   ```
> - sycl-ls が後にバージョン競合で壊れた場合、**PyTorch トレーニングには影響しません**；無視してください。

---

## 四、PyTorch XPU 環境インストール（仮想環境名：unsloth_env）

```bash
# 1. 仮想環境を作成
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# 2. pip をアップグレード
pip install --upgrade pip setuptools wheel

# 3. PyTorch XPU フルスタックをインストール（pytorch-triton-xpu 同梱；triton は個別にインストールしない）
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu \
    intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt \
    pytorch-triton-xpu tcmlib umf intel-pti \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple

# 4. PyTorch XPU を確認
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.is_available())"

# 5. Triton XPU を確認（正しい確認方法）
python -c "
import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

x = torch.rand(128, device='xpu')
out = torch.empty_like(x)
n_elements = x.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
test_kernel[grid](x, out, n_elements, BLOCK_SIZE=128)
print('Triton XPU test passed!')
"
```

> **⚠️注意事項**:
> - **`pip install triton` はしないでください**（pytorch-triton-xpu を上書きし、Intel XPU バックエンドが失われ、インストールされる汎用版 triton は XPU オペレータサポートを含みません）
> - **`pip install xformers` はしないでください**（CUDA のみ；NVIDIA ドライバーを引き込みます）
> - **`pip install intel_extension_for_pytorch` はしないでください**（PyTorch 2.7.1+xpu は既にネイティブ XPU サポートを持ち、IPEX はバージョン競合を引き起こします）

---

## 五、Unsloth インストール

```bash
source ~/unsloth_env/bin/activate

# 1. Unsloth をインストール（依存関係なしでインストールする必要がある）
pip install --no-deps unsloth unsloth-zoo

# 2. Unsloth のその他の依存関係を手動インストール（xformers と triton はスキップ）
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0 \
    cut_cross_entropy hf_transfer msgspec torchao tyro diffusers \
    nest-asyncio pydantic peft accelerate bitsandbytes \
    huggingface-hub tokenizers protobuf numpy scipy tqdm regex \
    sentencepiece safetensors psutil packaging
```

---

## 六、このスクリプトが適用した修正

### torch.xpu.memory.mem_get_info() 非対応
PyTorch issue #164057、Arc A770 WSL2/Linux ドライバーはこの API を実装していません。
**修正**: monkey-patch で固定値を返す。

### torch.xpu.get_device_properties() がクラッシュする可能性
**修正**: 例外時に FakeProps を返す。

### WSL2 下での Intel XPU 顕存割り当て関数の欠落
**修正**: 環境変数 `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` と `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True` を設定。

### transformers caching_allocator_warmup が OOM を引き起こす
**修正**: `import unsloth` の前に無効化。

### Triton JIT コンパイルが遅い（Intel XPU の共通問題）
**修正**: `TRITON_CACHE_DIR` を設定してコンパイル結果をキャッシュ、`IPEX_XPU_ONEDNN_LAYOUT=1` を設定してメモリスループットを加速。

### Unsloth fix_untrained_tokens と meta tensor の競合
**修正**: この関数を無効化。

---

## 七、完全なトレーニングコード（v11 最適化版）

リリースまたはリポジトリに提供されています。自分で探してください。最終的に動作するバージョンは v12 であり、最初の 11 バージョンは上記の問題を完全に修正していませんでした。

---

## 八、ワンクリックリビルドスクリプト

環境を破壊した場合、このスクリプトを実行してリビルドしてください：

```bash
set -e

echo ">>> 環境のリビルドを開始..."

# 1. 古い環境を削除
rm -rf ~/unsloth_env

# 2. 新しい環境を作成
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# 3. pip をアップグレード
pip install --upgrade pip setuptools wheel

# 4. PyTorch XPU をインストール
echo ">>> PyTorch XPU をインストール..."
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu \
    intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt \
    pytorch-triton-xpu tcmlib umf intel-pti \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple

# 5. PyTorch XPU を確認
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.is_available())"

# 6. Triton XPU を確認
echo ">>> Triton XPU を確認..."
python -c "
import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

x = torch.rand(128, device='xpu')
out = torch.empty_like(x)
n_elements = x.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
test_kernel[grid](x, out, n_elements, BLOCK_SIZE=128)
print('Triton XPU test passed!')
"

# 7. Unsloth をインストール（PyTorch を上書きしない）
echo ">>> Unsloth をインストール..."
pip install --no-deps unsloth unsloth-zoo

# 8. その他の依存関係を手動インストール（xformers と triton はスキップ）
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0 \
    cut_cross_entropy hf_transfer msgspec torchao tyro diffusers \
    nest-asyncio pydantic peft accelerate bitsandbytes \
    huggingface-hub tokenizers protobuf numpy scipy tqdm regex \
    sentencepiece safetensors psutil packaging
```

---

## 九、トレーニングの実行

**期待される出力**:
- モデルが `xpu:0` に読み込まれ、VRAM 使用量は約 8-10GB
- 最初のステップは遅い可能性がある（Triton JIT コンパイル）、約 10-20 秒
- 2 ステップ目からは安定して約 **11-15 秒/ステップ**
- GPU 利用率 70-85%

---

## 十、トラブルシューティング

| 問題 | 現象 | 解決策 |
|------|------|--------|
| **Ubuntu 22.04 ドライバーパッケージ名が違う** | `libze1` が見つからない、`sycl-ls` エラー | **24.04 (noble)** に変更；パッケージ名は `libze-dev` |
| **oneAPI が LD_LIBRARY_PATH を汚染** | PyTorch が `libur_loader.so` バージョン競合を報告 | `/etc/profile.d/oneapi.sh` を削除し、oneAPI 環境変数を読み込まない |
| **汎用 triton が xpu 版を上書き** | `0 active drivers` または `cannot import intel` | **`pip install triton` はしない**、`pytorch-triton-xpu` のみを使用 |
| **bitsandbytes 4-bit が XPU をサポートしない** | `cdequantize_blockwise_fp32` エラー | bf16 読み込みに変更し、4-bit を使用しない |
| **accelerate device_map トレーニング競合** | `Can't train model loaded with device_map='auto'` | モデルを GPU 全体に読み込む（`device_map="xpu"` + `low_cpu_mem_usage=True`） |
| **meta tensor backward エラー** | `Cannot copy out of meta tensor` | モデルが GPU 上に完全にあることを確認し、CPU へのオフロードをしない |
| **Triton JIT コンパイルが極めて遅い** | 最初のステップに 10-20 分 | 正常な動作；`TRITON_CACHE_DIR` を設定してキャッシュ；後続の再起動は速くなる |
| **Windows ネイティブトレーニングが極めて遅い** | 1523s/it、GPU 利用率 6% | **WSL2 への移行が必須**；Windows 上の Triton XPU バックエンドは最適化されていない |
| **HuggingFace 接続タイムアウト** | `Timed out after 120s` | `local_files_only=True` で強制オフライン読み込み |
| **モデルメモリが二重** | CPU メモリと GPU VRAM にそれぞれコピーがある | `low_cpu_mem_usage=True` + `device_map="xpu"` |

---

## 十一、パフォーマンス比較

| 環境 | モデル | 速度 | GPU 利用率 |
|------|--------|------|-----------|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3.5-4B bf16 | **11-15s/it** | **70-85%** |

---

## 十二、今後の予定アップデート

1. **ターミナル GUI の追加**: 使いやすさの向上
2. **最終エクスポート merged と gguf**: `llama.cpp の自動ダウンロードとエクスポート
3. **4bit モデルサポート**: Intel XPU bitsandbytes 4bit サポートはひどく壊れているので、私が修正するのを待ってください

---

> **最後に**: トレーニングが成功しますように！

---

---

# <a id="русский"></a>Русский

# Intel Arc A770 + Unsloth Руководство по двухплатформенной тонкой настройке

> **Windows**: Использование Intel Arc A770 16GB на Windows 11 для LoRA тонкой настройки 4-битных квантованных моделей (полноточные модели, вероятно, тоже работают; здесь в качестве примера используется Qwen3-8b-bnb-4bit). Поскольку бэкенд Intel XPU для Triton не был полностью протестирован на Windows, прямой запуск вызывает множество проблем совместимости MSVC/GCC. Часть этого репозитория направлена на предоставление решения.
>
> **WSL2/Linux**: Неквантованные модели с исходными весами < 16GB (например, Qwen3.5-4B, Qwen3-1.7B и т.д.). В этом руководстве в качестве устройства используется Intel Arc A770 16GB, а в качестве модели — Qwen3.5-4B.

---

# Intel Arc A770 + Unsloth + Windows Скрипт тонкой настройки

> **Сценарий использования**: LoRA тонкая настройка 4-битных квантованных моделей (полноточные модели, вероятно, тоже работают; здесь в качестве примера используется Qwen3-8b-bnb-4bit) с использованием Intel Arc A770 16GB на Windows 11.
>
> Поскольку бэкенд Intel XPU для Triton не был полностью протестирован на Windows, прямой запуск вызывает множество проблем совместимости MSVC/GCC. Часть этого репозитория направлена на предоставление решения.

---

## I. Функции

- **Qwen3-8B-BNB (только пример; тестируйте другие модели самостоятельно)** модель Unsloth LoRA тонкой настройки на **Windows 11** с использованием **Intel Arc A770**
- Автоматическое обнаружение и загрузка среды компиляции Intel oneAPI + MSVC
- Автоматическое исправление проблем совместимости параметров GCC Triton под Windows MSVC
- Автоматическое исправление проблем с отсутствующими путями заголовков/библиотек MSVC
- Поддержка экспорта LoRA / 16-битных полных весов / GGUF после обучения (поскольку полный запуск не был завершен, неизвестно, работает ли окончательное преобразование в полные веса и GGUF; тестируйте терпеливо на свое усмотрение)

---

## II. Предварительные требования

### Аппаратное обеспечение
- **GPU**: Дискретный GPU Intel Arc (интегрированная графика теоретически работает, но не тестировалась; B-серия теоретически работает, но не тестировалась)
- **ОС**: Windows 11

### Программное обеспечение

| Компонент | Версия/Требование | Назначение |
|-----------|------------------|------------|
| Драйвер графики Intel Arc | Последняя версия (31.0.101.xxx+) | GPU вычисления |
| Intel oneAPI Base Toolkit & DeepLearning Toolkit | 2025.2 и последняя | SYCL / Level Zero рантайм |
| Level Zero SDK | 1.28.x - 1.30.x | Бэкенд Triton XPU |
| Visual Studio 2022 | Community/Professional/Enterprise | Компилятор MSVC C++ |
| Python | 3.13 (Windows версия) | Среда выполнения |
| PyTorch | 2.12.1+xpu (официальный wheel Intel) | Фреймворк глубокого обучения XPU |
| Unsloth | 2026.6.9 | Фреймворк быстрой тонкой настройки |

### Обязательные рабочие нагрузки VS 2022
- **"Разработка классических приложений на C++"**
- **MSVC v143 - VS 2022 C++ x64/x86 инструменты сборки**
- **Windows 11 SDK**

---

## III. Поддерживаемые модели

| Модель | Формат | Статус |
|--------|--------|--------|
| Qwen3-8B-BNB | 4-bit BNB (предварительная квантование unsloth) | ✅ Проверено работает, но крайне медленно |

> Другие модели не тестировались. Теоретически любая модель, поддерживаемая Unsloth и загружаемая через BNB 4-bit, должна работать, но могут потребоваться дополнительные настройки.

---

## IV. Исправленные баги для Intel Arc

### 1. Параметры GCC Triton передаются MSVC, вызывая ошибку D8021
**Симптом**: `cl: Ошибка командной строки D8021 : недопустимый числовой аргумент "/Wno-psabi"`  
**Причина**: Бэкенд Triton Intel XPU генерирует команды компиляции в стиле GCC и передает их напрямую `cl.exe`  
**Исправление**: Перехват `triton.runtime.build._build`, фильтрация параметров GCC, таких как `-Wno-psabi`, `-Wno-deprecated-declarations`, `-fPIC`, и преобразование `-D`/`-I`/`-L`/`-l`/`-shared` в стиль MSVC `/D`/`-I`/`-LIBPATH:`/`lib`/`/LD`

### 2. MSVC не может найти заголовки стандартной библиотеки C++ (`cstddef` и т.д.)
**Симптом**: `fatal error C1083: Не удается открыть включаемый файл: "cstddef"`  
**Причина**: `vcvars64.bat` устанавливает только `PATH`, а не переменные среды `INCLUDE` и `LIB`  
**Исправление**: Автоматическое определение корневого каталога цепочки инструментов MSVC из пути `cl.exe` и дополнение `INCLUDE` (MSVC include + Windows SDK ucrt/shared/um + ATL/MFC) и `LIB` (MSVC lib/x64 + Windows SDK lib)

### 3. Заголовки SYCL требуют C++17
**Симптом**: `error C2338: static_assert failed: 'DPCPP does not support C++ version earlier than C++17.'`  
**Причина**: MSVC по умолчанию использует C++14, а заголовки SYCL проверяют версию через макрос `__cplusplus`  
**Исправление**: Добавление `/std:c++17` и `/Zc:__cplusplus` в команды компиляции (последнее заставляет MSVC правильно установить макрос `__cplusplus` на `201703L`)

### 4. Компоновщик не может найти `python313.lib`
**Симптом**: `LINK : Фатальная ошибка LNK1104: не удается открыть файл "python313.lib"`  
**Причина**: При компиляции расширений Python (`.pyd`) Triton `library_dirs` содержит только `Library/bin` и `Library/lib`, но не `libs`  
**Исправление**: Автоматическое обнаружение каталога `Python313/libs` и добавление его в `/LIBPATH`

### 5. Компоновщик требует точку входа (отсутствует `/LD`)
**Симптом**: `LINK : Фатальная ошибка LNK1561: точка входа должна быть определена`  
**Причина**: `.pyd` по сути является DLL и требует флаг `/LD`, который отсутствует в команде компиляции  
**Исправление**: Добавление `/LD` (создание DLL) в команду компиляции

### 6. Чрезвычайно медленная JIT-компиляция Triton (кэш не работает)
**Симптом**: Первый шаг обучения занимает 20+ минут, использование GPU близко к 0%  
**Причина**: Механизм кэширования JIT-ядер бэкенда Triton Intel XPU на Windows имеет проблемы, потенциально перекомпилируя SPIR-V на каждом шаге  
**Смягчение**: Установка `TRITON_CACHE_DIR` и `TRITON_DISABLE_AUTOTUNE=1` для уменьшения накладных расходов на повторную компиляцию

---

## V. Оставшиеся проблемы

- **Чрезвычайно медленная скорость обучения**: Даже после исправления проблем компиляции, эффективность выполнения JIT-ядер бэкенда Triton XPU на Windows намного ниже, чем на Linux. Использование GPU остается ниже 10% в течение длительных периодов, и один шаг обучения по-прежнему занимает несколько минут до десятков минут.
- **Кэш Triton не полностью надежен**: `TRITON_CACHE_DIR` иногда не срабатывает, вызывая многократную перекомпиляцию одного и того же ядра.
- **Несоответствие версий Level Zero SDK**: Переменная среды `ZE_PATH` указывает на 1.30.0, но команды компиляции Triton могут ссылаться на пути 1.28.2; требуется ручное унифицирование.
- **xformers не поддерживается**: Intel XPU не может использовать xformers (только CUDA), поэтому некоторые оптимизации FlashAttention от Unsloth неэффективны.
- **Различия mem_get_info между платформами**: `torch.xpu.memory.mem_get_info()` работает на драйверах Windows Intel Arc, но недоступен на WSL2/Linux (кроссплатформенные скрипты должны учитывать это).

---

## VI. Примечания к версиям (Release)

Восьмая версия — первая, которая может фактически начать запуск модели; первые семь версий не полностью исправили баги для нормальной отладки.

---

## VII. Быстрый старт

```powershell
# 1. Убедитесь, что все необходимое ПО установлено (см. выше)
# 2. Скачайте Release и распакуйте
# 3. Измените секцию CONFIG в верхней части скрипта (путь к модели, путь к датасету и т.д.)
# 4. Запустите
```

---

## VIII. Конфигурация

```python
CONFIG = {
    "model_path": r"H:/Qwen3-8B-unsloth-bnb-4bit",      # Путь к модели
    "dataset_path": r"D:/dataset.json",                  # Путь к датасету
    "output_dir": r"H:/unsloth_train/outputs",          # Выходной каталог
    "max_seq_length": 1024,
    "lora_r": 16,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "batch_size": 1,        # Настроить в зависимости от VRAM
    "grad_accum": 4,        # Общий batch = batch_size * grad_accum
    "max_steps": 3000,
    "warmup_steps": 5,
}
```

---

## IX. Устранение неполадок

| Проблема | Симптом | Решение |
|----------|---------|---------|
| **Ошибка параметра Triton GCC** | `D8021 : недопустимый числовой аргумент "/Wno-psabi"` | Используйте v2+; параметры GCC отфильтрованы |
| **Не найдены заголовки C++** | `fatal error C1083: "cstddef"` | Используйте v4+; INCLUDE исправлен автоматически |
| **SYCL требует C++17** | `DPCPP does not support C++ version earlier than C++17` | Используйте v5+; добавлен `/std:c++17` |
| **Не найден python313.lib** | `LNK1104: не удается открыть файл "python313.lib"` | Используйте v6+; библиотеки Python добавлены автоматически |
| **Точка входа должна быть определена** | `LNK1561: точка входа должна быть определена` | Используйте v7+; добавлен `/LD` |
| **Первый шаг обучения чрезвычайно медленный** | 20+ минут, использование GPU 0% | Используйте v8; установите `TRITON_CACHE_DIR`; если все еще очень медленно, перейдите на WSL2/Linux |
| **Неприемлемая скорость нативного обучения на Windows** | 1523s/it, использование GPU 6% | **Необходим переход на WSL2/Linux**; бэкенд Triton XPU на Windows не оптимизирован |
| **Несоответствие версий Level Zero** | В командах компиляции появляются пути разных версий | Унифицируйте переменную среды `ZE_PATH` с фактически установленной версией SDK |

---

## X. Сравнение производительности

| Среда | Модель | Скорость | Использование GPU |
|-------|--------|----------|-------------------|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3-8B bnb-4bit | 11-15s/it | 70-85% |

> **Вывод**: На Windows мы можем решить только проблему "будет ли это работать", но не проблему "будет ли это работать быстро". Для реального обучения настоятельно рекомендуется переход на WSL2/Linux.

---

## XI. Скрипт восстановления в один клик (проверка среды Windows)

```powershell
# Проверьте необходимые переменные среды
$env:ZE_PATH
$env:CC

# Проверьте установку VS 2022
Test-Path "C:\Program Files\Microsoft Visual Studio2\Community\VC\Auxiliary\Buildcvars64.bat"

# Проверьте версию Python
python --version  # Должно быть 3.13

# Проверьте PyTorch XPU
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"

# Проверьте Triton
python -c "import triton; print(triton.__version__)"
```

---

## XII. Будущие запланированные обновления

1. **Добавить терминальный GUI**: Улучшить удобство использования
2. **Окончательный экспорт merged & gguf**: Экспорт ожидает тестирования
3. **Оптимизация скорости Windows**: Ожидание официальных исправлений Intel/Triton для производительности бэкенда XPU на Windows или попытка портирования пакетов

---

> **Наконец**: Удачи в обучении! Для реального обучения, пожалуйста, перейдите на WSL2/Linux.

---

---

# Intel Arc A770 + Unsloth + WSL2 Скрипт тонкой настройки

> **Сценарий использования**: Неквантованные модели с исходными весами < 16GB (например, Qwen3.5-4B, Qwen3-1.7B и т.д.). В этом руководстве в качестве устройства используется Intel Arc A770 16GB, а в качестве модели — Qwen3.5-4B.

---

## I. Требования к оборудованию / среде

- **GPU**: Дискретный GPU Intel Arc (интегрированная графика не тестировалась; B-серия теоретически работает, но не тестировалась)
- **ОС**: Windows 11 21H2+, WSL2 включен
- **Дистрибутив WSL2**: Ubuntu 24.04 (Noble). В 22.04 отличаются имена пакетов драйверов Intel GPU и пути репозиториев; в 26.04 версия Python слишком высокая и не подходит.
- **Python**: 3.12

---

## II. Установка WSL2 Ubuntu 24.04

Выполните в Windows PowerShell (администратор):

```powershell
# Обновите WSL
wsl --update

# Установите Ubuntu 24.04
wsl --install Ubuntu-24.04
wsl --set-default Ubuntu-24.04
# Если говорит, что не найдено, серверы Microsoft временно недоступны; скачайте установщик Ubuntu из Store самостоятельно

# Проверьте версию
wsl --list --verbose
# Должно показывать Ubuntu-24.04 Running version 2
```

---

## III. Конфигурация драйвера и рантайма Intel GPU

Войдите в терминал WSL2 Ubuntu 24.04 и выполните (если возникнут проблемы с отсутствующими пакетами, установите их с помощью sudo apt самостоятельно или спросите ИИ):

```bash
# 1. Обновите систему
sudo apt update && sudo apt upgrade -y

# 2. Установите базовые инструменты
sudo apt install -y gpg-agent wget build-essential python3.12-dev

# 3. Добавьте репозиторий Intel GPU (версия Noble)
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified' | \
  sudo tee /etc/apt/sources.list.d/intel.gpu.noble.list

sudo apt update

# 4. Установите рантайм Intel GPU (ключевые пакеты)
sudo apt install -y libze-dev intel-opencl-icd intel-media-va-driver-non-free \
  libmfx1 libvpl2 libegl-mesa0 libegl1-mesa-dev libgbm1 libgl1-mesa-dev \
  libgl1-mesa-dri libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 \
  libxatracker2 mesa-va-drivers mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all

# 5. Добавьте пользователя в группу render (доступ к GPU)
sudo gpasswd -a ${USER} render
newgrp render

# 6. Проверьте видимость GPU
ls /dev/dri
# Должны видеть renderD128 и card0

clinfo | grep "Device Name"
# Должно показывать Intel(R) Arc(TM) A770 Graphics или что-то вроде 0x5860
```

> **⚠️ Абсолютное внимание**:
> - НЕ устанавливайте полный oneAPI Base Toolkit (он загрязнит LD_LIBRARY_PATH и вызовет конфликты библиотек PyTorch)
> - Если вы ранее устанавливали oneAPI и настроили /etc/profile.d/oneapi.sh, **вы ДОЛЖНЫ удалить его**:
>   ```bash
>   sudo rm /etc/profile.d/oneapi.sh
>   ```
> - Если sycl-ls позже сломается из-за конфликтов версий, **это не влияет на обучение PyTorch**; игнорируйте.

---

## IV. Установка среды PyTorch XPU (имя виртуальной среды: unsloth_env)

```bash
# 1. Создайте виртуальную среду
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# 2. Обновите pip
pip install --upgrade pip setuptools wheel

# 3. Установите полный стек PyTorch XPU (включает pytorch-triton-xpu; НЕ устанавливайте triton отдельно)
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu \
    intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt \
    pytorch-triton-xpu tcmlib umf intel-pti \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple

# 4. Проверьте PyTorch XPU
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.is_available())"

# 5. Проверьте Triton XPU (правильный метод проверки)
python -c "
import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

x = torch.rand(128, device='xpu')
out = torch.empty_like(x)
n_elements = x.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
test_kernel[grid](x, out, n_elements, BLOCK_SIZE=128)
print('Triton XPU test passed!')
"
```

> **⚠️ Примечания**:
> - **НЕ делайте** `pip install triton` (это перезапишет pytorch-triton-xpu, вызвав потерю бэкенда Intel XPU, а установленный общий triton не включает поддержку XPU операторов)
> - **НЕ делайте** `pip install xformers` (только CUDA; потянет драйверы NVIDIA)
> - **НЕ делайте** `pip install intel_extension_for_pytorch` (PyTorch 2.7.1+xpu уже имеет нативную поддержку XPU; IPEX вызовет конфликты версий)

---

## V. Установка Unsloth

```bash
source ~/unsloth_env/bin/activate

# 1. Установите Unsloth (должно быть без зависимостей)
pip install --no-deps unsloth unsloth-zoo

# 2. Вручную установите другие зависимости Unsloth (пропустите xformers и triton)
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0 \
    cut_cross_entropy hf_transfer msgspec torchao tyro diffusers \
    nest-asyncio pydantic peft accelerate bitsandbytes \
    huggingface-hub tokenizers protobuf numpy scipy tqdm regex \
    sentencepiece safetensors psutil packaging
```

---

## VI. Исправления, примененные этим скриптом

### torch.xpu.memory.mem_get_info() не поддерживается
PyTorch issue #164057, драйвер Arc A770 WSL2/Linux не реализовал этот API.
**Исправление**: Monkey-patch для возврата фиксированных значений.

### torch.xpu.get_device_properties() может аварийно завершиться
**Исправление**: Возврат FakeProps при исключении.

### Intel XPU отсутствуют функции выделения VRAM под WSL2
**Исправление**: Установка переменных среды `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` и `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True`.

### transformers caching_allocator_warmup вызывает OOM
**Исправление**: Отключить перед `import unsloth`.

### Медленная JIT-компиляция Triton (общая проблема Intel XPU)
**Исправление**: Установка `TRITON_CACHE_DIR` для кэширования результатов компиляции; установка `IPEX_XPU_ONEDNN_LAYOUT=1` для ускорения пропускной способности памяти.

### Конфликт Unsloth fix_untrained_tokens с meta tensor
**Исправление**: Отключить эту функцию.

---

## VII. Полный код обучения (оптимизированная версия v11)

Предоставлено в релизах или репозитории; найдите его самостоятельно. Окончательная рабочая версия — v12; первые 11 версий не полностью исправили вышеуказанные проблемы.

---

## VIII. Скрипт восстановления в один клик

Если вы сломали свою среду, запустите этот скрипт для восстановления:

```bash
set -e

echo ">>> Начинается восстановление среды..."

# 1. Удалите старую среду
rm -rf ~/unsloth_env

# 2. Создайте новую среду
python3 -m venv ~/unsloth_env
source ~/unsloth_env/bin/activate

# 3. Обновите pip
pip install --upgrade pip setuptools wheel

# 4. Установите PyTorch XPU
echo ">>> Установка PyTorch XPU..."
pip install torch==2.7.1+xpu torchvision==0.22.1+xpu torchaudio==2.7.1+xpu \
    intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt \
    pytorch-triton-xpu tcmlib umf intel-pti \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple

# 5. Проверьте PyTorch XPU
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.is_available())"

# 6. Проверьте Triton XPU
echo ">>> Проверка Triton XPU..."
python -c "
import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

x = torch.rand(128, device='xpu')
out = torch.empty_like(x)
n_elements = x.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
test_kernel[grid](x, out, n_elements, BLOCK_SIZE=128)
print('Triton XPU test passed!')
"

# 7. Установите Unsloth (не перезаписывайте PyTorch)
echo ">>> Установка Unsloth..."
pip install --no-deps unsloth unsloth-zoo

# 8. Вручную установите другие зависимости (пропустите xformers и triton)
pip install transformers==5.5.0 datasets==4.3.0 trl==0.24.0 \
    cut_cross_entropy hf_transfer msgspec torchao tyro diffusers \
    nest-asyncio pydantic peft accelerate bitsandbytes \
    huggingface-hub tokenizers protobuf numpy scipy tqdm regex \
    sentencepiece safetensors psutil packaging
```

---

## IX. Запуск обучения

**Ожидаемый вывод**:
- Модель загружена в `xpu:0`, использование VRAM примерно 8-10GB
- Первый шаг может быть медленным (JIT-компиляция Triton), примерно 10-20 секунд
- Со второго шага стабильно примерно **11-15 секунд/шаг**
- Использование GPU 70-85%

---

## X. Устранение неполадок

| Проблема | Симптом | Решение |
|----------|---------|---------|
| **Неверное имя пакета драйвера Ubuntu 22.04** | `libze1` не найден, ошибки `sycl-ls` | Перейдите на **24.04 (noble)**; имя пакета — `libze-dev` |
| **oneAPI загрязняет LD_LIBRARY_PATH** | PyTorch сообщает о конфликте версий `libur_loader.so` | Удалите `/etc/profile.d/oneapi.sh`, не загружайте переменные среды oneAPI |
| **Общий triton перезаписывает версию xpu** | `0 active drivers` или `cannot import intel` | **НЕ делайте** `pip install triton`; используйте только `pytorch-triton-xpu` |
| **bitsandbytes 4-bit не поддерживается на XPU** | Ошибка `cdequantize_blockwise_fp32` | Используйте загрузку bf16 вместо 4-bit |
| **Конфликт обучения accelerate device_map** | `Can't train model loaded with device_map='auto'` | Загрузите всю модель в GPU (`device_map="xpu"` + `low_cpu_mem_usage=True`) |
| **Ошибка meta tensor backward** | `Cannot copy out of meta tensor` | Убедитесь, что модель полностью на GPU, без оффлоада на CPU |
| **Чрезвычайно медленная JIT-компиляция Triton** | Первый шаг 10-20 минут | Нормальное поведение; установите `TRITON_CACHE_DIR` для кэширования; последующие перезапуски будут быстрее |
| **Чрезвычайно медленное нативное обучение на Windows** | 1523s/it, использование GPU 6% | **Необходим переход на WSL2**; бэкенд Triton XPU на Windows не оптимизирован |
| **Таймаут сети HuggingFace** | `Timed out after 120s` | `local_files_only=True` для принудительной автономной загрузки |
| **Дублирование памяти модели** | Копия в CPU RAM и GPU VRAM | `low_cpu_mem_usage=True` + `device_map="xpu"` |

---

## XI. Сравнение производительности

| Среда | Модель | Скорость | Использование GPU |
|-------|--------|----------|-------------------|
| Windows 11 | Qwen3-8B bnb-4bit | 1523s/it | 6% |
| WSL2 Ubuntu 24.04 | Qwen3.5-4B bf16 | **11-15s/it** | **70-85%** |

---

## XII. Будущие запланированные обновления

1. **Добавить терминальный GUI**: Улучшить удобство использования
2. **Окончательный экспорт merged и gguf**: `Автоматическая загрузка llama.cpp для экспорта
3. **Поддержка 4-битных моделей**: Поддержка bitsandbytes 4-bit для Intel XPU ужасно сломана; я исправлю это, когда доберусь

---

> **Наконец**: Удачи в обучении!
