<h1 align="center">StrataRTX</h1>

<p align="center"><b>Specialized NVIDIA RTX / CUDA Fork of Strata Engine</b><br>
Optimized for Multi-GPU Ampere (SM86), Ada Lovelace & Blackwell · Extreme Memory Hierarchy Tiering · Speculative Decoding</p>

<p align="center"><img src="Source/docs/media/how-it-works.svg" width="860" alt="Strata memory tiering across GPU, CPU, RAM and SSD"></p>

StrataRTX is a dedicated performance fork of [Strata](https://github.com/Niko1221/Strata), refactored and optimized exclusively for NVIDIA hardware. It focuses on multi-GPU pipeline balancing, native Ampere (SM86) kernel specialization, sub-warp expert scheduling, and zero-overhead memory tiering across host RAM, NVMe storage, and dual-GPU VRAM.

---

## Performance Benchmark: Dual RTX 3060 12GB

Tested on consumer desktop hardware running **Qwen3.8-Flash-Next (180B MoE)**:

| Parameter | Value |
| :--- | :--- |
| **Model** | Qwen3.8-Flash-Next-GSQ-RCO |
| **Base Architecture** | 180B Parameters, 24,576 Sparse Experts (Top-10 Active per Token) |
| **Quantization** | **IQ1_M** weights (~1.75 bpw) · **q4_0** KV Cache |
| **Active Context** | **262,144 tokens (256k)** |
| **Hardware** | **2x NVIDIA GeForce RTX 3060 12GB** (24 GB aggregate VRAM) |
| **Throughput** | **~70 tokens/s generation** · **>1,000 tokens/s prompt prefill** |

<p align="center"><img src="Source/docs/media/guess-and-check.svg" width="860" alt="Speculative decoding guess and check pipeline"></p>

---

## Key Customizations & Upstream Patches

This repository integrates upstream architectural advancements alongside custom hardware tuning:

### 1. Multi-GPU Layer Splitting & Dense Weight Trimming ([PR #716](https://github.com/Niko1221/Strata/pull/716))
* Explicit pipeline partitioning across GPUs (`--layer-split 27`).
* Each GPU retains only its relevant dense layers (`blk.N.*`) in VRAM instead of duplicating redundant layers across both devices (~3.4 GB VRAM freed per card).
* The reclaimed VRAM expands the GPU expert cache from ~8,800 to **10,600+ slots**, boosting VRAM cache hit rates above 98%.
* Layer parking and checkpoint snapshotting updated to orchestrate drainage on per-stage devices independently.

### 2. High-Capacity Quantized Sparse Attention ([PR #646](https://github.com/Niko1221/Strata/pull/646))
* Custom CUDA QSA selection kernels targeting Ampere SM86.
* Eliminates memory bus choke points during long-range context lookups across 256k tokens.
* Zero-doorbell verify graph, sub-warp expert packing, and shared-memory staging for batched PLE/MTP operations.

### 3. Exclusively NVIDIA RTX / CUDA Backend
* Stripped AMD HIP/ROCm compatibility layers, shims, and dead code paths to allow compiler optimizations, fast-math vectorization (`--use_fast_math`), and direct cuBLAS integration.
* Pure CUDA 12.x/13.x toolchain integration with MSVC and Ninja.

### 4. Hardware Tuning & Overclock Profile
* **GPU Core Clocks:** Locked to **2100 MHz** on both RTX 3060s.
* **VRAM Memory Clocks:** Overclocked to **7900 MHz** (+900 MHz offset) to maximize memory bus bandwidth.
* **PCIe Bus frac:** Zero-copy host staging buffer pins (`STRATA_ARENA_PIN_GIB=8`, `STRATA_STAGE_TRIM=1`).

---

## Path to 100+ tokens/s (Bottleneck Analysis)

The current rig delivers 70 tok/s. Hitting triple digits requires lifting the two primary hardware bottlenecks:

1. **System Memory Bandwidth (DDR4 2400 MHz):**
   * The host system runs 48 GB DDR4 at 2400 MHz. Inactive experts not held in VRAM cache must be processed by the CPU or streamed into staging buffers. Moving to 3600+ MHz DDR4 or DDR5 lifts the memory transfer ceiling.
2. **PCIe Bus Bottleneck (PCIe 3.0 via Ryzen 7 5700G):**
   * The AMD Ryzen 7 5700G APU is capped at PCIe Gen 3. Upgrading to a platform with full PCIe 4.0 or 5.0 lanes doubles host-to-device transfer bandwidth for expert offloading and speculative verification passes.

---

## Architecture Overview

<p align="center"><img src="Source/docs/media/runpagoda.png" width="860" alt="Strata engine monitor interface"></p>

* **Graphics Cards (2x RTX 3060 12GB):** Hold dense model cores, projection layers, KV cache (`q4_0`), and ~4,000 to 10,000 frequently used active experts.
* **Host RAM (32-64 GB):** Houses all 24,576 experts in mapped/pinned memory (`VirtualLock` / `cudaHostRegister`). Overflow experts execute on the CPU using AVX2 vector paths simultaneously.
* **NVMe SSD Storage:** Streams large parameter lookup tables (~29 GB) asynchronously on demand without occupying memory.
* **Speculative Decoding:** A compact draft helper predicts sequences 2-4 tokens ahead; the primary 180B model batch-verifies them in a single forward pass for a **1.6x - 1.8x** speedup.

---

## Building from Source

### Prerequisites
* Windows 10/11 x64
* Visual Studio 2022 (MSVC v143) with C++ Desktop Development tools
* NVIDIA CUDA Toolkit 12.8 or 13.x (with matching NVIDIA display driver)
* CMake 3.24+ and Ninja build system
* Python 3.10+ (for runtime server and tools)

### Build Engine (`strata.exe`)
You can use the automated build script:
```cmd
cd Source
build-custom-engine.bat
```

Or manually configure via CMake:
```cmd
mkdir build
cd build
cmake .. -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DSTRATA_BUILD_TESTS=OFF

ninja strata
```

---

## Running StrataRTX

1. Configure your model arguments in `App/strata-coder-iq1_m.json`:
```json
{
  "exe": "engine\\strata.exe",
  "args": [
    "--pack", "C:\\Strata-data\\packs\\coder-iq1_m",
    "--native", "C:\\Strata-data\\models\\coder-IQ1_M\\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf",
    "--expert-profile", "data\\expert-profile-coder.bin",
    "--expert-cache", "auto",
    "--spec", "5",
    "--spec-min-p", "0.62",
    "--max-context", "262144",
    "--kv", "q4_0",
    "--vram-reserve-mib", "750",
    "--pool-workers", "7"
  ],
  "gpu": [0, 1],
  "layer_split": "27"
}
```

2. Start the background tray server:
```cmd
python App\tray_server.py
```
Or launch through the tray executable `App\StrataServer.exe`.

3. Access the web interface at `http://127.0.0.1:8080` or connect any OpenAI/Anthropic-compatible client (`http://127.0.0.1:8080/v1`).

---

## Credits & Upstream Attribution

* **Original Project:** [Strata](https://github.com/Niko1221/Strata) by [Niko1221](https://github.com/Niko1221) and community contributors.
* **Multi-GPU Layer Split Optimization:** [PR #716](https://github.com/Niko1221/Strata/pull/716) by [Evan (eplat)](https://github.com/eplat) & [JeanP00l](https://github.com/JeanP00l).
* **CUDA QSA & Decode Kernels:** [PR #646](https://github.com/Niko1221/Strata/pull/646) by [Stuart Chapin](https://github.com/stuchapin).
* **Base Architecture & Weights:** [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Alibaba Qwen Team. Quantizations by ISTA-DASLab and Unsloth.
* **Underlying Runtimes:** Components adapted from [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp).

---

## License

This project is licensed under the [MIT License](Source/LICENSE) in accordance with the original Strata project.
