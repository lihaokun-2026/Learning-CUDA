# N 体引力模拟器（CUDA）

## 文件

- `nbody.cu`：CUDA 实现。使用共享内存分块计算 $O(N^2)$ 引力，并支持 Euler 与 Leapfrog 积分。
- `visualize.py`：读取二进制轨迹并生成 Matplotlib 二维或三维动画（默认三维）。
- `data/particles.txt`、`data/params.txt`：最小三体回归样例。
- `data/generate_cases.py`：生成多组确定性测试数据。

生成更多测试数据：

```powershell
python data\generate_cases.py
```

生成的案例包括。所有生成案例均设置 `record_interval = 1`，即每个模拟时间步都保存一个轨迹帧；因此 `num_steps = S` 时输出帧数为 `S + 1`（包括初始状态），不会出现只生成几十帧的问题：

`benchmark_4096` 和 `benchmark_65536` 是银河系尺度基准案例，采用一致的物理单位：质量为太阳质量 $M_\odot$，位置为光年（ly），时间为百万年（Myr），速度为 ly/Myr，引力常数为 $G=0.1559\ \mathrm{ly^3/(M_\odot\,Myr^2)}$。恒星盘半径约 50,000 ly、厚度约 3,000 ly，因此它们不是“1000 光年大小的银河系”，而是约 100,000 光年直径的银河系恒星盘近似模型。位置在盘体积内生成，速度根据显式中心黑洞的局部圆轨道速度加入随机扰动；由于模型没有单独实现银河系恒星盘和暗物质的背景势，它是银河系尺度的简化直接 N 体模型。

五类对象及质量范围为：黑洞（1 个，$4.3\times10^6 M_\odot$）、中子星/白矮星（约 0.2%，$0.5\sim2.3 M_\odot$）、恒星（约 1%，$0.08\sim100 M_\odot$）、行星（约 15%，$10^{-8}\sim10^{-3} M_\odot$）和小行星（剩余多数，$10^{-15}\sim10^{-8} M_\odot$）。类别信息和实际生成质量另存为同名的 `*_objects.txt` 文件。`two_body` 等小规模回归案例仍保留无量纲参数，用于积分器正确性测试，不应与银河系单位混用。

| 案例 | 粒子数 | 用途 |
|---|---:|---|
| `two_body` | 2 | 1000 步圆轨道和能量稳定性验证 |
| `solar_system` | 5 | 1000 步中心大质量天体和多轨道扰动 |
| `cluster_64` | 64 | 1000 步三维星团演化、聚团/散射 |
| `disk_1024` | 1025 | 1000 步粒子盘和中等规模性能测试 |
| `benchmark_4096` | 4096 | 基础版验收：1000 步、1001 帧 |
| `benchmark_65536` | 65536 | 进阶版验收：1000 步、1001 帧 |

例如运行 4096 粒子随机天体群并生成约 10 FPS 的 GIF：

```bash
./nbody data/benchmark_4096_particles.txt data/benchmark_4096_params.txt benchmark_4096.bin benchmark_4096.log
python3 visualize.py benchmark_4096.bin --dimension 3d --fps 10 --trail 10 --output benchmark_4096.gif
```

`--fps 10` 通过设置 GIF 帧间隔为约 100 ms 控制播放速度；它不会减少 BIN 中真实保存的 1001 个轨迹帧。若只想直接指定间隔，可使用 `--interval 100`。

注意：1000 步、`dt = 0.01 Myr` 对应总模拟时间 10 Myr；这是用于 CUDA 性能和宏观演化展示的简化模型，不是高精度银河系动力学重建。由于粒子数远少于真实银河系，单个模拟粒子代表一群未显式建模的天体，质量分布是分层抽样而非完整星表。

## 输入格式

银河系基准案例统一采用以下单位：

- 质量：太阳质量 $M_\odot$；
- 位置：光年（ly）；
- 时间：百万年（Myr）；
- 速度：ly/Myr；
- `G = 0.1559`，单位为 $\mathrm{ly^3/(M_\odot\,Myr^2)}$。

银河系恒星盘直径约 100,000 光年，半径约 50,000 光年；1000 光年更接近盘的厚度尺度，而不是银河系直径。

粒子文件每行 7 个浮点数：`x y z vx vy vz mass`。参数文件使用 `key = value`，允许 `#` 注释：

```text
dt = 1e-3
num_steps = 1000
record_interval = 100
G = 1.0
softening = 1e-4
integrator = leapfrog
```

## 输出格式

`trajectory.bin` 为小端二进制：文件头是两个 `int32`（粒子数 `P`、记录数 `R`），之后按**粒子顺序**写入该粒子的全部记录点，每个记录点是三个 `float32`：`x,y,z`。即文件布局为 `[P, R, xyz]`。二进制比 CSV 体积更小，读写和解析开销更低。注意：总时间步数是 `num_steps`，轨迹帧数是 `num_steps / record_interval + 1`；要求的 1000 步不等价于必须输出 1000 帧，但本项目测试案例按每步记录，输出至少 1001 帧。

默认生成 `performance.log`，也可用第 4 个参数指定日志路径。日志包含：

- GPU 模拟耗时、总墙钟时间、轨迹写盘耗时；
- 每步平均耗时、particle-steps/sec、interactions/sec；
- CUDA 设备名、显存占用；
- 小规模 CPU 同算法基准和估算 GPU/CPU 加速比；
- 初末总动量、相对动量误差；
- 初末总能量、相对能量误差。

其中 `gpu_kernel_compute_sec` 是 CUDA 事件测得的 GPU 模拟区间，`host_trajectory_copy_and_record_sec` 是每次把位置从 GPU 拷贝到主机并写入内存轨迹的累计时间，`trajectory_io_sec` 是最终 BIN 文件写盘时间，`wall_total_sec` 是模拟阶段总墙钟时间。可视化程序会另外生成 `<output>.visualization.log`，记录 GIF/MP4 生成耗时 `visualization_total_sec`。

粒子不超过约 4000 时能量使用完整 $O(N^2)$ 计算；更大规模使用固定 800 万对粒子的确定性采样估计，并在日志中标记 `energy_diagnostic=sampled`。

## 编译与运行（有 NVIDIA CUDA 环境的机器）

在本目录执行：

```powershell
nvcc -O3 -std=c++17 -lineinfo nbody.cu -o nbody.exe
.\nbody.exe data\particles.txt data\params.txt trajectory.bin
python -m pip install numpy matplotlib
	python visualize.py trajectory.bin --fps 10 --output trajectory.gif
```

生成 MP4 需要系统安装 FFmpeg：`python visualize.py trajectory.bin --output trajectory.mp4`。

## Linux 测试、正确性验收与计时流程

以下命令在有 NVIDIA GPU、CUDA Toolkit 和 Python 环境的 Linux 机器上执行：

```bash
cd /data/workspace/Learning-CUDA/08_nbody

# 1. 检查 GPU 和 CUDA 工具链
nvidia-smi
nvcc --version

# 2. 生成/更新输入案例（如果数据已从 Windows 同步，可跳过）
python3 data/generate_cases.py

# 3. 编译。RTX 4090 使用 sm_89；其他 GPU 请替换为对应架构
nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 nbody.cu -o nbody

# 4. 先运行小规模双体回归案例
./nbody data/two_body_particles.txt data/two_body_params.txt \
	two_body.bin two_body.performance.log

# 检查双体系统的守恒误差
grep -E 'relative_momentum_error|relative_energy_error|gpu_kernel_compute_sec|wall_total_sec' \
	two_body.performance.log

# 5. 运行 4096 粒子性能案例
./nbody data/benchmark_4096_particles.txt data/benchmark_4096_params.txt \
    benchmark_4096.bin benchmark_4096.performance.log

# 6. 读取 CUDA 程序的性能日志
grep -E 'gpu_kernel_compute_sec|host_trajectory_copy|trajectory_io_sec|wall_total_sec|interactions_per_sec' \
    benchmark_4096.performance.log

# 7. 生成 10 FPS 三维 GIF；程序结束时会打印生成耗时
python3 visualize.py benchmark_4096.bin --dimension 3d --fps 10 --trail 10 \
    --output benchmark_4096.gif --time-log benchmark_4096.visualization.log

# 8. 读取 GIF 生成耗时
cat benchmark_4096.visualization.log
```

建议分别记录以下四个指标：

| 指标 | 来源 | 含义 |
|---|---|---|
| `gpu_kernel_compute_sec` | `*.performance.log` | CUDA kernel 计算时间，不含最终 BIN 写盘 |
| `host_trajectory_copy_and_record_sec` | `*.performance.log` | GPU 到 CPU 的轨迹拷贝和主机记录时间 |
| `trajectory_io_sec` | `*.performance.log` | BIN 轨迹文件写盘时间 |
| `visualization_total_sec` | `*.visualization.log` | Python 读取轨迹、渲染并生成 GIF 的总时间 |

不要用 shell 的 `time ./nbody` 代替 CUDA 事件时间。`time` 包含进程启动、CPU 诊断、内存分配和文件写盘，而 `gpu_kernel_compute_sec` 才是 GPU 计算本身。可以额外使用 `/usr/bin/time -f '%e'` 记录端到端时间，但应单独标注为 shell wall-clock time。

```bash
/usr/bin/time -f 'nbody_process_wall_sec=%e' \
  ./nbody data/benchmark_4096_particles.txt data/benchmark_4096_params.txt \
  benchmark_4096.bin benchmark_4096.performance.log
```

4096 案例确认无误后再运行 65536 案例。直接 N 体算法每一步需要约 $N^2$ 次相互作用，65536 粒子、1000 步计算量很大：

```bash
./nbody data/benchmark_65536_particles.txt data/benchmark_65536_params.txt \
    benchmark_65536.bin benchmark_65536.performance.log
python3 visualize.py benchmark_65536.bin --dimension 3d --fps 10 --trail 0 \
    --output benchmark_65536.gif --time-log benchmark_65536.visualization.log
```

双体系统运行后检查动画是否保持近似圆轨道，同时检查 `two_body.log` 的 `relative_energy_error` 和 `relative_momentum_error`。Leapfrog 的误差应保持有界；具体阈值取决于 `dt` 和软化参数，建议双体案例两项误差均不超过 $10^{-3}$。多体系统重点检查动量误差；采样能量适合观察趋势，不应作为严格精确阈值。

```bash
python3 visualize.py two_body.bin --dimension 3d --fps 10 --trail 20 \
    --output two_body.gif --time-log two_body.visualization.log
cat two_body.performance.log
cat benchmark_4096.performance.log
```

### nvcc 弃用警告

`Support for offline compilation for architectures prior to ... 75 will be removed` 是警告而非错误，编译已经成功。它表示当前 nvcc 默认生成列表包含低于 `sm_75` 的旧 GPU 架构，而未来 CUDA 版本将移除这些架构的离线编译支持。RTX 4090 可显式指定 `-arch=sm_89` 消除该警告；仅隐藏警告可添加 `-Wno-deprecated-gpu-targets`，但显式指定实际 GPU 架构更合适。

无 CUDA 环境时不能执行 `nbody.cu`，但可以先验证可视化读取器：

```powershell
python -m pip install numpy matplotlib
python -c "import visualize; print(visualize.load_trajectory('trajectory.bin').shape)"
```

该命令需要一份由 CUDA 机器生成的 `trajectory.bin`，预期输出为 `(记录数, 粒子数, 3)`。

## 显存估算

当前 CUDA 程序使用三个 `float4[N]` 数组：位置、速度、加速度，每个粒子 16 字节，因此设备端显存约为：

$$M_{GPU} = 3 \times N \times 16 = 48N\text{ bytes}$$

轨迹会在主机内存中保存，约为：

$$M_{trajectory} = (S / R + 1) \times N \times 3 \times 4$$

其中 $N$ 是粒子数，$S$ 是总步数，$R$ 是记录间隔。生成的验收案例使用 $R=1$，所以 1000 步会输出 1001 帧。共享内存每个 block 约为 `256 * 16 = 4096` 字节。

仅按当前设备数组计算：

| 粒子数 | GPU 设备数组 |
|---:|---:|
| 4,096 | 0.19 MiB |
| 65,536 | 3.00 MiB |
| 1,000,000 | 45.78 MiB |
| 10,000,000 | 457.76 MiB |

因此 RTX 4090 的 24 GiB 显存完全可以运行题目要求的 4,096、65,536 粒子规模；甚至从“存储数组”角度可以容纳数亿粒子。但当前直接 N 体算法每一步需要 $N^2$ 次相互作用，真正瓶颈是计算时间而不是显存：$65,536^2 \approx 4.29$ 亿次相互作用/步，1000 步约 4.29 万亿次相互作用，可能需要较长时间。

此外，程序将所有轨迹保存在 CPU 内存中，若 $N=65,536$、记录 1001 次，轨迹约为 0.73 GiB；每步记录会增加主机内存和文件大小。RTX 4090 可以运行，但建议先用 `cluster_64`、`disk_1024` 验证，再执行 `benchmark_4096`，最后执行 `benchmark_65536`。
