# N 体引力模拟器（CUDA）

## 文件

- `nbody.cu`：CUDA 实现。使用共享内存分块计算 $O(N^2)$ 引力，并支持 Euler 与 Leapfrog 积分。
- `visualize.py`：读取二进制轨迹；默认使用 Matplotlib `Axes3D` + `FuncAnimation` 绘制三维动态轨迹，也提供显式 `--backend gpu` 的高性能栅格化 MP4 后端。
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

例如运行 4096 粒子随机天体群并生成 10 FPS、1920×1080 的 GPU 加速 MP4：

```bash
./nbody data/benchmark_4096_particles.txt data/benchmark_4096_params.txt benchmark_4096.bin benchmark_4096.log
python3 visualize.py benchmark_4096.bin --backend gpu --video-encoder auto \
    --fps 10 --trail 10 --z-scale 35 --width 1920 --height 1080 \
    --frame-stride 1 --output benchmark_4096.mp4
```

`--fps 10` 设置 MP4 播放帧率。MP4 默认 `--frame-stride 1`，即完整输出 1001 个轨迹记录；银河盘的 $x/y$ 半径约为 50,000 ly，而原始 $z$ 半厚度约为 1,500 ly，尺度相差约 33 倍，因此使用 `--z-scale 35` 在渲染投影中放大垂直方向，避免所有粒子看起来堆在同一个圆盘上；该参数只改变可视化比例，不改变模拟数据。

GPU 后端使用固定观察方向，不再让相机随帧旋转。五类天体会自动读取 `data/benchmark_*_objects.txt`，分别使用不同颜色和大小：中心黑洞为大型黄色点，中子星/白矮星为青色，恒星为橙色，行星为绿色，小行星为灰白色。MP4 支持高分辨率、全彩画面和较小文件体积，作为正式主输出；GIF 只用于兼容预览。

`--video-encoder auto` 会自动选择编码方式：RTX 4090 等带 NVENC 的普通消费级显卡使用 `h264_nvenc` 硬件编码；A800/A100 本身没有 NVENC 视频编码单元，因此仍由 A800 CUDA 加速粒子投影和栅格化，再自动使用跨平台 `libx264` 完成 MP4 压缩。该设计不依赖 Ada 等新架构特性，CUDA 渲染可运行于 A800、A100、RTX 30/40 系列等常见 CUDA GPU。

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

### Matplotlib 三维动画（题目验收模式）

题目要求的三维动态轨迹必须使用 Matplotlib 实现，因此验收时使用
`--backend matplotlib --dimension 3d`，不要用 GPU 栅格化后端替代它：

```bash
python3 visualize.py benchmark_4096.bin \
    --backend matplotlib --dimension 3d \
    --objects data/benchmark_4096_objects.txt \
    --fps 10 --trail 80 --z-scale 35 \
    --output benchmark_4096_matplotlib.mp4
```

该模式确实创建 `projection="3d"` 的 Matplotlib 坐标轴，并通过
`FuncAnimation` 逐帧更新五类天体的三维坐标。`z` 轴按显示比例放大，仅用于克服
银河盘半径约 50,000 ly、半厚度约 1,500 ly 的尺度差异；数据本身不被改变。
颜色和大小分别表示黑洞、中子星/白矮星、恒星、行星和小行星；三维尾迹使用
`Line3DCollection` 批量绘制，4096 粒子最多显示 1024 条轨迹，65536 粒子最多显示
512 条轨迹。尾迹长度由 `--trail` 控制，例如 `--trail 80` 会显示每个代表粒子最近
80 个时间帧的三维运动路径。65536 粒子时只对小行星抽样显示，避免 Matplotlib 创建
数万条独立轨迹线导致画面糊成一片，但 BIN 中仍保留全部粒子和全部帧。

`--backend gpu` 是面向大规模输出的性能模式：它使用 CUDA 投影和 FFmpeg 生成
栅格化 MP4，不满足“Matplotlib 3D + FuncAnimation”这一验收条件。两种模式可以
同时保留，前者用于题目展示，后者用于高分辨率性能输出。

## Linux 完整测试、正确性验收与计时流程

以下命令均在 `08_nbody` 目录执行。示例路径需要替换为服务器上的真实路径：

```bash
cd /public/home/lvxy/lhk/projects/cuda-task-4.0/08_nbody
```

### 1. 检查系统、CUDA 和 Python 环境

```bash
nvidia-smi
nvcc --version
python3 --version
/usr/bin/time --version
ffmpeg -version
ffprobe -version
```

安装 Python 依赖。PyTorch 必须是与服务器 NVIDIA 驱动兼容的 CUDA 版本：

```bash
python3 -m pip install numpy torch
python3 -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

`cuda=True` 才能使用 `--backend gpu`。同时检查 FFmpeg 编码器：

```bash
ffmpeg -hide_banner -encoders | grep -E 'libx264|h264_nvenc'
```

- A800/A100 没有 NVENC，预期使用 A800 CUDA 渲染加 `libx264` 编码；
- RTX 4090 有 NVENC，且 FFmpeg 包含 `h264_nvenc` 时会自动硬件编码；
- 两类显卡均使用同一份 `visualize.py`，无需修改代码。

### 2. 生成并检查测试数据

如果仓库中已有最新数据可以跳过生成，否则执行：

```bash
python3 data/generate_cases.py
wc -l data/benchmark_4096_particles.txt data/benchmark_65536_particles.txt
wc -l data/benchmark_4096_objects.txt data/benchmark_65536_objects.txt
cat data/benchmark_4096_params.txt
cat data/benchmark_65536_params.txt
```

预期：

- 粒子文件分别为 4096 行和 65536 行；
- 对象文件包含表头，分别为 4097 行和 65537 行；
- 两个参数文件均为 `num_steps = 1000`、`record_interval = 1`、`G = 0.1559` 和 `integrator = leapfrog`。

### 3. 编译 CUDA 程序

A800 使用 Ampere `sm_80`：

```bash
nvcc -O3 -std=c++17 -lineinfo -arch=sm_80 nbody.cu -o nbody
```

RTX 4090 使用 Ada `sm_89`：

```bash
nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 nbody.cu -o nbody
```

若同一可执行文件需要兼容 A800 和 RTX 4090，可同时嵌入两个架构：

```bash
nvcc -O3 -std=c++17 -lineinfo \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_89,code=sm_89 \
    nbody.cu -o nbody
```

检查可执行文件：

```bash
ls -lh nbody
```

### 4. 先运行双体正确性回归

```bash
./nbody data/two_body_particles.txt data/two_body_params.txt \
    two_body.bin two_body.performance.log

grep -E 'relative_momentum_error|relative_energy_error|gpu_kernel_compute_sec|wall_total_sec' \
    two_body.performance.log
```

Leapfrog 误差应保持有界，建议双体案例的 `relative_energy_error` 和 `relative_momentum_error` 均不超过 $10^{-3}$。确认双体案例正常后再运行大规模案例。

### 5. 运行并验收 4096 粒子案例

记录程序端到端时间、CPU 时间和最大常驻内存：

```bash
    ./nbody \
    data/benchmark_4096_particles.txt \
    data/benchmark_4096_params.txt \
    benchmark_4096.bin \
    benchmark_4096.performance.log
```

检查模拟结果和 CUDA 性能日志：

```bash
ls -lh benchmark_4096.bin
cat benchmark_4096.process.time
grep -E 'gpu_kernel_compute_sec|gpu_simulation_sec|host_trajectory_copy_and_record_sec|trajectory_io_sec|wall_total_sec|interactions_per_sec|relative_momentum_error|relative_energy_error|energy_diagnostic' \
    benchmark_4096.performance.log
```

检查 BIN 形状和数值有效性：

```bash
python3 -c "import numpy as np, visualize; x=visualize.load_trajectory('benchmark_4096.bin'); print('shape=', x.shape); print('finite=', np.isfinite(x).all()); print('min=', x.min(axis=(0,1))); print('max=', x.max(axis=(0,1)))"
```

预期为 `shape=(1001, 4096, 3)` 且 `finite=True`。

使用 GPU 固定视角渲染 1920×1080、10 FPS、1001 帧 MP4：

```bash
/usr/bin/time \
    -f 'visualizer_process_wall_sec=%e\nuser_sec=%U\nsystem_sec=%S\nmax_rss_kb=%M' \
    -o benchmark_4096.visualizer.process.time \
    python3 visualize.py benchmark_4096.bin \
    --backend gpu \
    --video-encoder auto \
    --objects data/benchmark_4096_objects.txt \
    --fps 10 \
    --trail 10 \
    --z-scale 35 \
    --width 1920 \
    --height 1080 \
    --frame-stride 1 \
    --output benchmark_4096.mp4 \
    --time-log benchmark_4096.visualization.log
```

检查 MP4 和可视化计时：

```bash
ls -lh benchmark_4096.mp4
cat benchmark_4096.visualization.log
cat benchmark_4096.visualizer.process.time
ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames \
    -of default=noprint_wrappers=1 benchmark_4096.mp4
```

预期：`codec_name=h264`、`width=1920`、`height=1080`、`r_frame_rate=10/1`、`nb_frames=1001`。

### 6. 运行并验收 65536 粒子案例

4096 案例完全通过后执行。该案例约进行 $4.29\times10^{12}$ 次粒子相互作用，并生成约 0.73 GiB 的 BIN 轨迹，运行前应确认磁盘和主机内存充足。

```bash
/usr/bin/time \
    -f 'process_wall_sec=%e\nuser_sec=%U\nsystem_sec=%S\nmax_rss_kb=%M' \
    -o benchmark_65536.process.time \
    ./nbody \
    data/benchmark_65536_particles.txt \
    data/benchmark_65536_params.txt \
    benchmark_65536.bin \
    benchmark_65536.performance.log
```

检查模拟结果：

```bash
ls -lh benchmark_65536.bin
cat benchmark_65536.process.time
grep -E 'gpu_kernel_compute_sec|gpu_simulation_sec|host_trajectory_copy_and_record_sec|trajectory_io_sec|wall_total_sec|interactions_per_sec|relative_momentum_error|relative_energy_error|energy_diagnostic' \
    benchmark_65536.performance.log

python3 -c "import numpy as np, visualize; x=visualize.load_trajectory('benchmark_65536.bin'); print('shape=', x.shape); print('finite=', np.isfinite(x).all()); print('min=', x.min(axis=(0,1))); print('max=', x.max(axis=(0,1)))"
```

预期为 `shape=(1001, 65536, 3)` 且 `finite=True`。

生成分辨率高于 4096 案例的 2560×1440 MP4：

```bash
/usr/bin/time \
    -f 'visualizer_process_wall_sec=%e\nuser_sec=%U\nsystem_sec=%S\nmax_rss_kb=%M' \
    -o benchmark_65536.visualizer.process.time \
    python3 visualize.py benchmark_65536.bin \
    --backend gpu \
    --video-encoder auto \
    --objects data/benchmark_65536_objects.txt \
    --fps 10 \
    --trail 10 \
    --z-scale 35 \
    --width 2560 \
    --height 1440 \
    --frame-stride 1 \
    --output benchmark_65536.mp4 \
    --time-log benchmark_65536.visualization.log
```

检查输出：

```bash
ls -lh benchmark_65536.mp4
cat benchmark_65536.visualization.log
cat benchmark_65536.visualizer.process.time
ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames \
    -of default=noprint_wrappers=1 benchmark_65536.mp4
```

预期：`codec_name=h264`、`width=2560`、`height=1440`、`r_frame_rate=10/1`、`nb_frames=1001`。

### 7. 视频视觉验收标准

两个 MP4 均应满足：

- 固定观察视角，不随时间旋转；
- `--z-scale 35` 后能明显观察到垂直厚度和运动，不再是完全平铺的圆盘；
- 黑洞为大型黄色点，中子星/白矮星为青色，恒星为橙色，行星为绿色，小行星为灰白色；
- 粒子运动和尾迹连续，没有空帧、NaN 或画面损坏；
- 65536 粒子视频为 2560×1440，分辨率高于 4096 粒子的 1920×1080；
- A800 日志预期包含 `backend=gpu`、`gpu_name=NVIDIA A800...`、`video_encoder=libx264`；
- RTX 4090 且 FFmpeg 支持 NVENC 时预期包含 `video_encoder=h264_nvenc`。

若画面仍然过薄，只需重新执行可视化命令并改为 `--z-scale 50`；若垂直拉伸过大，改为 `--z-scale 25`，无需重新运行 N 体模拟。

### 8. 计时指标说明

| 指标 | 来源 | 含义 |
|---|---|---|
| `gpu_kernel_compute_sec` | `*.performance.log` | CUDA 模拟区间时间 |
| `host_trajectory_copy_and_record_sec` | `*.performance.log` | GPU 到 CPU 轨迹拷贝与主机内存记录时间 |
| `trajectory_io_sec` | `*.performance.log` | BIN 轨迹写盘时间 |
| `wall_total_sec` | `*.performance.log` | N 体程序内部测得的总墙钟时间 |
| `gpu_render_and_transfer_sec` | `*.visualization.log` | CUDA 投影、栅格化以及 RGB 帧传回 CPU 的累计时间 |
| `video_encoding_and_pipe_wait_sec` | `*.visualization.log` | FFmpeg H.264 编码和管道等待时间 |
| `visualization_total_sec` | `*.visualization.log` | BIN 读取、GPU 渲染和 MP4 编码的端到端时间 |
| `process_wall_sec` | `*.process.time` | Linux 外部测得的模拟进程总时间 |
| `visualizer_process_wall_sec` | `*.visualizer.process.time` | Linux 外部测得的可视化进程总时间 |

不能用 shell 的 `time` 代替 CUDA Event 时间：外部时间包含进程启动、内存分配、CPU 诊断和文件 I/O，而 `gpu_kernel_compute_sec` 用于报告 GPU 模拟计算区间。

### 9. 常见问题

- 报错 `PyTorch cannot access a CUDA GPU`：检查 PyTorch 是否为 CUDA 版本以及 `torch.cuda.is_available()`；
- 报错 `ffmpeg not found`：安装 FFmpeg 或将其加入 `PATH`；
- A800 显示 `video_encoder=libx264`：这是正常结果，A800 没有 NVENC，但粒子渲染仍由 CUDA 加速；
- 4090 未使用 `h264_nvenc`：检查 `ffmpeg -encoders | grep h264_nvenc`，没有该项时脚本自动回退 `libx264`；
- 65536 视频生成仍较慢：可临时设置 `--frame-stride 2` 输出约 501 帧进行预览，正式验收再使用 `--frame-stride 1`；
- GPU 显存不足：先用 `nvidia-smi` 清理无关进程；65536 粒子轨迹上传 GPU 约需 0.73 GiB，另需帧缓冲和 PyTorch运行空间。

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
