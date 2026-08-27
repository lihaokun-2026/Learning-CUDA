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

| 案例 | 粒子数 | 用途 |
|---|---:|---|
| `two_body` | 2 | 1000 步圆轨道和能量稳定性验证 |
| `solar_system` | 5 | 1000 步中心大质量天体和多轨道扰动 |
| `cluster_64` | 64 | 1000 步三维星团演化、聚团/散射 |
| `disk_1024` | 1025 | 1000 步粒子盘和中等规模性能测试 |
| `benchmark_4096` | 4096 | 基础版验收：1000 步、1001 帧 |
| `benchmark_65536` | 65536 | 进阶版验收：1000 步、1001 帧 |

例如运行 1024 粒子盘：

```powershell
.\nbody.exe data\disk_1024_particles.txt data\disk_1024_params.txt disk_1024.bin
python visualize.py disk_1024.bin --output disk_1024.gif
```

## 输入格式

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

粒子不超过约 4000 时能量使用完整 $O(N^2)$ 计算；更大规模使用固定 800 万对粒子的确定性采样估计，并在日志中标记 `energy_diagnostic=sampled`。

## 编译与运行（有 NVIDIA CUDA 环境的机器）

在本目录执行：

```powershell
nvcc -O3 -std=c++17 -lineinfo nbody.cu -o nbody.exe
.\nbody.exe data\particles.txt data\params.txt trajectory.bin
python -m pip install numpy matplotlib
python visualize.py trajectory.bin --output trajectory.gif
```

生成 MP4 需要系统安装 FFmpeg：`python visualize.py trajectory.bin --output trajectory.mp4`。

## 正确性测试

Linux 编译和验收命令：

```bash
# 编译nbody
nvcc -std=c++17 -lineinfo -arch=sm_89 nbody.cu -o nbody
# 测试 两个粒子
./nbody data/two_body_particles.txt data/two_body_params.txt two_body.bin two_body.log
# 测试 4096个粒子
./nbody data/benchmark_4096_particles.txt data/benchmark_4096_params.txt benchmark_4096.bin benchmark_4096.log
# 测试 65536个粒子
./nbody data/benchmark_65536_particles.txt data/benchmark_65536_params.txt benchmark_65536.bin benchmark_65536.log
```

RTX 4090 对应 `sm_89`。如果可执行文件需要在其他 GPU 上运行，可使用 `-gencode` 同时嵌入多个架构，或省略 `-arch=sm_89` 使用 nvcc 默认目标。

### 正确性验收

双体系统运行后检查动画是否保持近似圆轨道，同时检查 `two_body.log` 的 `relative_energy_error` 和 `relative_momentum_error`。Leapfrog 的误差应保持有界；具体阈值取决于 `dt` 和软化参数，建议双体案例两项误差均不超过 $10^{-3}$。多体系统重点检查动量误差；采样能量适合观察趋势，不应作为严格精确阈值。

```bash
python3 visualize.py two_body.bin --dimension 3d --trail 20 --output two_body.gif
cat two_body.log
cat benchmark_4096.log
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
