# N 体引力模拟器（CUDA）

## 文件

- `nbody.cu`：CUDA 实现。使用共享内存分块计算 $O(N^2)$ 引力，并支持 Euler 与 Leapfrog 积分。
- `visualize.py`：读取二进制轨迹并生成 Matplotlib 二维动画。
- `data/particles.txt`、`data/params.txt`：最小三体回归样例。
- `data/generate_cases.py`：生成多组确定性测试数据。

生成更多测试数据：

```powershell
python data\generate_cases.py
```

生成的案例包括：

| 案例 | 粒子数 | 用途 |
|---|---:|---|
| `two_body` | 2 | 圆轨道和能量稳定性验证 |
| `solar_system` | 5 | 中心大质量天体和多轨道扰动 |
| `cluster_64` | 64 | 三维星团演化、聚团/散射 |
| `disk_1024` | 1025 | 粒子盘和中等规模性能测试 |

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

`trajectory.bin` 为小端二进制：文件头是两个 `int32`（粒子数 `P`、记录数 `R`），之后按记录、粒子、坐标顺序写入 `float32` 的 `x,y,z`。程序同时输出耗时、粒子步/秒和相互作用/秒。

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

样例是近似等边三体系统。先将 `data\params.txt` 中的 `num_steps` 设为 `1000`，运行程序；检查输出中的 `records=11`，并用可视化脚本确认轨迹没有出现 NaN 或爆炸。然后将 `integrator` 分别设为 `euler` 和 `leapfrog`，比较两次结果；Leapfrog 通常具有更好的长期能量稳定性。

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

其中 $N$ 是粒子数，$S$ 是总步数，$R$ 是记录间隔。共享内存每个 block 约为 `256 * 16 = 4096` 字节。

仅按当前设备数组计算：

| 粒子数 | GPU 设备数组 |
|---:|---:|
| 4,096 | 0.19 MiB |
| 65,536 | 3.00 MiB |
| 1,000,000 | 45.78 MiB |
| 10,000,000 | 457.76 MiB |

因此 RTX 4090 的 24 GiB 显存完全可以运行题目要求的 4,096、65,536 粒子规模；甚至从“存储数组”角度可以容纳数亿粒子。但当前直接 N 体算法每一步需要 $N^2$ 次相互作用，真正瓶颈是计算时间而不是显存：$65,536^2 \approx 4.29$ 亿次相互作用/步，1000 步约 4.29 万亿次相互作用，可能需要较长时间。

此外，程序将所有轨迹保存在 CPU 内存中，若 $N=65,536$、记录 1001 次，轨迹约为 0.73 GiB；应增大 `record_interval`，避免主机内存和文件过大。RTX 4090 可以运行，但建议先用 `cluster_64`、`disk_1024` 验证，再逐步扩大规模。
