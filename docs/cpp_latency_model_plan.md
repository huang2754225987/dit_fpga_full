# Supra2 小 NPU C++ 延时建模计划

## 1. 目标与边界

实现一个离散事件、周期级的 C++ 模型，用于估算 Supra2 小 NPU 单个 block-13 事务的延时，并解释计算资源、DMA 和 DDR 队列分别贡献了多少周期。模型必须以本项目的任务图、RTL 事务行为和 AXI512 + MIG 仿真计数为依据。

首版范围是**单个 block-13、单个 AXI512 MIG 配置**。CFG、skip trace、多 block 和 50 步流程作为后续扩展接口，不作为首版验收项。输出属于经 RTL/MIG 行为仿真校准的周期模型结果，不是 FPGA 板上实测，也不是完整推理结果。

## 2. 代码组织与接口建议

建议将可执行模型放在 `simulator/cpp_latency/`，避免把模型实现混入 RTL 或现有 Python 编译器。最小文件划分如下：

| 文件 | 职责 |
|---|---|
| `CMakeLists.txt` | 构建 C++17 命令行程序和测试目标。 |
| `include/model_config.h` | 时钟、AXI 位宽、固定首响应延迟、传输速率、队列深度和校准数据结构。 |
| `include/task_graph.h` | Task、依赖、资源需求、读写字节数和阶段标签。 |
| `include/event_simulator.h` | 事件队列、依赖完成时间、资源占用和模拟统计接口。 |
| `include/axi_ddr_model.h` | burst 拆分、读写服务时间、队列反压和内存统计。 |
| `src/main.cpp` | 解析参数、载入任务/配置、执行模拟、输出 JSON 与人类可读摘要。 |
| `tests/latency_model_tests.cpp` | 固定任务图、单 burst、队列阻塞和校准数据检查。 |

建议的核心类型：

```cpp
using Cycle = std::uint64_t;

struct Task {
    std::string id;
    std::string stage;
    std::vector<std::string> dependencies;
    std::vector<std::string> resources;
    Cycle duration_cycles;
    std::uint64_t bytes_read;
    std::uint64_t bytes_written;
};

struct MemoryRequest {
    std::uint64_t address;
    std::uint64_t bytes;
    bool is_write;
    std::string owner_task;
};
```

CLI 至少支持 `--config <path>`、`--workload <path>`、`--output <path>` 和 `--help`。配置必须能注明每个延时/带宽参数是来自 MIG 仿真、RTL 计数、明确假设还是命令行覆盖。工作负载先采用版本化 JSON；任务字段包括 ID、stage、依赖、资源、周期时长以及 DDR 读写字节。模型遇到未知资源、缺失依赖、重复 ID、依赖环、非法配置或不支持的工作负载版本时应报错并返回非零状态，不静默忽略。

## 3. 周期推进、资源调度与 DDR 行为

### 3.1 事件驱动任务调度

使用按 `event_cycle` 排序的最小优先队列推进时间，不逐周期空转。事件至少包含任务可启动、计算完成、DDR 请求可服务和 DDR 响应返回。任务启动时间不得早于所有依赖完成时间，也不得早于其所有资源可用时间；多资源任务只有在所有请求资源都可用时才可启动。每次完成事件释放资源并唤醒后继任务。相同周期事件采用稳定顺序（按事件类别、再按任务 ID），确保重复运行结果确定。

任务依赖与资源定义应从现有 tile schedule 及其任务/阶段数据转换，不在 C++ 中重新手写一套不同的工作负载。当前 schedule replay 的重叠调度值是排程估计，不能直接当作已由数据通路执行的 RTL 延时；校准基准采用通过的连接式 MIG + DMA 单 block 计数。

### 3.2 AXI burst 与简化 DDR 模型

首版内存模型包括有限 outstanding 队列、AXI 数据位宽、burst beat 数、固定首响应延迟、beat 间隔/持续传输率，以及响应队列反压。将 byte 数转换为 beat 数时向上取整；跨越配置的最大 burst 长度或 4 KiB 边界时拆分请求。读写通道是否独立、是否共享服务资源需由配置显式指定；默认按当前 DMA 的串行/阶段式访问约束建模，不擅自假设读写并发。排队时间、服务时间、反压时间分别累计，避免把首延迟和每 beat 时间重复计入。

首版**不模拟 DDR bank/row、FR-FCFS、刷新或 GPU cache**。AXI512、300.120048 MHz UI 时钟和 DDR 行为由板级 MIG 配置锚定；首响应延迟和每 beat 服务间隔应通过现有 MIG 波形/计数及 burst microbenchmark 校准，HANDOFF 中的总事务和三段周期只能校验整段结果，不能单独推出这些参数。尚未量出的参数须标成“待校准假设”，并允许敏感性扫描，不能伪装成实测配置。

### 3.3 可复用逻辑与许可证

- 参考 `gpgpu-sim_distribution-dev/src/gpgpu-sim/l2cache.cc` 的 `memory_partition_unit::simple_dram_model_cycle()`：定时请求队列、满足 ready cycle 后返回请求，以及返回队列满时施加反压的控制流程。该成员函数依赖 GPU 的 memory partition、`mem_fetch` 和统计对象，不能原样单独编译；只抽取其通用控制逻辑到本项目独立类型。
- 参考 `gpgpu-sim_distribution-dev/src/gpgpu-sim/delayqueue.h` 的 `fifo_pipeline<T>` 延迟 FIFO 语义。不要直接 include 整个头文件，因为它依赖 GPGPU-Sim 的 `statwrapper.h`、`gpu-misc.h` 和相关类型。仅在确认实际需要固定拍数 FIFO 后复制最小实现；本模型按事件 cycle 队列即可表达延时的部分，不必为“复用”而保留不需要的模板代码。
- 参考本项目 `scripts/simulate_block_pipeline.py` 的依赖/资源 list scheduling 逻辑，以及 `compiler/supra2/compile_block13_tile_program.py` 的任务、资源和 measured-cycle 数据组织。将有用算法迁移至 C++ 并定义稳定的工作负载输入格式，不复制 Python 运行环境或输出目录逻辑。
- 若复制 GPGPU-Sim 源码片段，保留对应文件版权头和 BSD 许可证文本，在新文件注释中注明源路径及本项目修改；复制的二进制分发也须附许可证要求的声明。不得复制 GPGPU-Sim 的 GPU DRAM 时序配置作为 NPU 参数。

## 4. 统计输出与校准

程序应输出 JSON 结果及简短终端摘要，至少包含：

- 总事务周期和按配置时钟换算的时间；
- 输入 DMA、计算 block、输出 DMA 的周期；
- 按 stage/resource 统计的 busy cycles、利用率和排队等待；
- DDR 读写字节、burst 数、beat 数、有效带宽及请求等待/服务/反压周期；
- 参数来源、工作负载版本、配置摘要和模型版本。

已有连接式 AXI512 + 实际板级 MIG IP + 四颗 x16 DDR4 器件模型的 block-13 仿真是首版校准锚点：

| 项目 | RTL/MIG 参考值 |
|---|---:|
| 输入 DMA | 178,086 cycles |
| Block 执行 | 4,442,603 cycles |
| 输出 DMA | 43,145 cycles |
| 总事务 | 4,663,834 cycles |
| MIG UI 周期 | 3,332 ps（约 300.120048 MHz） |

权重等待 385,969 cycles 已包含在 block 执行周期中，不能再加到总周期。总周期满足三阶段之和。参考仿真读取 5,709,312 bytes、写回 294,912 bytes；其 host preload 和 host verification read 不属于上述事务周期。

校准报告需逐项给出 `model_cycles`、`reference_cycles`、绝对误差和相对误差，并明确区分：模型估计、RTL/MIG 行为仿真数据、假设参数。未经独立计数支持，不得声称某个请求级 DDR 首延迟或 beat 周期已被测得。有效带宽的分子/分母必须注明采用的字节数和时间区间，不能把有效 MB/s 说成 MIG 瞬时带宽。

## 5. 测试与验收标准

使用轻量 C++ 单元测试/CTest，不运行 FPGA 仿真、综合或权重生成。至少覆盖：

1. **依赖顺序：** A→B→C 逐级启动，检查开始时间不早于依赖完成。
2. **独立任务并行：** 无依赖且资源不同的任务可重叠，makespan 等于较慢分支完成时间。
3. **资源互斥：** 使用同一 array 的两个任务不得重叠；多资源任务需等到全部资源可用。
4. **AXI burst：** 以小型已知参数检查 ceil beat 计算、首响应延迟和 beat 间隔；覆盖 4 KiB/最大 burst 拆分边界。
5. **队列反压：** 队列满或响应端阻塞时请求不得丢失，等待时间计数正确。
6. **输入错误：** 环依赖、缺失依赖、重复 ID、无效位宽/零速率和未知版本均给出可读错误并失败退出。
7. **确定性：** 同一输入重复运行输出相同的任务启动/完成周期与 makespan。
8. **端到端校准：** 三个阶段及总事务与上述 MIG 参考计数逐项比较；未达到项目约定误差门限时报告失败和差异，不通过调参隐藏误差。若参数尚不足以复现参考值，先报告未校准状态，不虚构 PASS。

## 6. 实施顺序

1. 从当前编译器/仿真记录导出带版本号的 block-13 工作负载和配置样例，人工核对任务计数、依赖、周期单位及 DDR 字节数。
2. 实现并测试事件队列、资源占用和 DAG 校验；先用合成小任务图验证调度语义。
3. 实现 AXI burst 拆分、有限队列、延时响应和反压；用小型确定性用例覆盖边界。
4. 接入 block-13 工作负载，先核对三阶段周期，再引入可校准 DDR 参数和详细统计。
5. 生成校准报告，记录所有残余差异与未测参数。通过后再扩展 CFG、skip trace、多 block 和完整 50 步流程。

## 7. 默认假设

- 文档与后续面向用户的解释使用中文；代码、类型名、命令行选项和 JSON 字段使用英文。
- C++17，CMake 构建；测试只覆盖软件模型本身。
- 默认硬件基准为 512-bit AXI、MIG UI 300.120048 MHz；不同配置必须显式切换配置文件并重新校准。
- 首版是周期模型，不追求 DDR 物理时序精确模拟；未知参数显式标为假设。
- 本文档用于后续实现 C++ 模型；创建本文档本身不代表模型代码已实现或验证。
