# 毕业论文可用文献清单（稀疏 / 量化 / 剪枝 / 硬件加速）

> 检索日期：2026-10-19。按"与 Supra2-IMG FPGA 项目的结合点"分类，每篇注明用途。
> 标 ★ 的为建议精读的核心文献（约 10 篇），其余按章节需要引用。

---

## 一、量化（与本项目 Hadamard+INT8 recipe 最直接相关）

| 文献 | 出处 | 用途 / 结合点 |
| --- | --- | --- |
| ★ QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs | arXiv:2404.00456 | **本项目量化方案的理论源头**：Hadamard 旋转消除激活 outlier、旋转融入权重（W←W·H）、per-token 在线量化。第 3 章量化方案必须引用，用来论证"块对角 Sylvester Hadamard + W@H"的合理性 |
| SpinQuant: LLM Quantization with Learned Rotations | NeurIPS 2024 | QuaRot 的后续：可学习旋转矩阵。用于相关工作里说明旋转量化这条线的演进 |
| ★ Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers | CVPR 2025 (arXiv:2406.17343) | **DiT 量化最直接的对比工作**：细粒度 group 量化 + 动态激活量化 + 进化搜索粒度分配。本文"动态 per-token 量化"与 Q-DiT 的动态激活思想呼应；第 2/3 章对比表用它 |
| ★ PTQ4DiT: Post-training Quantization for Diffusion Transformers | NeurIPS 2024 | DiT 通道均衡（channel equalization）PTQ。与 Q-DiT 并列作为 DiT 量化基线 |
| ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation | ICLR 2025 | 静-动混合量化策略，图像+视频 DiT。相关工作引用，说明"静态权重+动态激活"组合是社区共识 |
| DiTAS: Quantizing Diffusion Transformers via Enhanced Activation Smoothing | WACV 2025 | 时间步聚合平滑（TAS）的无数据 PTQ；切入点是 DiT 激活随 timestep 变化——与本项目"跨 50 步激活分布漂移"现象直接相关 |
| TaQ-DiT: Time-aware Quantization for Diffusion Transformers | arXiv 2024 | 时间步感知量化。同上，支撑"动态量化应对 timestep 变化"的论述 |
| HQ-DiT: Efficient Diffusion Transformer with FP4 Hybrid Quantization | arXiv 2025 | FP4 混合精度 DiT。用于"更低位宽"方向的展望 |
| An Analysis on Quantizing Diffusion Transformers | CVPRW | DiT 量化行为分析，写问题动机（activation/weight 方差大导致量化退化）时引用 |
| 背景：SmoothQuant / GPTQ / LSQ / Deep Compression / PTQD / EfficientDM | ICML 2023 / arXiv 2022 / ICLR 2024 等 | 第 1、2 章量化背景综述引用；PTQD、EfficientDM 是 UNet 扩散 PTQ 代表，用于说明"DiT 结构与 UNet 不同、需专门 PTQ"的动机 |

## 二、剪枝 / 稀疏（论文"进一步工作"或第 4 章优化技术）

| 文献 | 出处 | 用途 / 结合点 |
| --- | --- | --- |
| ★ ToMe: Token Merging, Your ViT But Faster / ToMeSD | CVPRW 2023 | token merging 开山之作及扩散版。与本项目 256 token 序列的稀疏化最相关；也是"token 级稀疏"综述入口 |
| ★ SparseDM: Toward Sparse Efficient Diffusion Models | ICME 2025 | 扩散模型稀疏化代表工作 |
| SparseDiT: Token Sparsification for Efficient Diffusion Transformer | NeurIPS 2025 | DiT 专门 token 稀疏化，最新对比对象 |
| Diff-Pruning: Structural Pruning for Diffusion Models | NeurIPS 2023 | 扩散模型结构化剪枝（同门 Delta-Pruning 系） |
| TinyFusion: Diffusion Transformers Learned Shallow | arXiv 2024 | 层剪枝/浅层化 DiT，与"块跳过"思想相关 |
| U-DiTs: Downsample Tokens in U-Shaped Diffusion Transformers | NeurIPS 2024 | U 形 token 下采样 DiT |
| FlexDiT: Dynamic Token Density Control for Diffusion Transformer | arXiv 2024 | 动态 token 密度控制——与本项目"按 timestep/块动态决定计算量"（50×14 skip trace，跳过率 40.857%）思想一致，**强烈建议引用** |
| δ-DiT: A Training-Free Acceleration Method Tailored for Diffusion Transformers | arXiv:2406.01125 | 免训练 DiT 加速；作为 timestep 冗余利用的代表 |
| DeepCache (CVPR 2024) / TeaCache / DiCache (arXiv:2508.17356) | CVPR 2024 / arXiv 2024-2025 | 特征缓存系：利用相邻 timestep 特征相似性跳计算。与本项目 skip trace + 40.857% 跳过率直接对应，写"动态调度"章节引用 |

## 三、硬件 / FPGA 加速器（第 4 章架构与第 6 章对比）

| 文献 | 出处 | 用途 / 结合点 |
| --- | --- | --- |
| ★ SDA: Low-Bit Stable Diffusion Acceleration on Edge FPGAs | FPL 2024 | **与本项目最接近的硬件对标**：首个边缘 FPGA 低比特 Stable Diffusion 加速器，同时支持 Transformer block 和 ResNet block。第 1/6 章必须对比 |
| ★ FlightLLM: Efficient LLM Inference with a Complete Mapping Flow on FPGAs | FPGA 2024 (arXiv:2401.03868) | FPGA 上大模型完整映射流程；其"可配置稀疏 DSP 链支持 N:M 稀疏"是稀疏量化硬件化的标杆思路 |
| ShatterQuant: Block-Wise Mixed-Precision on a Systolic Transformer Hardware Accelerator | arXiv:2610.00207 | 32×32 脉动阵列 + 混合精度 Transformer 加速器，INT32→INT8 输出量化路径与本项目 epilogue 设计高度相似，**架构对比首选** |
| DFX: A Low-Latency Multi-FPGA Appliance for Transformer-based Text Generation | MICRO 2022 | 多 FPGA transformer 加速，相关工作引用 |
| FACT / CTA（FPGA LLM 加速器） | FPGA 2023/2024 | 稀疏注意力硬件化对比（FlightLLM 文中系统对比过） |
| TPU: An In-Order Datacenter Architecture (Jouppi et al.) | ISCA 2017 | 脉动阵列经典引用；Kung & Leiserson 1979 为 systolic array 源头 |

---

## 四、建议的"故事线"用法

1. **第 2 章（问题动机）**：Q-DiT / An Analysis on Quantizing DiT 的"DiT 权重与激活方差大 → 量化退化"问题 → 引出本项目 Hadamard 旋转方案。
2. **第 3 章（量化方案）**：以 QuaRot 为理论根基（旋转消除 outlier、融入权重、per-token 动态量化），说明本项目在 DiT + 256 token + INT8 阵列上的实例化与精度验证（rel-peak 0.37%–0.46%）。
3. **第 4 章（稀疏/调度优化）**：以 FlexDiT / DeepCache / δ-DiT 的 timestep 自适应计算为理论支撑，把项目已有的 50×14 skip trace（40.857% 跳过率）和 masked cross-attention KV 打包（QK −75%、PV −52.9%）写成"结构化稀疏的硬件利用"；展望部分接 ToMe / SparseDiT 的 token 级稀疏。
4. **第 6 章（对比实验）**：SDA（边缘 FPGA 扩散加速）、FlightLLM（FPGA 量化+稀疏映射）、ShatterQuant（脉动阵列混合精度）作为三个维度对比。

## 五、注意

- 以上条目均来自 2026-10-19 的公开网络检索；引用前请以 arXiv/出版社页面核对卷期页码。
- scholar 插件因本机未配置 API key 未能使用；如需补检索（如 2:4 结构化稀疏、FP8 量化在扩散模型上的应用），可配置 `~/.kimi/agent-gw.json` 后再查。
