# VLM 构建场景图 / 拓扑图：中文研究综述

> 整理时间：2025–2026  
> 覆盖范围：基本思路、图表示、构图流程、VLM/LLM 用法、与 VLN/Embodied AI 结合、代表性工作时间线、局限与未来方向

---

## 目录

1. [基本思路与系统框架](#1-基本思路与系统框架)
2. [常见图表示](#2-常见图表示)
3. [构图流程](#3-构图流程)
4. [VLM/LLM 在构图中的用法](#4-vlmllm-在构图中的用法)
5. [与 VLN / Embodied AI 的结合](#5-与-vln--embodied-ai-的结合)
6. [代表性工作与时间线（2020–2026）](#6-代表性工作与时间线20202026)
7. [局限与未来方向](#7-局限与未来方向)

---

## 1. 基本思路与系统框架

### 1.1 为什么需要场景图 / 拓扑图

经典导航与场景理解系统通常依赖固定词表的语义地图（occupancy grid + 类别标签），存在以下局限：

- **封闭词表**：无法识别训练集之外的对象或属性；
- **缺乏关系信息**：知道"有椅子"，但不知道"椅子在桌子旁边"；
- **难以语言对齐**：导航指令里的"红色沙发左边的门"无法直接在地图里检索；
- **长程规划困难**：栅格地图缺乏高层拓扑结构，路径规划开销大。

**场景图（Scene Graph）** 和 **拓扑图（Topological Map）** 提供了更紧凑、可推理、可语言查询的结构化环境表示，是近年 VLN（视觉-语言导航）和 Embodied AI 的核心研究方向之一。

### 1.2 整体系统框架

```
原始传感器输入
  RGB / RGB-D / 单目  ──►  感知模块  ──►  对象/区域检测 & 分割
  IMU / 里程计 / SLAM  ──►  位姿估计

         │                          │
         ▼                          ▼
  VLM 语义特征提取           3D 定位 & 数据关联
  (CLIP / BLIP / GPT-4V)
         │                          │
         └──────────────────────────┘
                      │
                      ▼
              图节点 & 边构建
           (对象节点 / 地点节点)
           (空间关系边 / 可达边)
                      │
                      ▼
              图融合 & 增量更新
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       语言查询检索       规划 & 决策
   (文本 → 图检索)    (GNN / LLM / A*)
```

VLM 在整个管线中的核心作用是：**将图像/视频转换为可与自然语言对齐的语义节点属性和关系标签**，从而让场景图真正"可被语言查询"。

---

## 2. 常见图表示

### 2.1 节点类型

| 节点类型 | 含义 | 典型属性 |
|---------|------|---------|
| **对象节点（Object Node）** | 场景中可识别的物理实体 | 类别、颜色、尺寸、可供性（affordance）、CLIP embedding、文本描述 |
| **地点节点（Place/Viewpoint Node）** | 机器人曾到达的历史位置（viewpoint） | 全景图特征、地理坐标、连通性、时间戳 |
| **房间/区域节点（Region Node）** | 语义区域（厨房、卧室、走廊） | 区域类别、包含的对象集合、面积 |
| **概念节点（Concept Node）** | 知识图谱中的抽象概念（可选） | 来自 ConceptNet/WordNet，提供常识推理能力 |

### 2.2 边类型

| 边类型 | 含义 | 示例 |
|-------|------|------|
| **可达边（Nav Edge）** | 两个地点节点之间可直接导航 | viewpoint_A → viewpoint_B |
| **空间关系边（Spatial Edge）** | 对象间的相对位置关系 | chair **left-of** table |
| **包含边（Contains Edge）** | 区域包含对象 | kitchen **contains** microwave |
| **功能关系边（Functional Edge）** | 对象间的功能/交互关系 | knife **used-with** cutting-board |
| **时序边（Temporal Edge）** | 跨帧/跨时刻的同一对象 | obj_t1 **same-as** obj_t2 |

### 2.3 多模态属性

现代场景图节点通常同时携带多种模态的属性：

- **视觉 embedding**：CLIP image feature、ViT patch feature、DINO feature
- **语言描述**：由 VLM（BLIP-2、LLaVA、GPT-4V）生成的对象 caption
- **3D 几何**：中心坐标（x, y, z）、边界框（bounding box）、法向量
- **语义标签**：开放词表类别（CLIP zero-shot classification）
- **可供性**：可坐、可开/关、可抓取等（由 VLM 推理）

---

## 3. 构图流程

### 3.1 感知阶段

**输入**：RGB（或 RGB-D）视频流 + 相机位姿（来自 SLAM 或外部定位）

**目标**：从图像中提取可信的对象/区域语义特征

- **检测**：DETR、DINO-DETR、GroundingDINO（开放词表）
- **分割**：SAM（Segment Anything）、Mask2Former
- **深度估计**：MiDaS、DPT（单目深度）；ZED/RealSense 直接输出
- **VLM 特征提取**：CLIP（整图或 mask crop）、BLIP-2（生成 caption）

```
RGB 帧  ──► GroundingDINO ──► 对象 Bounding Box + 类别文本
        ──► SAM ──► 精细 Mask
        ──► CLIP ──► 每个 Mask 的视觉 embedding
        ──► BLIP-2 ──► 整图/对象级 caption
```

### 3.2 分割 / 跟踪

**目的**：跨帧保持对象身份，避免同一对象在不同帧中被重复创建为多个节点。

- **多目标跟踪（MOT）**：ByteTrack、OC-SORT 等；适用于动态场景
- **特征匹配**：CLIP embedding 相似度 + IOU 匹配；适用于静态环境
- **3D 位置约束**：若两帧对象 3D 中心距离 < 阈值，认为同一对象

### 3.3 跨帧数据关联（Data Association）

核心挑战：**不同视角/不同时刻观测到的同一对象，如何合并到同一节点？**

常见策略：
1. **2D IoU + 类别一致性**：简单但受视角变化影响大
2. **CLIP 特征相似度（cosine similarity > threshold）**：对类别泛化更好
3. **3D 中心点距离**：需要深度信息，精度依赖深度质量
4. **图神经网络（GNN）学习关联**：端到端学习数据关联，泛化更强（如 ConQueR）

### 3.4 3D 定位

将 2D 检测/Mask 反投影到 3D 空间，为节点赋予 3D 坐标：

```
像素坐标 (u, v, d)
    ──► 相机坐标系: X_c = (u - cx) * d / fx
    ──► 世界坐标系: X_w = R^{-1} * (X_c - t)
```

- **点云中心**：取 Mask 内深度点云的中位数中心
- **TSDF / 体素融合**：KinectFusion / Open3D 类方法，几何精度更高
- **NeRF / 3DGS 语义融合**：LeRF、LangSplat 等，可查询但实时性差

### 3.5 图融合与更新

**增量融合策略**：

| 策略 | 描述 | 适用场景 |
|-----|------|---------|
| **运行均值（Running Mean）** | 新观测特征与已有特征做加权平均 | 静态场景，对象固定 |
| **置信度加权** | 按观测次数或置信度加权融合 | 噪声较大的传感器 |
| **最新优先（Latest）** | 总是用最新观测覆盖 | 动态场景 |
| **最大响应** | 保留每维度最高激活值 | CLIP 特征融合实验表明此策略有时更优 |

**拓扑图更新**：
- 每到达新位置，检查与已有地点节点的距离；若 > 阈值则新建节点，否则合并
- 检测到可穿越区域（门/走廊）时，在相邻地点节点之间新建导航边
- 闭环检测（Loop Closure）：识别到曾访问的地点时，修正累积漂移并合并重复节点

---

## 4. VLM/LLM 在构图中的用法

### 4.1 开放词表对象识别

传统检测器依赖固定类别集（COCO 80类 / LVIS 1200类），无法识别导航指令里的任意名词。

**VLM 解决方案**：

- **CLIP zero-shot 分类**：将文本查询与图像 patch embedding 做相似度排序
- **GroundingDINO**：开放集目标检测，直接输入文本 prompt，输出对应检测框
- **OWLv2（Open-Vocabulary Detection）**：Google 提出的开放词表检测器，效果更优
- **LLaVA / InternVL / GPT-4V**：将对象描述问题化（"这张图里有哪些物体？"），生成结构化回答

### 4.2 关系抽取

**目标**：从图像或对象集合中自动推断空间/功能关系，填充场景图的边。

- **基于 VLM Prompt**：
  ```
  Prompt: "描述图中 [对象A] 和 [对象B] 之间的空间关系，从以下选择：
  left-of / right-of / in-front-of / behind / above / below / inside / on-top-of"
  ```
- **Scene Graph Generation（SGG）模型**：Motifs、VCTree、BGNN 等专门的关系检测模型
- **LLM 常识推理**：对于功能关系（"刀通常与切菜板一起使用"），用 LLM 推理比视觉更可靠

### 4.3 Caption / Grounding

**Caption（图像描述）**：

用 BLIP-2、LLaVA、CogVLM 等模型为每个地点节点或对象节点生成自然语言描述：

```python
# 伪代码：为地点节点生成 caption
caption = vlm.generate(
    image=viewpoint_image,
    prompt="简短描述这个位置，包括主要物体和空间布局。"
)
node.caption = caption
node.text_embedding = text_encoder(caption)
```

这些 caption 可以被检索、被 LLM 推理，也可以直接作为导航历史记忆。

**Grounding（语言定位）**：

给定语言描述（如"蓝色沙发旁的电视"），定位到场景图中具体节点：

- **文本→图节点检索**：将 query 编码为 text embedding，与所有节点的 text embedding 计算相似度
- **MLLM 推理**：将场景图文本化，交给 LLM 推理哪个节点最符合描述

### 4.4 图查询（Graph Query）

场景图构建完成后，VLM/LLM 可通过不同方式回答空间查询：

| 查询类型 | 示例 | 方法 |
|--------|------|------|
| **对象定位** | "哪里有杯子？" | CLIP 相似度检索节点 |
| **关系查询** | "椅子在桌子哪边？" | 遍历对应边标签 |
| **路径查询** | "从厨房到卧室怎么走？" | 图上 Dijkstra/BFS + 地点节点标签 |
| **多跳推理** | "装牛奶的东西在哪里？" | LLM 推理（牛奶→冰箱→位置） |
| **目标规划** | "找到可以坐的地方" | 可供性节点过滤 + 路径规划 |

---

## 5. 与 VLN / Embodied AI 的结合

### 5.1 VLN 中的拓扑图建图

在主流 VLN 基准（R2R、REVERIE、SOON）中，代理在离散 Matterport3D 环境里导航，常见地图形式：

```
历史轨迹拓扑图：
  - 节点：已访问的 viewpoint（全景图像 + CLIP/ViT 特征）
  - 边：相邻可达关系
  - VLM 增强：为每个节点生成 caption / landmark 描述
  - 决策：Transformer 融合指令 + 图上历史节点特征 → 预测下一步
```

代表性工作：**DUET**（图+局部双粒度 Transformer）、**HAMT**（层次注意力记忆）、**MapGPT**（LLM 驱动图规划）

### 5.2 连续动作空间（VLN-CE）

更接近真实机器人，使用 Habitat 模拟器，动作连续：

- **BEV 语义地图 + 拓扑节点**：将场景语义投影到俯视图，同时维护可通行性
- **VLM 指路点生成（Waypoint）**：CWP、GWP 等用 VLM 预测下一个中间目标点
- 深度 + 语义分割 + 位姿融合 → 增量 BEV 地图 → 局部/全局规划

### 5.3 ObjectNav / Embodied Question Answering（EQA）

**ObjectNav**：用语言描述找目标物体

- **VLMaps**：CLIP 特征融合到 3D 栅格 → 文本 query 热力图 → 目标点规划
- **CoCoNut / MOPA**：对象级地图 + LLM 常识规划（"找微波炉→先找厨房"）
- **OpenFMNav**：GPT-4 推理 + 开放词表检测 + 前沿探索策略

### 5.4 基于场景图的规划与决策

**图上推理的优势**：
- 多跳查询：A 与 B 相邻，B 在 C 上方 → A 与 C 的间接关系
- 符号可解释性：决策依据可追溯（相比纯神经网络黑盒）
- 外部知识融合：将 ConceptNet 类常识图谱与场景图对接

**典型决策流程（LLM + 场景图）**：
```
1. 接收语言指令："把红色苹果放到冰箱里"
2. 场景图查询：找到"红色苹果"节点 → 坐标 p1
3. 场景图查询：找到"冰箱"节点 → 坐标 p2
4. LLM 规划：["导航到 p1", "抓取苹果", "导航到 p2", "开冰箱", "放入苹果"]
5. 执行每个子步骤，更新场景图
```

**代表性工作**：SayPlan（LLM + 场景图规划）、3D-LLM（3D场景图作为 LLM 输入）、EmbodiedScan

---

## 6. 代表性工作与时间线（2020–2026）

### 6.1 Scene Graph Generation（2D/视频 SGG）

| 年份 | 工作 | 核心贡献 |
|-----|------|---------|
| 2020 | **BGNN**（Bi-directional Graph Neural Network）| 双向 GNN 建模场景图关系，SOTA on VG |
| 2021 | **HOTR** | Transformer 端到端人-物交互检测 |
| 2021 | **RelTR** | 完全 Transformer 化的关系检测（无 proposal） |
| 2022 | **SGTR** | 场景图生成端到端 Transformer |
| 2023 | **FACTUAL** | 用于 VQA 和 caption 的事实场景图 |
| 2024 | **OpenPSG** | 开放词表全景场景图生成（OWL-ViT） |
| 2024 | **VL-SAT** | VLM 语义辅助的场景图生成 |

### 6.2 3D Scene Graph

| 年份 | 工作 | 核心贡献 |
|-----|------|---------|
| 2020 | **3DSSG**（3D Scene Graph Dataset） | 首个大规模 3D 场景图数据集与基准（RGB-D） |
| 2021 | **SceneGraphFusion** | 增量式 RGB-D 流 3D 场景图构建 |
| 2022 | **Hydra** | 层次式动态场景图（对象/房间/建筑多层级，MIT 机器人） |
| 2022 | **ConceptGraphs** | CLIP + SAM + 3D 点云融合，开放词表 3D 对象图（CMU/MIT） |
| 2023 | **OpenScene** | 用 CLIP/LSeg 将语义融合进 3D 点云，开放词表 3D 场景理解 |
| 2023 | **OVSG**（Open-Vocabulary Scene Graph） | 结合 LLM 进行 3D 场景图查询推理 |
| 2024 | **SQA3D** | 3D 场景问答基准，场景图作为推理中间表示 |
| 2024 | **EmbodiedScan** | 统一 3D 感知（检测/分割/问答）+ 场景图，大规模数据集 |
| 2025 | **SceneVerse** | 百万级 3D 场景图-语言对数据集，VLM 预训练 |

### 6.3 Embodied / Robot Mapping（基于 VLM 的建图）

| 年份 | 工作 | 核心贡献 |
|-----|------|---------|
| 2022 | **VLMaps** | CLIP 特征写入 3D 栅格，文本 query 定位目标（Google DeepMind） |
| 2023 | **CoCoNut** | 对象级语义图 + LLM 常识推理，机器人开放目标导航 |
| 2023 | **ConceptGraphs** | 开放词表 3D 对象级场景图，SAM + CLIP + LLM 关系推理 |
| 2023 | **OpenFMNav** | GPT-4 + 探索策略 + 开放词表检测，零样本 ObjectNav |
| 2023 | **SayPlan** | LLM + 层次式场景图做任务规划，可执行机器人指令 |
| 2024 | **GaussianMap** | 3D Gaussian Splatting + 语义标签，可查询密集重建地图 |
| 2024 | **LangSplat** | 在 3DGS 上融合 CLIP 语言特征，精细开放词表 3D 查询 |
| 2024 | **MOPA** | 模块化对象导航，VLM 感知 + LLM 规划 + 场景图记忆 |
| 2025 | **ActiveSG** | 主动建图（主动视角选择）+ 实时 3D 场景图更新 |

### 6.4 VLN 拓扑图（Navigation Topological Map）

| 年份 | 工作 | 核心贡献 |
|-----|------|---------|
| 2020 | **VLN-BERT** | BERT 融合视觉-语言，首次在 R2R 上大规模预训练 |
| 2021 | **HAMT** | 层次历史感知 Transformer，全景图+对象级多粒度历史记忆 |
| 2022 | **DUET** | 双粒度图 Transformer（全局拓扑图 + 局部视点），R2R SOTA |
| 2022 | **CWP**（Cross-modal Waypoint Prediction）| VLN-CE 中 VLM 预测路径点 |
| 2023 | **NavGPT** | 纯 LLM（GPT-4）零样本导航，自然语言思维链推理 |
| 2023 | **MapGPT** | LLM 维护文本化拓扑地图，在线规划导航 |
| 2023 | **OpenVLN** | 利用 LLM 数据增强 + 开放词表增强 VLN 泛化 |
| 2024 | **VLN-SIG** | 场景指令图（Scene Instruction Graph），跨指令多跳对齐 |
| 2024 | **GOAT** | Zero-shot 多目标导航（语言/图像/对象类别），拓扑+语义图 |
| 2025 | **VLN-R1** | 强化学习 + VLM 推理，无需导航专项训练的泛化导航 |


---

## 7. 局限与未来方向

### 7.1 实时性（Real-time Performance）

**现状问题**：
- 大型 VLM（GPT-4V、LLaVA-34B）推理延迟高（秒级），难以跟上机器人实时移动（10+ fps 需求）
- 3D 点云融合 + CLIP 特征更新在高分辨率下算力需求大

**研究方向**：
- 轻量化 VLM（MobileVLM、TinyLLaVA）用于边缘部署
- 异步双速率架构：快速几何建图 + 慢速语义更新
- 增量式图更新：只重新处理变化区域，而非全图
- 硬件加速：NPU/专用芯片 + 模型量化（INT4/INT8）

### 7.2 鲁棒性（Robustness）

**现状问题**：
- VLM 对光照变化、遮挡、模糊图像的识别鲁棒性仍不足
- 开放词表检测存在误检/漏检，错误节点会在图中持续传播
- 动态物体（人、宠物）会在静态地图中留下"幽灵"节点

**研究方向**：
- 基于置信度的节点老化/删除机制（Temporal Decay）
- 多传感器融合（LiDAR + 视觉）提升感知可靠性
- 主动感知：对不确定区域主动补充观测
- 对抗训练提升 VLM 鲁棒性（噪声、光照扰动）

### 7.3 闭环一致性（Loop Closure Consistency）

**现状问题**：
- SLAM 累积漂移导致同一对象被建图在不同位置
- 闭环检测后几何地图修正，但语义节点坐标未同步更新
- 大规模场景（多楼层、多房间）的全局一致性难以保持

**研究方向**：
- 场景图层面的闭环：识别到重复地点节点时做图合并（Graph Merge）
- 语义锚点（Semantic Anchor）：用稳定对象（固定家具）作为定位参考
- 基于位置识别的场景图匹配（图同构 / 子图匹配）
- 因子图优化（Factor Graph）将语义约束纳入位姿图优化

### 7.4 评测基准（Evaluation Benchmarks）

**现有基准的不足**：
- 多数 VLN/ObjectNav 基准（R2R、HM3D）不直接评估场景图质量
- 3DSSG、ScanSG 等场景图基准规模较小（数百场景），泛化评估受限
- 缺乏**在线增量建图**的评估协议（多数基准假设图已构建完毕）
- 缺乏**语言查询精度**的系统性评估（retrieval precision/recall on scene graphs）

**研究方向**：
- 面向开放词表查询的场景图检索基准（Query-SG Bench）
- 增量建图质量指标：节点精确率/召回率、关系 F1、随时间的一致性
- 端到端任务评估：场景图质量 → 导航成功率的因果链评测
- 大规模真实场景采集（对标 ScanNet++）+ 自动化 3D 场景图标注流水线

### 7.5 其他开放问题

| 问题 | 描述 |
|-----|------|
| **动态场景** | 人物移动、物体状态变化（门开/关）如何实时更新图 |
| **跨模态一致性** | 视觉观测与语言描述的语义对齐精度 |
| **可解释性与可信度** | LLM 推理的场景图查询结果是否可靠，如何量化不确定性 |
| **隐私与安全** | 携带详细对象属性的场景图可能泄露隐私（家庭布局等） |
| **长时记忆** | 数天/数周跨度的场景变化管理与地图版本控制 |

---

## 参考文献（按类别）

### Scene Graph Generation
- Xu et al., 2017. Scene Graph Generation by Iterative Message Passing. *CVPR 2017*
- Zellers et al., 2018. Neural Motifs: Scene Graph Parsing with Global Context. *CVPR 2018*
- Lin et al., 2020. BGNN: Bipartite Graph Neural Network. *CVPR 2020*
- Cong et al., 2023. FACTUAL. *ACL 2023*

### 3D Scene Graph
- Wald et al., 2020. 3DSSG: Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions. *CVPR 2020*
- Wu et al., 2021. SceneGraphFusion. *CVPR 2021*
- Hughes et al., 2022. Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization. *RSS 2022*
- Peng et al., 2023. OpenScene. *CVPR 2023*
- Gu et al., 2024. ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning. *ICRA 2024*

### Embodied / Robot Mapping
- Huang et al., 2022. Visual Language Maps for Robot Navigation. *ICRA 2023*（VLMaps）
- Shah et al., 2023. LM-Nav: Robotic Navigation with Large Pre-Trained Models. *CoRL 2022*
- Yokoyama et al., 2023. VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation. *ICRA 2024*
- Rana et al., 2023. SayPlan. *CoRL 2023*
- Kerr et al., 2023. LERF: Language Embedded Radiance Fields. *ICCV 2023*
- Qin et al., 2024. LangSplat: 3D Language Gaussian Splatting. *CVPR 2024*

### VLN Topological Map
- Majumdar et al., 2020. Improving Vision-and-Language Navigation with Image-Text Pairs from the Web. *ECCV 2020*（VLN-BERT）
- Chen et al., 2021. History Aware Multimodal Transformer for Vision-and-Language Navigation. *NeurIPS 2021*（HAMT）
- Chen et al., 2022. Think Global, Act Local: Dual-scale Graph Transformer for Vision-and-Language Navigation. *CVPR 2022*（DUET）
- Zhou et al., 2023. NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models. *AAAI 2024*
- Chen et al., 2023. MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation. *ACL 2024*
- Yokoyama et al., 2024. GOAT-Bench: A Benchmark for Multi-Modal Lifelong Navigation. *CVPR 2024*

---

*本文档持续更新，如有建议或补充欢迎提 issue 或 PR。*
