# InternNav 项目架构文档

## 项目概述

InternNav 是一个综合性的视觉-语言导航(VLN)框架,支持多种导航模型、多个模拟环境和真实机器人部署。项目提供了从训练、评估到实际部署的完整工具链。

---

## 目录结构

```
InternNav/
├── internnav/                          # 主Python包
│   ├── agent/                         # 导航代理实现
│   ├── model/                         # 模型架构实现
│   ├── dataset/                       # 数据集实现
│   ├── evaluator/                     # 评估器实现
│   ├── env/                           # 环境实现
│   ├── trainer/                       # 训练器实现
│   ├── configs/                       # 配置管理
│   ├── habitat_extensions/            # Habitat模拟器扩展
│   └── utils/                         # 通用工具
│
├── scripts/                           # 脚本和演示
│   ├── train/                         # 训练脚本
│   ├── eval/                          # 评估脚本
│   ├── realworld/                     # 真实世界部署
│   ├── demo/                          # 演示应用
│   ├── dataset_converters/            # 数据集转换工具
│   ├── iros_challenge/                # IROS挑战赛脚本
│   └── notebooks/                     # Jupyter演示笔记本
│
├── tests/                             # 测试
├── requirements/                      # 依赖管理
├── src/                              # 子模块
└── 配置文件                           # setup.py, pyproject.toml等
```

---

## 核心模块详解

### 1. Agent 模块 (`internnav/agent/`)

**功能**: 执行导航任务的智能代理

#### 代理类型

| 代理类型 | 文件 | 功能描述 |
|---------|------|---------|
| `Agent` | `base.py` | 抽象基类,定义代理接口 |
| `CmaAgent` | `cma_agent.py` | 跨模态联盟模型代理 |
| `RdpAgent` | `rdp_agent.py` | 扩散政策代理 |
| `Seq2SeqAgent` | `seq2seq_agent.py` | 序列到序列模型代理 |
| `InternVLAN1Agent` | `internvla_n1_agent.py` | 一体化导航基础模型 |
| `InternVLAN1AgentRealWorld` | `internvla_n1_agent_realworld.py` | 真实世界部署代理 |

#### 核心接口

```python
class Agent:
    def step(self, obs: Dict[str, Any]) -> action
    def reset(self) -> None

    @classmethod
    def init(cls, config: AgentCfg) -> Agent

    @classmethod
    def register(cls, agent_type: str) -> decorator
```

#### 工具类
- `agent_utils.py`: 张量字典处理、通用工具函数
- `utils/`: 代理辅助功能

---

### 2. Model 模块 (`internnav/model/`)

**功能**: 导航模型的架构定义和权重管理

#### 支持的模型

##### CMA (Cross-Modal Alliance)
- **位置**: `basemodel/cma/`
- **配置**: `configs/model/cma.py`
- **功能**: 多模态融合视觉-语言导航
- **特点**: 跨模态对齐,离散动作输出

##### RDP (Recurrent Diffusion Policy)
- **位置**: `basemodel/rdp/`
- **配置**: `configs/model/rdp.py`
- **功能**: 基于扩散模型的策略
- **特点**: 连续轨迹输出,时序建模

##### Seq2Seq
- **位置**: `basemodel/seq2seq/`
- **配置**: `configs/model/seq2seq.py`
- **功能**: 编码器-解码器架构
- **特点**: 经典VLN模型

##### NavDP (Navigation Diffusion Policy)
- **位置**: `basemodel/navdp/`
- **配置**: `configs/model/navdp.py`
- **功能**: 导航专用扩散政策
- **特点**: 优化的扩散过程

##### InternVLA-N1
- **位置**: `basemodel/internvla_n1/`
- **配置**: `configs/model/internvla_n1.py`
- **功能**: 双系统导航基础模型
- **特点**: 支持VLN-CE和VN任务,集成System1和System2

#### 编码器组件 (`model/encoder/`)
- **视觉编码器**: ResNet等主干网络
- **语言编码器**: LongCLIP, Qwen2.5-VL等
- **状态编码器**: RNN/GRU用于时间建模

#### 辅助组件
- `LongCLIP/`: 长文本CLIP编码器
- `diffusion_policy_modified/`: 修改的扩散政策实现

---

### 3. Dataset 模块 (`internnav/dataset/`)

**功能**: 处理多种格式的导航数据集

#### 数据集类

| 数据集类 | 文件 | 格式 |
|---------|------|------|
| `CmaLerobotDataset` | `cma_lerobot_dataset.py` | LeRobot格式 |
| `RdpLerobotDataset` | `rdp_lerobot_dataset.py` | LeRobot格式 |
| `NavDPDataset` | `navdp_dataset.py` | 原生格式 |
| `NavDPLerobotDataset` | `navdp_dataset_lerobot.py` | LeRobot格式 |
| `CMALMDBDataset` | `cma_lmdb_dataset.py` | LMDB格式 |
| `RDPLMDBDataset` | `rdp_lmdb_dataset.py` | LMDB格式 |

#### 支持的数据集
- **VLN-CE**: R2R, RxR
- **VLN-PE**: Flash, Physical
- **ClutteredEnv**: 视觉导航
- **GRScenes-100**: InternUtopia场景
- **InternData-N1**: 专有数据集

#### 数据处理流程
1. 多工作进程并行加载
2. 动态特征提取
3. 指令嵌入编码
4. 批处理和填充

---

### 4. Evaluator 模块 (`internnav/evaluator/`)

**功能**: 在模拟或真实环境中评估导航性能

#### 评估器类型

| 评估器 | 文件 | 功能 |
|--------|------|------|
| `BaseEvaluator` | `base.py` | 单机评估基类 |
| `DistributedBaseEvaluator` | `distributed_base.py` | 分布式评估框架 |
| `DefaultEvaluator` | `default_evaluator.py` | 默认评估实现 |
| `VLNDistributedEvaluator` | `vln_distributed_evaluator.py` | VLN任务分布式评估 |

#### 评估工具 (`evaluator/utils/`)
- `discrete_planner.py`: 离散动作规划器
- `continuous_planner.py`: 连续轨迹规划器
- `stuck_checker.py`: 检测代理卡住状态
- `data_collector.py`: 数据收集和统计
- `result_logger.py`: 结果日志记录
- `visualize_util.py`: 结果可视化

#### 关键评估指标
- **SR** (Success Rate): 成功率
- **SPL** (Success weighted by Path Length): 路径加权成功率
- **NE** (Navigation Error): 导航误差
- **OS** (Oracle Success): 理想成功率

---

### 5. Env 模块 (`internnav/env/`)

**功能**: 集成不同的模拟和真实环境

#### 环境类型

| 环境 | 文件 | 说明 |
|------|------|------|
| `InternutopiaEnv` | `internutopia_env.py` | InternUtopia模拟器 |
| `RealWorldAgilexEnv` | `realworld_agilex_env.py` | Unitree机器人真实环境 |
| `HabitatEnv` | `habitat_env.py` (在habitat_extensions/) | Habitat 2.0模拟器 |

#### 环境扩展 (`env/utils/`)
- `episode_loader/`: 数据集加载器
- `agilex_extensions/`: AGilex机器人扩展
- `internutopia_extension/`: InternUtopia扩展

#### Habitat扩展 (`habitat_extensions/`)
- 自定义传感器 (VLN摄像头等)
- 自定义控制器 (离散/连续)
- 自定义任务定义
- 自定义指标计算

---

### 6. Trainer 模块 (`internnav/trainer/`)

**功能**: 统一的模型训练框架

#### 训练器类型

| 训练器 | 文件 | 模型 |
|--------|------|------|
| `BaseTrainer` | `base.py` | 通用训练基类 |
| `CMATrainer` | `cma_trainer.py` | CMA模型 |
| `RDPTrainer` | `rdp_trainer.py` | RDP模型 |
| `NavDPTrainer` | `navdp_trainer.py` | NavDP模型 |

#### 训练特性
- 分布式训练支持 (DDP, FSDP)
- 混合精度训练 (FP16/BF16)
- 梯度累积
- 检查点管理
- TensorBoard日志
- 基于HuggingFace Transformers

---

### 7. Configs 模块 (`internnav/configs/`)

**功能**: 统一的配置管理系统

#### 配置层级

```
AgentCfg         -> 代理配置 (模型、检查点、服务器地址)
EvalCfg          -> 评估配置 (环境、任务、数据集)
ModelCfg         -> 模型配置 (架构参数)
ExpCfg           -> 实验配置 (训练超参数)
TrainerCfg       -> 训练器配置 (优化器、学习率等)
TaskCfg          -> 任务配置 (场景、机器人、指标)
EnvCfg           -> 环境配置 (模拟器设置)
```

#### 配置文件组织
- `agent/`: 代理配置
- `evaluator/`: 评估配置
- `model/`: 模型配置
- `trainer/`: 训练配置

---

## 技术栈

### 核心框架

| 库 | 版本 | 用途 |
|----|------|------|
| PyTorch | 最新 | 深度学习框架 |
| Transformers | 4.51.0 | 预训练模型和分词器 |
| Diffusers | 0.33.1 | 扩散模型实现 |
| Gym/Gymnasium | 0.29.1 | 强化学习环境接口 |
| Pydantic | 2.11.0+ | 配置数据验证 |
| Tyro | 0.9.26+ | CLI参数解析 |
| Ray | 2.47.1 | 分布式计算 |

### 视觉和多模态

- **OpenCV**: 图像处理
- **PIL/Pillow**: 图像I/O
- **Open3D**: 3D点云处理
- **LongCLIP**: 长文本CLIP编码
- **Qwen2.5-VL**: 视觉-语言预训练

### 模拟环境

- **Habitat**: Habitat 2.0导航模拟器
- **Isaac Sim**: NVIDIA物理模拟

### 数据处理

- **NumPy**: 数值计算
- **Pandas**: 数据分析
- **PyArrow**: 列式数据格式
- **Safetensors**: 模型序列化

### 工具和服务

- **FastAPI + Starlette**: 模型服务API
- **Uvicorn**: ASGI服务器
- **Rich**: 美化输出

---

## 主要入口点

### 1. 训练入口 (`scripts/train/train.py`)

```bash
python scripts/train/train.py --model_name=cma --name=my_exp
```

**功能流程**:
1. 解析CLI参数 (模型名称)
2. 加载对应的训练配置
3. 初始化模型、数据集、优化器
4. 创建训练器实例
5. 执行 `trainer.train()`
6. 保存检查点和配置

**支持的模型**: `cma`, `cma_plus`, `seq2seq`, `seq2seq_plus`, `rdp`, `navdp`

### 2. 评估入口 (`scripts/eval/eval.py`)

```bash
python scripts/eval/eval.py --config scripts/eval/configs/h1_cma_cfg.py
```

**功能流程**:
1. 解析命令行参数 (配置文件路径)
2. 动态加载配置模块
3. 填充默认评估配置
4. 根据 `eval_type` 创建评估器
5. 执行 `evaluator.eval()`
6. 收集和报告结果指标

**评估类型**:
- `default`: 单机评估
- `vln_distributed`: VLN分布式评估
- `habitat`: Habitat环境评估

### 3. 代理服务器入口 (`scripts/eval/start_server.py`)

```bash
python scripts/eval/start_server.py
```

**功能**:
- 启动FastAPI服务器 (默认8087端口)
- 注册所有Agent类
- 暴露REST API端点:
  - `POST /agent/init`: 初始化代理
  - `POST /agent/step`: 执行一步
  - `POST /agent/reset`: 重置代理

### 4. 真实世界部署 (`scripts/realworld/http_internvla_server.py`)

```bash
python http_internvla_server.py --config h1_internvla_n1_cfg.py
```

**功能**:
- 初始化机器人环境 (H1/Go2)
- 启动导航服务器
- 处理观测和返回动作
- 安全停止和错误恢复

**支持机器人**: Unitree H1, Unitree Go2

### 5. Demo应用 (`scripts/demo/vln_gradio_backend.py`)

```bash
python vln_gradio_backend.py
```

**功能**:
- 启动Gradio Web UI
- 集成导航模型推理
- 实时可视化

---

## 工作流程

### 训练工作流

```
配置定义
    ↓
加载数据集 (LeRobot/LMDB)
    ↓
初始化模型 (Transformers)
    ↓
初始化优化器 (AdamW等)
    ↓
创建训练器 (BaseTrainer/CMATrainer)
    ↓
执行训练循环
    ├─ 数据加载 (多工作进程)
    ├─ 前向传播
    ├─ 损失计算
    ├─ 反向传播
    ├─ 梯度更新
    └─ 日志记录
    ↓
保存检查点
```

### 评估工作流

```
加载配置
    ↓
初始化环境 (InternUtopia/Habitat)
    ↓
初始化代理 (从检查点加载)
    ↓
对每个回合:
    ├─ reset() -> 初始观测
    ├─ step(obs) -> 动作
    ├─ env.step(action) -> 新观测
    ├─ 收集指标 (SR, SPL等)
    └─ 重复直到完成
    ↓
计算和报告结果
```

### 推理工作流

```
代理初始化
    ↓
环境交互循环
    ├─ 获取观测
    │   ├─ RGB图像
    │   ├─ 深度图像
    │   ├─ 指令文本
    │   └─ 机器人状态
    ├─ 模型前向传播
    │   ├─ 视觉编码
    │   ├─ 语言编码
    │   ├─ 多模态融合
    │   └─ 动作输出
    ├─ 执行动作
    └─ 重复直到完成任务
```

---

## 扩展指南

### 添加新代理

```python
from internnav.agent.base import Agent

@Agent.register('my_agent')
class MyAgent(Agent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        # 初始化模型

    def step(self, obs):
        # 推理逻辑
        return action

    def reset(self):
        # 重置逻辑
        pass
```

### 添加新模型

1. 在 `internnav/model/basemodel/` 创建模型目录
2. 实现模型类继承自 `PreTrainedModel`
3. 在 `internnav/model/__init__.py` 中注册
4. 创建对应的配置文件 `internnav/configs/model/my_model.py`

### 添加新环境

```python
from internnav.env.base import Env

@Env.register('my_env')
class MyEnv(Env):
    def __init__(self, env_cfg, task_cfg):
        super().__init__(env_cfg, task_cfg)
        # 初始化环境

    def reset(self):
        return observation

    def step(self, action):
        return observation, reward, done, info
```

---

## 配置文件详解

### 依赖配置 (`requirements/`)

```
requirements/
├── core_requirements.txt             # 核心依赖(FastAPI, Pydantic等)
├── model_requirements.txt            # 模型库(Transformers等)
├── habitat_requirements.txt          # Habitat模拟器
├── isaac_requirements.txt            # Isaac Sim依赖
└── internvla_n1.txt                  # InternVLA-N1专有依赖
```

### 模型配置 (`internnav/configs/model/`)

每个模型都有对应的配置文件,定义模型架构参数:
- `cma.py`: CMA模型配置
- `rdp.py`: RDP模型配置
- `seq2seq.py`: Seq2Seq模型配置
- `navdp.py`: NavDP模型配置
- `internvla_n1.py`: InternVLA-N1配置

### 训练配置 (`scripts/train/configs/`)

提供了各种训练场景的示例配置:
- `cma.py`, `cma_plus.py`: CMA训练配置
- `seq2seq.py`, `seq2seq_plus.py`: Seq2Seq训练配置
- `rdp.py`: RDP训练配置
- `navdp.py`: NavDP训练配置

### 评估配置 (`scripts/eval/configs/`)

不同环境和模型组合的评估配置:
- `h1_cma_cfg.py`: H1机器人 + CMA
- `h1_rdp_cfg.py`: H1机器人 + RDP
- `h1_seq2seq_cfg.py`: H1机器人 + Seq2Seq
- `h1_internvla_n1_async_cfg.py`: H1机器人 + InternVLA-N1
- `habitat_s2_cfg.py`: Habitat + System2
- `habitat_dual_system_cfg.py`: Habitat + 双系统

---

## 关键接口

### Agent接口

```python
class Agent:
    """代理基类"""
    def step(self, obs: Dict[str, Any]) -> action
    def reset(self) -> None

    @classmethod
    def init(cls, config: AgentCfg) -> Agent

    @classmethod
    def register(cls, agent_type: str) -> decorator
```

### Evaluator接口

```python
class Evaluator:
    """评估器基类"""
    def __init__(self, config: EvalCfg)
    def eval(self) -> results

    @classmethod
    def init(cls, config: EvalCfg) -> Evaluator

    @classmethod
    def register(cls, evaluator_type: str) -> decorator
```

### Env接口

```python
class Env:
    """环境基类"""
    def reset(self) -> observation
    def step(self, action) -> (observation, reward, done, info)
    def close(self) -> None
    def render(self) -> None
    def get_observation(self) -> Dict
    def get_info(self) -> Dict

    @classmethod
    def init(cls, env_cfg: EnvCfg, task_cfg: TaskCfg) -> Env

    @classmethod
    def register(cls, env_type: str) -> decorator
```

---

## 快速导航表

| 需求 | 对应文件/目录 |
|------|-------------|
| 添加新模型 | `internnav/model/basemodel/` |
| 添加新数据集 | `internnav/dataset/` |
| 修改训练配置 | `scripts/train/configs/` |
| 修改评估配置 | `scripts/eval/configs/` |
| 实现新环境 | `internnav/env/` |
| 添加新代理 | `internnav/agent/` |
| 编码器扩展 | `internnav/model/encoder/` |
| 评估指标 | `internnav/evaluator/utils/` |
| 模型配置 | `internnav/configs/model/` |
| 通信服务 | `internnav/utils/comm_utils/` |

---

## 项目统计

- **Python文件总数**: 351个
- **核心模块**: 7个 (agent, model, dataset, evaluator, env, trainer, configs)
- **支持模型**: 6个 (CMA, RDP, Seq2Seq, NavDP, InternVLA-N1及其变体)
- **支持环境**: 3个 (InternUtopia, Habitat, 真实机器人)
- **数据格式**: 3种 (LeRobot, LMDB, 原生)
- **Python版本**: 3.8 - 3.12
- **许可证**: MIT License

---

## 版本信息

- **当前版本**: 0.2.0
- **最近更新**: Habitat重构和分布式VLNPE重构
- **开发分支**: `main`, `dev`

---

此架构文档为 InternNav 项目提供了全面的结构化参考,涵盖目录组织、核心模块、技术栈、工作流程和扩展指南,便于理解项目设计、开发新功能和集成第三方组件。
