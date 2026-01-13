# InternVLA-N1 测试 Notebooks

本目录包含用于测试 InternVLA-N1 模型的 Jupyter Notebook 和 Python 脚本。

## 文件说明

### 1. `test_internvla_n1.ipynb`
完整的 Jupyter Notebook，包含 12 个独立单元格：
- 依次运行每个单元格即可完成测试
- 包含可视化和详细说明
- 适合交互式测试和调试

### 2. `test_blocks.py`
分块的 Python 脚本：
- 代码块以 `# ========== 块 X: ... ==========` 分隔
- 可以复制粘贴到 Jupyter Notebook 中
- 每个块都可以独立运行

## 使用方法

### 方法 1: 使用 Jupyter Notebook（推荐）

#### 步骤 1: 启动服务器
在终端中运行：
```bash
cd /home/user/InternNav
python scripts/eval/start_server.py --port 8087
```

#### 步骤 2: 启动 Jupyter
在另一个终端中运行：
```bash
cd /home/user/InternNav
jupyter notebook notebooks/test_internvla_n1.ipynb
```

#### 步骤 3: 修改配置
在 Notebook 的**块 2**中，修改以下参数：
```python
# 修改为你的实际检查点路径
checkpoint_path = '/data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger'

# 如果需要，修改其他参数
device = 'cuda:0'
instruction = 'go to the red car'
```

#### 步骤 4: 依次运行
依次运行每个单元格（Shift + Enter）

### 方法 2: 使用 Python 脚本

#### 步骤 1: 启动服务器
```bash
python scripts/eval/start_server.py --port 8087
```

#### 步骤 2: 打开 Jupyter Notebook
```bash
jupyter notebook
```

#### 步骤 3: 创建新 Notebook
在 Jupyter 中创建新的 Notebook

#### 步骤 4: 复制代码块
打开 `test_blocks.py`，复制每个代码块到新的单元格中

#### 步骤 5: 依次运行
修改配置参数后，依次运行每个单元格

## 代码块说明

### 必需块（必须运行）
1. **块 1**: 导入依赖库
2. **块 2**: 配置参数（需要修改）
3. **块 3**: 验证路径
4. **块 4**: 配置 Agent
5. **块 5**: 初始化 Agent Client
6. **块 6**: 加载观察数据
8. **块 8**: 运行模型推理
9. **块 9**: 解析动作

### 可选块
- **块 7**: 可视化观察数据
- **块 10**: 查看完整结果（调试用）
- **块 11**: 批量测试
- **块 12**: 保存测试结果

## 配置参数说明

```python
# 必须修改的参数
checkpoint_path = '/data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger'  # 检查点路径

# 可选修改的参数
rs_meta_path = '../scripts/iros_challenge/onsite_competition/captures/rs_meta.json'  # 观察数据
device = 'cuda:0'          # CUDA 设备
instruction = 'go to the red car'  # 导航指令

# 高级参数（通常不需要修改）
camera_intrinsic = [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]]
width = 640
height = 480
hfov = 79
```

## 预期输出

### 成功的输出示例

**块 5 (初始化 Agent):**
```
连接到服务器 localhost:8087...
✓ Agent 客户端初始化成功！
  模型已加载到服务器
```

**块 6 (加载观察数据):**
```
✓ 观察数据加载成功
  RGB shape: (480, 640, 3)
  Depth shape: (480, 640)
  Instruction: go to the red car
  Timestamp: 1759218399.0439963
```

**块 8 (推理结果):**
```
================================================================================
推理结果
================================================================================
预测动作: 2
推理时间: 3.45 秒
================================================================================
```

**块 9 (动作解析):**
```
动作编号: 2
动作名称: TURN_LEFT
中文含义: 左转

✓ 测试完成！模型成功预测了动作。
```

## 动作映射

| 编号 | 英文 | 中文 |
|-----|------|------|
| 0 | MOVE_FORWARD | 前进 |
| 1 | TURN_RIGHT | 右转 |
| 2 | TURN_LEFT | 左转 |
| 3 | STOP | 停止 |

## 故障排除

### 问题 1: 无法导入模块

**错误**: `ModuleNotFoundError: No module named 'internnav'`

**解决**:
```bash
# 确保在项目根目录
cd /home/user/InternNav

# 安装项目
pip install -e .
```

### 问题 2: 无法连接服务器

**错误**: `✗ 初始化失败: Connection refused`

**解决**:
1. 检查服务器是否启动：
   ```bash
   ps aux | grep start_server.py
   ```
2. 重新启动服务器：
   ```bash
   python scripts/eval/start_server.py --port 8087
   ```

### 问题 3: 检查点路径不存在

**错误**: `✗ 检查点路径不存在`

**解决**:
1. 检查路径：
   ```bash
   ls -la /data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger
   ```
2. 在块 2 中修改为正确的路径

### 问题 4: 观察数据加载失败

**错误**: `Failed to read depth image`

**解决**:
1. 检查文件存在：
   ```bash
   ls -la scripts/iros_challenge/onsite_competition/captures/
   ```
2. 确认文件完整性
3. 或使用自己的观察数据

## 使用自定义数据

### 保存自己的观察数据

```python
from scripts.iros_challenge.onsite_competition.sdk.save_obs import save_obs

# 准备观察数据
obs = {
    'rgb': rgb_image,      # numpy array (H, W, 3) BGR uint8
    'depth': depth_image,  # numpy array (H, W) float32 in meters
    'timestamp_s': time.time(),
    'intrinsics': {
        'fx': 585.0,
        'fy': 585.0,
        'cx': 320.0,
        'cy': 240.0,
    }
}

# 保存
save_obs(obs, outdir='./my_data', prefix='my_obs')
```

### 加载自定义数据

在块 2 中修改：
```python
rs_meta_path = './my_data/my_obs_meta.json'
```

## 提示和技巧

### 1. 快速重新测试
如果只想改变指令，无需重新初始化：
```python
# 直接在新单元格中运行
obs['instruction'] = 'new instruction'
result = agent.step([obs])
action = result[0]['action'][0]
print(f"动作: {action}")
```

### 2. 批量测试多个检查点
```python
checkpoints = [
    '/path/to/checkpoint1',
    '/path/to/checkpoint2',
]

for ckpt in checkpoints:
    # 修改配置
    agent_cfg.model_settings['model_path'] = ckpt
    # 重新初始化
    agent = AgentClient(agent_cfg)
    # 测试...
```

### 3. 保存推理结果
```python
results = []
for instruction in test_instructions:
    obs['instruction'] = instruction
    result = agent.step([obs])
    results.append({
        'instruction': instruction,
        'action': result[0]['action'][0]
    })

# 保存
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## 下一步

测试成功后，你可以：

1. **实际部署**: 使用 `internnav.env.real_world_env`
2. **完整评估**: 运行 `scripts/eval/eval.py`
3. **可视化**: 启用 `vis_debug` 选项
4. **调优**: 修改模型参数和配置

## 相关文档

- [QUICK_START.md](../QUICK_START.md) - 命令行快速开始
- [TESTING_GUIDE.md](../TESTING_GUIDE.md) - 详细测试指南
- [scripts/test_internvla_n1.py](../scripts/test_internvla_n1.py) - 命令行测试脚本

---

**祝测试愉快！** 如有问题，请查看相关文档或提交 issue。
