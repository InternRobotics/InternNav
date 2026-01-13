# InternVLA-N1 Model Testing Guide

本指南说明如何测试 InternVLA-N1 模型（单个系统）。

## 前置要求

### 1. 下载检查点

#### InternVLA-N1 预训练检查点
- 下载最新的预训练检查点并将其放置在 `checkpoints` 目录下
- 例如：`checkpoints/InternVLA-N1` 或 `/data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger`

#### DepthAnything v2 检查点
- 下载 DepthAnything v2 预训练检查点
- 将检查点放置在 `checkpoints` 目录下

### 2. 安装依赖

确保已正确安装 InternNav：

```bash
pip install -e .
```

## 快速开始

### 方法 1: 使用辅助脚本（推荐）

#### 步骤 1: 启动服务器

在**第一个终端**中运行：

```bash
./scripts/start_test_server.sh
```

或指定端口：

```bash
./scripts/start_test_server.sh 8087
```

服务器启动后，你应该看到类似如下输出：

```
Starting Agent Server...
Registering agents...
INFO:     Started server process [18877]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8087 (Press CTRL+C to quit)
```

#### 步骤 2: 运行测试

在**第二个终端**中运行：

```bash
python scripts/test_internvla_n1.py \
    --checkpoint /data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger
```

### 方法 2: 手动执行

#### 步骤 1: 启动服务器

```bash
python scripts/eval/start_server.py --port 8087
```

#### 步骤 2: 运行测试脚本

```bash
python scripts/test_internvla_n1.py --checkpoint <checkpoint_path>
```

## 测试脚本选项

```bash
python scripts/test_internvla_n1.py [OPTIONS]
```

### 必需参数：

- `--checkpoint PATH`: InternVLA-N1 检查点目录路径

### 可选参数：

- `--rs-meta PATH`: rs_meta.json 文件路径（默认: scripts/iros_challenge/onsite_competition/captures/rs_meta.json）
- `--server-host HOST`: 服务器地址（默认: localhost）
- `--server-port PORT`: 服务器端口（默认: 8087）
- `--device DEVICE`: CUDA 设备（默认: cuda:0）
- `--instruction TEXT`: 导航指令（默认: "go to the red car"）

### 示例

#### 基本测试

```bash
python scripts/test_internvla_n1.py \
    --checkpoint checkpoints/InternVLA-N1
```

#### 使用自定义观察数据

```bash
python scripts/test_internvla_n1.py \
    --checkpoint checkpoints/InternVLA-N1 \
    --rs-meta /path/to/your/rs_meta.json
```

#### 指定不同的导航指令

```bash
python scripts/test_internvla_n1.py \
    --checkpoint checkpoints/InternVLA-N1 \
    --instruction "go to the kitchen"
```

#### 使用不同的 GPU

```bash
python scripts/test_internvla_n1.py \
    --checkpoint checkpoints/InternVLA-N1 \
    --device cuda:1
```

## 预期输出

### 服务器输出

```
Loading navdp model: NavDP_Policy_DPT_CriticSum_DAT
Pretrained: None
No pretrained weights provided, initializing randomly.
Loading checkpoint shards: 100%|██████████| 4/4 [00:03<00:00,  1.06it/s]
INFO:     ::1:38332 - "POST /agent/init HTTP/1.1" 201 Created
```

### 测试脚本输出

```
================================================================================
InternVLA-N1 Model Test
================================================================================
Checkpoint: checkpoints/InternVLA-N1
RS Meta: scripts/iros_challenge/onsite_competition/captures/rs_meta.json
Server: localhost:8087
Device: cuda:0
Instruction: go to the red car
================================================================================

[Step 1] Configuring InternVLA-N1 agent...
Agent configuration created successfully.

[Step 2] Initializing agent client...
Connecting to server at localhost:8087...
✓ Agent client initialized successfully!

[Step 3] Loading observation from RS meta file...
✓ Observation loaded successfully!
  RGB shape: (480, 640, 3)
  Depth shape: (480, 640)
  Instruction: go to the red car

[Step 4] Running model inference...
This may take a moment...
============ output 1  ←←←←
s2 infer finish!!
get s2 output lock
=============== [2, 2, 2, 2] =================
Output discretized traj: [2] 0
✓ Inference completed in 3.45 seconds!

================================================================================
RESULT:
================================================================================
Action taken: 2
Action meaning: TURN_LEFT
================================================================================

✓ Test completed successfully!
```

## 动作含义

模型输出的离散动作映射：

- `0`: MOVE_FORWARD（前进）
- `1`: TURN_RIGHT（右转）
- `2`: TURN_LEFT（左转）
- `3`: STOP（停止）

## 故障排除

### 问题 1: 无法连接到服务器

**错误信息:**
```
✗ Failed to initialize agent client: ...
```

**解决方案:**
1. 确保服务器正在运行：
   ```bash
   ./scripts/start_test_server.sh
   ```
2. 检查端口是否被占用：
   ```bash
   lsof -i :8087
   ```
3. 确保防火墙未阻止该端口

### 问题 2: 检查点路径不存在

**错误信息:**
```
Error: Checkpoint path does not exist: ...
```

**解决方案:**
1. 验证检查点路径：
   ```bash
   ls -la /path/to/checkpoint
   ```
2. 确保已下载检查点文件
3. 检查路径拼写是否正确

### 问题 3: 观察数据加载失败

**错误信息:**
```
✗ Failed to load observation: ...
```

**解决方案:**
1. 检查 rs_meta.json 文件是否存在：
   ```bash
   ls -la scripts/iros_challenge/onsite_competition/captures/rs_meta.json
   ```
2. 验证 meta 文件中引用的图像路径是否正确
3. 使用自己的观察数据：参见下一节

### 问题 4: CUDA 内存不足

**错误信息:**
```
CUDA out of memory
```

**解决方案:**
1. 使用不同的 GPU：
   ```bash
   python scripts/test_internvla_n1.py --checkpoint ... --device cuda:1
   ```
2. 减小批处理大小（需要修改配置）
3. 释放 GPU 内存：
   ```bash
   nvidia-smi
   # 杀掉占用 GPU 的进程
   kill -9 <PID>
   ```

## 使用自定义观察数据

### 从 RealSense 相机保存数据

```python
from scripts.iros_challenge.onsite_competition.sdk.save_obs import save_obs

# 从相机获取数据
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

# 保存观察数据
save_obs(obs, outdir='./my_captures', prefix='my_obs')
```

### 加载自定义数据进行测试

```bash
python scripts/test_internvla_n1.py \
    --checkpoint checkpoints/InternVLA-N1 \
    --rs-meta ./my_captures/my_obs_meta.json \
    --instruction "your custom instruction"
```

## 下一步

成功完成单次推理测试后，你可以：

1. **集成到实际机器人控制器**：
   - 使用 `internnav.env.real_world_env` 模块
   - 将预测的动作应用到机器人控制器

2. **批量评估**：
   - 使用完整的评估脚本：
     ```bash
     python scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_async_cfg.py
     ```

3. **可视化结果**：
   - 设置 `vis_debug=True` 在配置中
   - 查看 `./logs/test_n1/vis_debug` 目录

4. **查看演示视频**：
   - 查看实际部署演示视频了解完整流程

## 参考文献

- [InternNav Documentation](https://github.com/yourusername/InternNav)
- [Agent Configuration](internnav/configs/agent/__init__.py)
- [Agent Client](internnav/utils/comm_utils/client.py)
- [Real World Environment](internnav/env/real_world_env)

## 联系支持

如遇到问题，请：
1. 检查日志文件
2. 查看 GitHub Issues
3. 联系技术支持团队
