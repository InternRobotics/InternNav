# InternVLA-N1 快速测试指南

## 📋 文件概览

我已经为你创建了以下测试文件：

1. **`scripts/test_internvla_n1.py`** - 主测试脚本
2. **`scripts/start_test_server.sh`** - 服务器启动脚本
3. **`TESTING_GUIDE.md`** - 详细测试指南

## 🚀 快速开始

### 第 1 步：启动服务器

在**第一个终端**运行：

```bash
# 方法 1: 使用辅助脚本
./scripts/start_test_server.sh

# 方法 2: 直接运行
python scripts/eval/start_server.py --port 8087
```

### 第 2 步：运行测试

在**第二个终端**运行：

```bash
python scripts/test_internvla_n1.py \
    --checkpoint /data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger
```

⚠️ **重要提示**：请根据你的实际检查点路径修改 `--checkpoint` 参数。

## 📝 完整命令示例

```bash
# 基本用法
python scripts/test_internvla_n1.py \
    --checkpoint /data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger

# 使用自定义观察数据
python scripts/test_internvla_n1.py \
    --checkpoint /data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger \
    --rs-meta /path/to/your/rs_meta.json

# 使用自定义指令
python scripts/test_internvla_n1.py \
    --checkpoint /data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger \
    --instruction "go to the kitchen"

# 使用不同的 GPU
python scripts/test_internvla_n1.py \
    --checkpoint /data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger \
    --device cuda:1

# 查看所有选项
python scripts/test_internvla_n1.py --help
```

## 🔧 测试脚本参数

### 必需参数
- `--checkpoint PATH`: InternVLA-N1 检查点目录路径

### 可选参数
- `--rs-meta PATH`: 观察数据文件路径（默认：使用示例数据）
- `--server-host HOST`: 服务器地址（默认：localhost）
- `--server-port PORT`: 服务器端口（默认：8087）
- `--device DEVICE`: CUDA 设备（默认：cuda:0）
- `--instruction TEXT`: 导航指令（默认："go to the red car"）

## 📊 预期输出

成功运行后，你会看到类似如下输出：

```
================================================================================
InternVLA-N1 Model Test
================================================================================
Checkpoint: /data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger
...
✓ Inference completed in 3.45 seconds!

================================================================================
RESULT:
================================================================================
Action taken: 2
Action meaning: TURN_LEFT
================================================================================
```

## 🎮 动作映射

模型输出的离散动作：

- `0` = MOVE_FORWARD（前进）
- `1` = TURN_RIGHT（右转）
- `2` = TURN_LEFT（左转）
- `3` = STOP（停止）

## 🐛 常见问题

### 1. 检查点路径不存在

**错误**：`Error: Checkpoint path does not exist`

**解决**：
```bash
# 检查检查点是否存在
ls -la /data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger

# 如果路径不同，使用正确的路径
python scripts/test_internvla_n1.py --checkpoint <正确的路径>
```

### 2. 无法连接到服务器

**错误**：`Failed to initialize agent client`

**解决**：
- 确保服务器在第一个终端中正在运行
- 检查端口 8087 是否被占用：`lsof -i :8087`

### 3. 缺少依赖

**错误**：`ModuleNotFoundError`

**解决**：
```bash
# 安装 InternNav
pip install -e .

# 或安装特定依赖
pip install pydantic fastapi uvicorn
```

## 📚 更多信息

查看 **`TESTING_GUIDE.md`** 获取：
- 详细的故障排除指南
- 如何使用自定义观察数据
- 批量评估方法
- 实际机器人部署步骤

## 📁 项目结构

```
InternNav/
├── scripts/
│   ├── test_internvla_n1.py          # 测试脚本（新增）
│   ├── start_test_server.sh          # 服务器启动脚本（新增）
│   ├── eval/
│   │   ├── start_server.py           # 服务器主程序
│   │   └── configs/
│   │       └── h1_internvla_n1_async_cfg.py
│   └── iros_challenge/
│       └── onsite_competition/
│           ├── sdk/save_obs.py       # 观察数据处理
│           └── captures/
│               ├── rs_meta.json      # 示例观察数据
│               ├── rs_rgb.jpg
│               └── rs_depth_mm.png
├── TESTING_GUIDE.md                  # 详细测试指南（新增）
└── QUICK_START.md                    # 本文件（新增）
```

## ✅ 验证清单

运行测试前，请确保：

- [ ] 已下载 InternVLA-N1 检查点
- [ ] 已下载 DepthAnything v2 检查点（如需要）
- [ ] 已安装项目依赖：`pip install -e .`
- [ ] 服务器已在第一个终端启动
- [ ] 检查点路径正确

## 🎯 下一步

成功运行单次测试后，你可以：

1. **批量评估**：使用 `scripts/eval/eval.py`
2. **实际部署**：集成到机器人控制器
3. **自定义配置**：修改模型参数和设置

---

**祝测试顺利！** 如有问题，请查看 `TESTING_GUIDE.md` 或提交 issue。
