"""
InternVLA-N1 模型测试脚本（分块版本）

使用方法：
1. 先在终端启动服务器：python scripts/eval/start_server.py --port 8087
2. 复制下面的代码块到 Jupyter Notebook 中
3. 依次运行每个代码块

每个代码块以 # ========== 分隔，可以独立运行
"""

# ========== 块 1: 导入依赖库 ==========
import sys
sys.path.append('.')
sys.path.append('..')

import os
import time
import numpy as np
import cv2

from internnav.configs.agent import AgentCfg
from internnav.utils import AgentClient
from scripts.iros_challenge.onsite_competition.sdk.save_obs import load_obs_from_meta

print("✓ 依赖库导入成功")


# ========== 块 2: 配置参数 ==========
# 模型检查点路径（请根据实际情况修改）
checkpoint_path = '/data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger'

# 观察数据文件路径
rs_meta_path = '../scripts/iros_challenge/onsite_competition/captures/rs_meta.json'

# 服务器配置
server_host = 'localhost'
server_port = 8087

# 设备配置
device = 'cuda:0'

# 导航指令
instruction = 'go to the red car'

# 相机参数
camera_intrinsic = [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]]
width = 640
height = 480
hfov = 79

print("配置参数：")
print(f"  检查点路径: {checkpoint_path}")
print(f"  观察数据: {rs_meta_path}")
print(f"  服务器: {server_host}:{server_port}")
print(f"  设备: {device}")
print(f"  指令: {instruction}")


# ========== 块 3: 验证路径 ==========
# 检查检查点路径
if os.path.exists(checkpoint_path):
    print(f"✓ 检查点路径存在: {checkpoint_path}")
    files = os.listdir(checkpoint_path)
    print(f"  包含 {len(files)} 个文件/目录")
else:
    print(f"✗ 检查点路径不存在: {checkpoint_path}")
    print("  请修改 checkpoint_path 为正确的路径")

# 检查观察数据路径
if os.path.exists(rs_meta_path):
    print(f"✓ 观察数据文件存在: {rs_meta_path}")
else:
    print(f"✗ 观察数据文件不存在: {rs_meta_path}")
    print("  请修改 rs_meta_path 为正确的路径")


# ========== 块 4: 配置 Agent ==========
agent_cfg = AgentCfg(
    server_host=server_host,
    server_port=server_port,
    model_name='internvla_n1',
    ckpt_path='',
    model_settings={
        'policy_name': "InternVLAN1_Policy",
        'state_encoder': None,
        'env_num': 1,
        'sim_num': 1,
        'model_path': checkpoint_path,
        'camera_intrinsic': camera_intrinsic,
        'width': width,
        'height': height,
        'hfov': hfov,
        'resize_w': 384,
        'resize_h': 384,
        'max_new_tokens': 1024,
        'num_frames': 32,
        'num_history': 8,
        'num_future_steps': 4,
        'device': device,
        'predict_step_nums': 32,
        'continuous_traj': True,
    }
)

print("✓ Agent 配置创建成功")
print(f"  模型名称: {agent_cfg.model_name}")
print(f"  模型路径: {agent_cfg.model_settings['model_path']}")


# ========== 块 5: 初始化 Agent Client ==========
# ⚠️ 注意：运行此块前，请确保服务器已启动！
# 在终端运行：python scripts/eval/start_server.py --port 8087

print(f"连接到服务器 {server_host}:{server_port}...")

try:
    agent = AgentClient(agent_cfg)
    print("✓ Agent 客户端初始化成功！")
    print("  模型已加载到服务器")
except Exception as e:
    print(f"✗ 初始化失败: {e}")
    print("\n请检查：")
    print("  1. 服务器是否正在运行")
    print("  2. 端口号是否正确")
    print("  3. 检查点路径是否正确")
    raise


# ========== 块 6: 加载观察数据 ==========
# 加载观察数据
obs = load_obs_from_meta(rs_meta_path)

# 添加导航指令
obs['instruction'] = instruction

print("✓ 观察数据加载成功")
print(f"  RGB shape: {obs['rgb'].shape}")
print(f"  Depth shape: {obs['depth'].shape}")
print(f"  Instruction: {obs['instruction']}")
print(f"  Timestamp: {obs['timestamp_s']}")


# ========== 块 7: 可视化观察数据（可选）==========
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# RGB 图像（BGR -> RGB）
rgb_display = cv2.cvtColor(obs['rgb'], cv2.COLOR_BGR2RGB)
axes[0].imshow(rgb_display)
axes[0].set_title('RGB Image')
axes[0].axis('off')

# 深度图像
depth_display = obs['depth']
im = axes[1].imshow(depth_display, cmap='viridis')
axes[1].set_title('Depth Image (meters)')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()

print(f"Depth range: {np.nanmin(depth_display):.2f}m - {np.nanmax(depth_display):.2f}m")


# ========== 块 8: 运行模型推理 ==========
print("开始推理...")
print(f"指令: {obs['instruction']}")
print()

# 记录开始时间
start_time = time.time()

# 执行推理
result = agent.step([obs])

# 计算推理时间
inference_time = time.time() - start_time

# 提取动作
action = result[0]['action'][0]

print("=" * 80)
print("推理结果")
print("=" * 80)
print(f"预测动作: {action}")
print(f"推理时间: {inference_time:.2f} 秒")
print("=" * 80)


# ========== 块 9: 解析动作 ==========
# 动作映射
action_map = {
    0: "MOVE_FORWARD",
    1: "TURN_RIGHT",
    2: "TURN_LEFT",
    3: "STOP"
}

action_names = {
    0: "前进",
    1: "右转",
    2: "左转",
    3: "停止"
}

if action in action_map:
    print(f"动作编号: {action}")
    print(f"动作名称: {action_map[action]}")
    print(f"中文含义: {action_names[action]}")
else:
    print(f"未知动作: {action}")

print(f"\n✓ 测试完成！模型成功预测了动作。")


# ========== 块 10: 查看完整结果（调试用）==========
print("完整结果字典：")
print()
for key, value in result[0].items():
    print(f"{key}:")
    if isinstance(value, np.ndarray):
        print(f"  类型: numpy.ndarray")
        print(f"  形状: {value.shape}")
        print(f"  数据类型: {value.dtype}")
        if value.size < 10:
            print(f"  值: {value}")
    else:
        print(f"  值: {value}")
    print()


# ========== 块 11: 批量测试（可选）==========
# 示例：测试不同的指令
test_instructions = [
    'go to the red car',
    'go to the kitchen',
    'turn left and go straight',
    'find the bedroom'
]

print("批量测试不同指令：")
print("=" * 80)

for idx, test_instruction in enumerate(test_instructions, 1):
    # 创建新的观察数据
    test_obs = obs.copy()
    test_obs['instruction'] = test_instruction

    # 执行推理
    start_time = time.time()
    result = agent.step([test_obs])
    inference_time = time.time() - start_time

    action = result[0]['action'][0]

    # 输出结果
    print(f"测试 {idx}: {test_instruction}")
    print(f"  动作: {action} ({action_map.get(action, 'Unknown')} / {action_names.get(action, '未知')})")
    print(f"  时间: {inference_time:.2f}s")
    print()

print("=" * 80)
print("批量测试完成！")


# ========== 块 12: 保存测试结果（可选）==========
import json
from datetime import datetime

# 准备保存的结果
test_result = {
    'timestamp': datetime.now().isoformat(),
    'checkpoint': checkpoint_path,
    'instruction': instruction,
    'action': int(action),
    'action_name': action_map.get(action, 'Unknown'),
    'inference_time': inference_time,
    'device': device,
}

# 保存到文件
output_file = 'test_result.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(test_result, f, indent=2, ensure_ascii=False)

print(f"✓ 测试结果已保存到: {output_file}")
print("\n内容：")
print(json.dumps(test_result, indent=2, ensure_ascii=False))
