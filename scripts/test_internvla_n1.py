#!/usr/bin/env python
"""
InternVLA-N1 Model Testing Script

This script tests the InternVLA-N1 model using the client-server architecture.
Make sure to start the server first using: python scripts/eval/start_server.py --port 8087

Usage:
    python scripts/test_internvla_n1.py --checkpoint <path_to_checkpoint>
"""

import sys
sys.path.append('.')

import argparse
import os
import time
from pathlib import Path

from internnav.configs.agent import AgentCfg
from internnav.utils import AgentClient
from scripts.iros_challenge.onsite_competition.sdk.save_obs import load_obs_from_meta


def test_internvla_n1(
    checkpoint_path: str,
    rs_meta_path: str = None,
    server_host: str = 'localhost',
    server_port: int = 8087,
    device: str = 'cuda:0',
    instruction: str = 'go to the red car'
):
    """
    Test InternVLA-N1 model inference.

    Args:
        checkpoint_path: Path to the InternVLA-N1 checkpoint
        rs_meta_path: Path to the rs_meta.json file (default: scripts/iros_challenge/onsite_competition/captures/rs_meta.json)
        server_host: Agent server host
        server_port: Agent server port
        device: CUDA device to use
        instruction: Navigation instruction
    """

    # Validate checkpoint path
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        print(f"Please provide a valid checkpoint path.")
        return False

    # Set default rs_meta_path if not provided
    if rs_meta_path is None:
        rs_meta_path = 'scripts/iros_challenge/onsite_competition/captures/rs_meta.json'

    if not os.path.exists(rs_meta_path):
        print(f"Error: RS meta file does not exist: {rs_meta_path}")
        print(f"Please provide a valid rs_meta.json path or use the default sample.")
        return False

    print("=" * 80)
    print("InternVLA-N1 Model Test")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"RS Meta: {rs_meta_path}")
    print(f"Server: {server_host}:{server_port}")
    print(f"Device: {device}")
    print(f"Instruction: {instruction}")
    print("=" * 80)

    # Step 1: Configure agent
    print("\n[Step 1] Configuring InternVLA-N1 agent...")

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
            'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            'width': 640,
            'height': 480,
            'hfov': 79,
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

    print(f"Agent configuration created successfully.")

    # Step 2: Initialize agent client
    print("\n[Step 2] Initializing agent client...")
    print(f"Connecting to server at {server_host}:{server_port}...")

    try:
        agent = AgentClient(agent_cfg)
        print("✓ Agent client initialized successfully!")
    except Exception as e:
        print(f"✗ Failed to initialize agent client: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the server is running:")
        print(f"   python scripts/eval/start_server.py --port {server_port}")
        print("2. Check if the port is correct and not blocked by firewall")
        return False

    # Step 3: Load observation data
    print("\n[Step 3] Loading observation from RS meta file...")

    try:
        obs = load_obs_from_meta(rs_meta_path)
        obs['instruction'] = instruction

        print(f"✓ Observation loaded successfully!")
        print(f"  RGB shape: {obs['rgb'].shape}")
        print(f"  Depth shape: {obs['depth'].shape}")
        print(f"  Instruction: {obs['instruction']}")
    except Exception as e:
        print(f"✗ Failed to load observation: {e}")
        return False

    # Step 4: Run inference
    print("\n[Step 4] Running model inference...")
    print("This may take a moment...")

    try:
        start_time = time.time()
        result = agent.step([obs])
        inference_time = time.time() - start_time

        action = result[0]['action'][0]

        print(f"✓ Inference completed in {inference_time:.2f} seconds!")
        print(f"\n{'=' * 80}")
        print(f"RESULT:")
        print(f"{'=' * 80}")
        print(f"Action taken: {action}")

        # Action mapping for discrete actions
        action_map = {
            0: "MOVE_FORWARD",
            1: "TURN_RIGHT",
            2: "TURN_LEFT",
            3: "STOP"
        }

        if action in action_map:
            print(f"Action meaning: {action_map[action]}")

        print(f"{'=' * 80}")

        # Print full result for debugging
        print("\n[Debug] Full result:")
        for key, value in result[0].items():
            print(f"  {key}: {value}")

        print("\n✓ Test completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test InternVLA-N1 model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to InternVLA-N1 checkpoint directory'
    )
    parser.add_argument(
        '--rs-meta',
        type=str,
        default=None,
        help='Path to rs_meta.json file (default: scripts/iros_challenge/onsite_competition/captures/rs_meta.json)'
    )
    parser.add_argument(
        '--server-host',
        type=str,
        default='localhost',
        help='Agent server host (default: localhost)'
    )
    parser.add_argument(
        '--server-port',
        type=int,
        default=8087,
        help='Agent server port (default: 8087)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='CUDA device (default: cuda:0)'
    )
    parser.add_argument(
        '--instruction',
        type=str,
        default='go to the red car',
        help='Navigation instruction (default: "go to the red car")'
    )

    args = parser.parse_args()

    # Run test
    success = test_internvla_n1(
        checkpoint_path=args.checkpoint,
        rs_meta_path=args.rs_meta,
        server_host=args.server_host,
        server_port=args.server_port,
        device=args.device,
        instruction=args.instruction
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
