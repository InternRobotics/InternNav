import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time
    start_time = time.time()

    image_file = request.files['image']
    depth_file = request.files['depth']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)
    depth = depth.astype(np.float32) / 10000.0
    print(f"read http data cost {time.time() - start_time}")

    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    instruction = "Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor"
    policy_init = data['reset']
    if policy_init:
        start_time = time.time()
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        print("init reset model!!!")
        agent.reset()

    idx += 1

    look_down = False
    t0 = time.time()
    dual_sys_output = {}

    dual_sys_output = agent.step(
        image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
    )
    if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
        look_down = True
        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
        )

    json_output = {}
    if dual_sys_output.output_action is not None:
        json_output['discrete_action'] = dual_sys_output.output_action
    if dual_sys_output.output_pixel is not None:
        json_output['pixel_goal'] = dual_sys_output.output_pixel

    t1 = time.time()
    generate_time = t1 - t0
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    return jsonify(json_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="/path/to/InternVLA-N1")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    args = parser.parse_args()

    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    agent = InternVLAN1AsyncAgent(args)
    agent.reset()

    app.run(host='0.0.0.0', port=5801)
