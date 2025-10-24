import argparse
import base64
import json
import os
import sys
import uuid

# from utils.log_util import logger
from enum import Enum
from typing import Dict, Optional

import numpy as np
import torch
import uvicorn
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, status
from pydantic import BaseModel
from transformers import AutoProcessor

from internnav.env.utils.habitat_extensions.evaluator_single import VLNEvaluator
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.utils.dist import *

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT_PATH)
print(f"PROJECT_ROOT_PATH {PROJECT_ROOT_PATH}", flush=True)


global instruction


class VideoRequest(BaseModel):
    """
    Frontend post json object template
    """

    user: str
    task: str
    job_id: str
    data: str

    """
    data json template
        Manipulation:
        {
            model_type: str,
            instruction: str,
            scene_type: str,
            max_step: str
        }

        Navigation:
        {
            model_type: str,
            instruction: str,
            episode_type: str
        }
    """


class TaskInfo:
    def __init__(self, task_id, status, result_path):
        self.task_id: str = task_id
        self.status: str = status
        self.result_path: str = result_path


class TaskStatus(str, Enum):
    pending = ('pending',)
    completed = 'completed'
    failed = 'failed'
    terminated = 'terminated'


class BackendServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.app = FastAPI(title='Backend Service')
        self._router = APIRouter(prefix='/predict')
        self._register_routes()
        self.app.include_router(self._router)

        self.GPU_COUNT = torch.cuda.device_count()
        self.tasks: Dict[str, TaskInfo] = {}
        self.MAX_TASK_LIMIT = 48

    def _register_routes(self):
        route_config = [
            ('/video', self.predict, ['POST'], None),
            ('/task/{task_id}', self.get_task_status, ['GET'], Dict[str, Optional[str]]),
        ]

        for path, handler, methods, response_model in route_config:
            self._router.add_api_route(
                path=path,
                endpoint=handler,
                methods=methods,
                response_model=response_model if 'GET' in methods else None,
                status_code=status.HTTP_200_OK if 'GET' in methods else None,
            )

    async def predict(self, request: VideoRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
        # Safety: allow tasks pending to MAX_TASK_LIMIT
        # TODO: may need to improve
        if sum(task.status == "pending" for task in self.tasks.values()) >= self.MAX_TASK_LIMIT:
            print(f"Failed to START Task: reach to limit")
            raise HTTPException(status_code=429, detail=f"Failed to start new task: reach to limit")

        task_id = str(uuid.uuid4())
        path = os.path.join(output_path, task_id)
        print(f"Create new task: ID={task_id}, output path={path}")
        self.tasks[task_id] = TaskInfo(task_id=task_id, status="pending", result_path=path)

        background_tasks.add_task(self._submit_task, task_id, request.data, path)

        print(f"Start Task: {task_id} for user: {request.user}, task: {request.task}")

        return {"task_id": task_id, "status": "pending"}

    def _submit_task(self, task_id: str, data: str, path: str):

        print(f"process task: ID={task_id}")
        print(f"receive data: {data}...")  # 只打印前100个字符
        try:
            data_dict = json.loads(data)
            if data_dict.get("task_type") == "vln_eval":
                print("=======VLN Eval Task=======")
                cache_dir = f"/tmp/InternNav/.triton"
                os.makedirs(cache_dir, exist_ok=True)
                os.chmod(cache_dir, 0o777)

                evaluator.infer_scene_id = int(data_dict["scene_index"]) - 1
                evaluator.infer_episode_id = int(data_dict["episode_index"]) - 1
                evaluator.infer_instruction = data_dict["instruction"]
                evaluator.output_path = path
                evaluator.infer_data_ready = True
                evaluator.run_single_eval()

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            self.tasks[task_id].status = "failed"
            print(f"Task {task_id} failed: {e}")

    async def get_task_status(self, task_id: str) -> Dict[str, Optional[str]]:
        print(f"call get_task_status")
        task = self.tasks[task_id]
        if not evaluator.infer_success:
            return {"status": "pending", "result": task.result_path}

        video_path = os.path.join(task.result_path, f"res_{evaluator.infer_success_cnt}.mp4")
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
            video_data = base64.b64encode(video_bytes).decode("utf-8")

        return {"status": 'completed', "result": task.result_path, "video": video_data}

    def run(self):
        uvicorn.run('__main__:server.app', host=self.host, port=self.port)


if __name__ == "__main__":
    output_path = f"log/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1")
    parser.add_argument("--habitat_config_path", type=str, default='scripts/eval/configs/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='train')
    parser.add_argument("--output_path", type=str, default='./exps_pix/val_unseen/debug_coord_wm')
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--predict_step_nums", type=int, default=32)
    parser.add_argument("--continuous_traj", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--gpu', default=0, type=int, help='gpu')
    parser.add_argument('--port', default='2443')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    args = parser.parse_args()
    init_distributed_mode(args)
    local_rank = args.local_rank
    np.random.seed(local_rank)

    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = 'left'

    device = torch.device(f"cuda:{local_rank}")
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": device}
    )
    model.eval()
    world_size = get_world_size()
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        processor=processor,
        epoch=0,
        args=args,
    )
    # evaluator.eval_action(0)

    server = BackendServer(host="0.0.0.0", port=8001)
    server.run()
