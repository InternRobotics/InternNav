import base64
import json

# from utils.log_util import logger
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Dict, Optional

import torch
import uvicorn
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, status
from pydantic import BaseModel

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT_PATH)
print(f"PROJECT_ROOT_PATH {PROJECT_ROOT_PATH}", flush=True)

try:
    import ray
    from ray.exceptions import GetTimeoutError

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                }
            },
            # num_cpus=config.ray_init.num_cpus,
        )
    RAY_AVAILABLE = True
    logging.getLogger("ray").setLevel(logging.INFO)
except Exception as e:
    RAY_AVAILABLE = False
    print(f"Ray not available: {str(e)}")

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
        self.ray_future = None


class TaskStatus(str, Enum):
    pending = ('pending',)
    completed = 'completed'
    failed = 'failed'
    terminated = 'terminated'


if RAY_AVAILABLE:

    @ray.remote(num_gpus=1)
    def run_inference(cmd: list, cwd: Optional[str] = None, env: Optional[Dict] = None) -> str:
        # 合并环境变量
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        start = time.time()
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=72000,  # 延长超时时间（20小时）
        )
        duration = time.time() - start
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return f"[DONE in {duration:.2f}s]\n{result.stdout}"


class BackendServer:
    """
    A FastAPI-based backend service for dispatching model inference jobs to available GPUs.

    This server exposes a RESTful API at the route:
        POST {host}:{port}/predict/video

    Key Features:
    - Manages GPU resource allocation with per-GPU locking to prevent conflicts.
    - Launches subprocesses (e.g., shell scripts or model inference tasks) isolated per GPU using CUDA_VISIBLE_DEVICES.
    - Supports concurrent requests via a thread pool executor.
    - Automatically retries if all GPUs are busy, ensuring no job is dropped.

    Methods:
    --------
    __init__(host: str, port: int)
        Initializes the FastAPI app, GPU locks, and thread executor.

    _register_routes()
        Registers API routes with the FastAPI app.

    async predict(request: VideoRequest)
        Asynchronous entrypoint for the /predict/video route.
        Delegates execution to sync_gpu_predict in a background thread.

    sync_gpu_predict(data: str) -> str
        Synchronously attempts to acquire an available GPU and runs a subprocess
        with the given data. Retries until a GPU becomes available.

    run()
        Starts the FastAPI server using uvicorn.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.app = FastAPI(title='Backend Service')
        self._router = APIRouter(prefix='/predict')
        self._register_routes()
        self.app.include_router(self._router)

        self.GPU_COUNT = torch.cuda.device_count()
        self.gpu_locks = [threading.Lock() for _ in range(self.GPU_COUNT)]
        self.executor = ThreadPoolExecutor(max_workers=self.GPU_COUNT)

        self.tasks: Dict[str, TaskInfo] = {}
        self.MAX_TASK_LIMIT = 48

    def _register_routes(self):
        route_config = [
            ('/video', self.predict, ['POST'], None),
            ('/terminate/{task_id}', self.terminate_task, ['POST'], None),
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
            print("Failed to START Task: reach to limit")
            raise HTTPException(status_code=429, detail="Failed to start new task: reach to limit")

        task_id = str(uuid.uuid4())
        path = os.path.join(output_path, task_id)
        print(f"Create new task: ID={task_id}, output path={path}")
        self.tasks[task_id] = TaskInfo(task_id=task_id, status="pending", result_path=path)

        background_tasks.add_task(self._submit_task, task_id, request.data, path)

        print(f"Start Task: {task_id} for user: {request.user}, task: {request.task}")
        print('Available GPUs: ' + str(ray.available_resources().get('GPU', 0)))
        print('Used GPUs: ' + str(ray.cluster_resources().get('GPU', 0) - ray.available_resources().get('GPU', 0)))

        return {"task_id": task_id, "status": "pending"}

    def _submit_task(self, task_id: str, data: str, path: str):

        print(f"process task: ID={task_id}")
        print(f"receive data: {data[:100]}...")  # 只打印前100个字符
        try:
            data_dict = json.loads(data)
            if data_dict.get("task_type") == "vln_eval":
                print("=======VLN Eval Task=======")
                cache_dir = "/tmp/InternNav/.triton"
                os.makedirs(cache_dir, exist_ok=True)
                os.chmod(cache_dir, 0o777)

                env = os.environ.copy()
                env.update(
                    {
                        "MAGNUM_LOG": "quiet",
                        "HABITAT_SIM_LOG": "quiet",
                        "NCCL_SOCKET_IFNAME": "bond0",
                        "NCCL_IB_HCA": "mlx5_2,mlx5_3,mlx5_4,mlx5_5",
                        "TRITON_CACHE_DIR": cache_dir,
                    }
                )
                model_path = "checkpoints/InternVLA-N1/"
                cmd = [
                    "python",
                    "-u",
                    "internnav/habitat_extensions/evaluator_single.py",
                    "--model_path",
                    model_path,
                    "--predict_step_nums",
                    "32",
                    "--continuous_traj",
                    "--output_path",
                    path,
                    "--instruction",
                    data_dict["instruction"],
                ]

                cwd = PROJECT_ROOT_PATH
                if RAY_AVAILABLE:
                    future = run_inference.remote(cmd, cwd, env)
                    self.tasks[task_id].ray_future = future

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            self.tasks[task_id].status = "failed"
            print(f"Task {task_id} failed: {e}")

    async def get_task_status(self, task_id: str) -> Dict[str, Optional[str]]:
        print("call get_task_status")
        if task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        task = self.tasks[task_id]
        if task.status in ["completed", "terminated", "failed"]:

            video_path = os.path.join(task.result_path, "res.mp4")
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
                video_data = base64.b64encode(video_bytes).decode("utf-8")

            return {"status": task.status, "result": task.result_path, "video": video_data}

        if RAY_AVAILABLE and task.ray_future:
            try:
                result = ray.get(task.ray_future, timeout=0.2)
                task.status = "completed"
                print('task finish!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                video_path = os.path.join(task.result_path, "res.mp4")
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()
                    video_data = base64.b64encode(video_bytes).decode("utf-8")
                print(f"Task [{task_id}]: {result}")
                return {"status": "completed", "result": task.result_path, "video": video_data}
            except GetTimeoutError:
                return {"status": "pending", "result": task.result_path}
            except Exception as e:
                import traceback

                print(traceback.format_exc())
                task.status = "failed"
                return {"status": "failed", "result": str(e)}
        else:
            return {"status": task.status, "result": task.result_path}

    async def terminate_task(self, task_id: str) -> Dict[str, str]:
        if task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        task = self.tasks[task_id]
        try:
            if RAY_AVAILABLE and task.ray_future:
                ray.cancel(task.ray_future, force=True)
                task.status = "terminated"
            else:
                if hasattr(task, "job") and task.job is not None:
                    task.job.terminate()
                    task.status = "terminated"
            return {"status": "success", "message": f"Task {task_id} terminated"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Termination failed: {e}")

    def run(self):
        uvicorn.run('__main__:server.app', host=self.host, port=self.port)


if __name__ == "__main__":
    output_path = "log/"
    print(torch.cuda.device_count())
    server = BackendServer(host="0.0.0.0", port=8001)
    server.run()
