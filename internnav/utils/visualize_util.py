import os
import time
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union, Any, List

import numpy as np

from internnav import PROJECT_ROOT_PATH
from .common_log_util import get_task_name

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

try:
    import imageio.v2 as imageio
    _IMAGEIO_AVAILABLE = True
except Exception:
    _IMAGEIO_AVAILABLE = False


viz_logger = logging.getLogger("visualize_util")
viz_logger.setLevel(logging.INFO)


@dataclass
class TrajectoryVizInfo:
    trajectory_id: str
    frames_dir: str
    video_path: str
    fps: int
    start_time: float
    end_time: Optional[float] = None
    frame_count: int = 0
    result: Optional[str] = None
    saved_frames: List[str] = None


class VisualizeUtil:
    """
    Save per-step observations as images and export a video at the end of each trajectory.

    If you already have saving functions, pass them in:
        save_frame_fn(image, out_path) and save_video_fn(frames_dir, out_path, fps)
    Otherwise, built-ins (PIL + imageio) are used.
    """
    def __init__(
        self,
        dataset_name: str,
        fps: int = 10,
        img_ext: str = "png",
        video_ext: str = "mp4",
        root_subdir: str = "viz",
        save_frame_fn: Optional[Callable[[Any, str], None]] = None,
        save_video_fn: Optional[Callable[[str, str, int], None]] = None,
    ):
        self.dataset_name = dataset_name
        self.fps = fps
        self.img_ext = img_ext
        self.video_ext = video_ext
        self.trajectories: Dict[str, TrajectoryVizInfo] = {}

        # Set up log directory and file handler (mirrors your progress logger style)
        base_dir = os.path.join(PROJECT_ROOT_PATH, "logs", get_task_name(), root_subdir)
        os.makedirs(base_dir, exist_ok=True)

        file_handler = logging.FileHandler(os.path.join(base_dir, f"{dataset_name}.log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        # Avoid adding duplicate handlers in repeated inits
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_handler.baseFilename
                   for h in viz_logger.handlers):
            viz_logger.addHandler(file_handler)

        self.base_dir = base_dir

        # Pluggable savers
        self._save_frame_fn = save_frame_fn or self._default_save_frame
        self._save_video_fn = save_video_fn or self._default_save_video

        # Metrics
        self._global_start: Optional[float] = None
        self._global_end: Optional[float] = None
        self._finished = 0

    # ---------- public API ----------

    def trace_start(self, trajectory_id: str):
        if self._global_start is None:
            self._global_start = time.time()
        traj_dir = os.path.join(self.base_dir, self.dataset_name, trajectory_id)
        frames_dir = os.path.join(traj_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        video_path = os.path.join(traj_dir, f"{trajectory_id}.{self.video_ext}")
        self.trajectories[trajectory_id] = TrajectoryVizInfo(
            trajectory_id=trajectory_id,
            frames_dir=frames_dir,
            video_path=video_path,
            fps=self.fps,
            start_time=time.time(),
            saved_frames=[]
        )
        viz_logger.info(f"[start] trajectory_id={trajectory_id}")

    def save_observation(
        self,
        trajectory_id: str,
        image: Union[np.ndarray, "Image.Image"],
        step_index: Optional[int] = None,
        filename: Optional[str] = None,
    ):
        """
        Save one frame. `image` can be a HxWxC numpy uint8 array (RGB/BGR) or PIL.Image.
        If step_index is given, used in filename ordering; otherwise uses running frame_count.
        """
        ti = self._require_traj(trajectory_id)

        if step_index is None:
            step_index = ti.frame_count

        # zero-padded name for lexicographic order
        fname = filename or f"{step_index:06d}.{self.img_ext}"
        out_path = os.path.join(ti.frames_dir, fname)
        self._save_frame_fn(image, out_path)

        ti.frame_count += 1
        if ti.saved_frames is not None:
            ti.saved_frames.append(out_path)

    def trace_end(self, trajectory_id: str, result: Optional[str] = None, assemble_video: bool = True):
        """
        Mark trajectory finished and (optionally) assemble video.
        """
        ti = self._require_traj(trajectory_id)
        ti.end_time = time.time()
        ti.result = result or ti.result
        self._finished += 1

        duration = round((ti.end_time - ti.start_time), 2)
        fps_eff = round((ti.frame_count / (duration + 1e-10)), 2)
        viz_logger.info(
            f"[finish] trajectory_id={trajectory_id} "
            f"[frames:{ti.frame_count}] [duration:{duration}s] [eff_fps:{fps_eff}] [result:{ti.result}]"
        )

        if assemble_video:
            self._save_video_fn(ti.frames_dir, ti.video_path, ti.fps)
            viz_logger.info(f"[video] saved {ti.video_path}")

    def report(self):
        """
        Summarize across trajectories.
        """
        self._global_end = time.time()
        if self._global_start is None:
            viz_logger.info("No trajectories recorded.")
            return

        duration = round((self._global_end - self._global_start), 2)
        total_frames = sum(t.frame_count for t in self.trajectories.values())
        fps = round((total_frames / (duration + 1e-10)), 2)

        result_map: Dict[str, int] = {}
        for t in self.trajectories.values():
            key = t.result or "unknown"
            result_map[key] = result_map.get(key, 0) + 1

        viz_logger.info(
            f"[report] dataset:{self.dataset_name} "
            f"[duration:{duration}s] [frames:{total_frames}] [avg_fps:{fps}] results:{result_map}"
        )

    # ---------- defaults & helpers ----------

    def _require_traj(self, trajectory_id: str) -> TrajectoryVizInfo:
        if trajectory_id not in self.trajectories:
            raise KeyError(f"trajectory_id not started: {trajectory_id}")
        return self.trajectories[trajectory_id]

    @staticmethod
    def _default_save_frame(image: Union[np.ndarray, "Image.Image"], out_path: str) -> None:
        if _PIL_AVAILABLE is False:
            raise RuntimeError("PIL not available and no custom save_frame_fn provided.")
        # Normalize input to PIL Image
        if isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                # grayscale -> RGB
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 3:
                img = Image.fromarray(arr)
            elif arr.shape[-1] == 4:
                img = Image.fromarray(arr, mode="RGBA")
            else:
                raise ValueError(f"Unsupported array shape for image saving: {arr.shape}")
        elif _PIL_AVAILABLE and hasattr(image, "save"):
            img = image
        else:
            raise TypeError("image must be a numpy array or PIL.Image")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)

    @staticmethod
    def _default_save_video(frames_dir: str, out_path: str, fps: int) -> None:
        if _IMAGEIO_AVAILABLE is False:
            raise RuntimeError("imageio not available and no custom save_video_fn provided.")

        # gather frames in lexicographic order
        names = [n for n in os.listdir(frames_dir) if n.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not names:
            raise RuntimeError(f"No frames found in {frames_dir}")
        names.sort()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with imageio.get_writer(out_path, mode="I", fps=fps) as writer:
            for name in names:
                frame_path = os.path.join(frames_dir, name)
                frame = imageio.imread(frame_path)
                writer.append_data(frame)
