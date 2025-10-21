# aligned_realsense.py
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from save_obs import save_obs


class AlignedRealSense:
    def __init__(
        self,
        serial_no: Optional[str] = None,
        color_res: Tuple[int, int, int] = (640, 480, 30),  # (w,h,fps)
        depth_res: Tuple[int, int, int] = (640, 480, 30),
        warmup_frames: int = 15,
    ):
        self.serial_no = serial_no
        self.color_res = color_res
        self.depth_res = depth_res
        self.warmup_frames = warmup_frames

        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self.depth_scale: Optional[float] = None
        self.started = False

    def start(self):
        if self.started:
            return
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if self.serial_no:
            cfg.enable_device(self.serial_no)

        cw, ch, cfps = self.color_res
        dw, dh, dfps = self.depth_res

        # open stream for color and depth
        cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cfps)
        cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)

        profile = self.pipeline.start(cfg)

        # 深度缩放（将 z16 转米）
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        # align to color
        self.align = rs.align(rs.stream.color)

        # warm up
        for _ in range(self.warmup_frames):
            self.pipeline.wait_for_frames()

        # align check
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        assert color and depth, "warm up align failed"
        rgb = np.asanyarray(color.get_data())
        depth_raw = np.asanyarray(depth.get_data())
        if depth_raw.shape != rgb.shape[:2]:
            depth_raw = cv2.resize(depth_raw, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.started = True

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
        self.pipeline = None
        self.started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, et, ev, tb):
        self.stop()

    def get_observation(self, timeout_ms: int = 1000) -> Dict:
        """
        Returns:
            {
              "rgb": uint8[H,W,3] (BGR),
              "depth": float32[H,W] (meters),
              "timestamp_s": float
            }
        """
        if not self.started:
            self.start()

        frames = self.pipeline.wait_for_frames(timeout_ms)
        frames = self.align.process(frames)

        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("can not align color/depth frame")

        bgr = np.asanyarray(color.get_data())  # HxWx3, uint8 (BGR)
        rgb = bgr[..., ::-1]  # HxWx3, uint8 (convert to RGB)
        depth_raw = np.asanyarray(depth.get_data())  # HxW, uint16
        if depth_raw.shape != rgb.shape[:2]:
            # Extreme fallback (theoretically should be consistent after alignment).
            depth_raw = cv2.resize(depth_raw, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        depth_m = depth_raw.astype(np.float32) * float(self.depth_scale)
        ts_ms = color.get_timestamp() or frames.get_timestamp()
        ts_s = float(ts_ms) / 1000.0 if ts_ms is not None else time.time()

        return {"rgb": rgb, "depth": depth_m, "timestamp_s": ts_s}


if __name__ == "__main__":
    with AlignedRealSense(serial_no=None) as cam:
        obs = cam.get_observation()
        print("RGB:", obs["rgb"].shape, obs["rgb"].dtype)
        print("Depth:", obs["depth"].shape, obs["depth"].dtype, "(meters)")
        meta = save_obs(obs, outdir="./captures", prefix="rs")
        print("Saved:", meta)
