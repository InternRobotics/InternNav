# aligned_realsense.py
import time
from typing import Dict, Optional, Tuple

import cv2  # 仅用于极端兜底 resize，可去掉
import numpy as np
import pyrealsense2 as rs


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

        # 开启彩色和深度流
        cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cfps)
        cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)

        profile = self.pipeline.start(cfg)

        # 深度缩放（将 z16 转米）
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        # 对齐到彩色
        self.align = rs.align(rs.stream.color)

        # 预热
        for _ in range(self.warmup_frames):
            self.pipeline.wait_for_frames()

        # 做一次对齐检查，确保后续帧尺寸一致
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        assert color and depth, "预热对齐失败：未获得 color/depth 帧"
        rgb = np.asanyarray(color.get_data())
        depth_raw = np.asanyarray(depth.get_data())
        if depth_raw.shape != rgb.shape[:2]:
            # 理论上不应发生；发生时先兜底到相同尺寸（注意：仅尺寸匹配，几何可能不严格）
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
            raise RuntimeError("未获得对齐后的 color/depth 帧")

        rgb = np.asanyarray(color.get_data())  # HxWx3, uint8 (BGR)
        depth_raw = np.asanyarray(depth.get_data())  # HxW, uint16
        if depth_raw.shape != rgb.shape[:2]:
            # 极端兜底（理论上 align 后应一致）
            depth_raw = cv2.resize(depth_raw, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        depth_m = depth_raw.astype(np.float32) * float(self.depth_scale)
        ts_ms = color.get_timestamp() or frames.get_timestamp()
        ts_s = float(ts_ms) / 1000.0 if ts_ms is not None else time.time()

        return {"rgb": rgb, "depth": depth_m, "timestamp_s": ts_s}


# --- 最小示例 ---
if __name__ == "__main__":
    with AlignedRealSense(serial_no=None) as cam:
        obs = cam.get_observation()
        print("RGB:", obs["rgb"].shape, obs["rgb"].dtype)
        print("Depth:", obs["depth"].shape, obs["depth"].dtype, "(meters)")
