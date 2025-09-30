# save_obs.py
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def save_obs(
    obs: dict,
    outdir: str = "./captures",
    prefix: str = None,
    max_depth_m: float = 3.0,  # 伪彩可视化的裁剪上限
    save_rgb: bool = True,
    save_depth_16bit: bool = True,
    save_depth_vis: bool = True,
):
    """
    将 obs = {"rgb": HxWx3 uint8 (BGR), "depth": HxW float32 (m), "timestamp_s": float, "intrinsics": {...}}
    保存到磁盘。
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    ts = float(obs.get("timestamp_s", time.time()))
    stamp = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S_%f")
    prefix = prefix or f"{stamp}"

    rgb = obs.get("rgb", None)
    depth_m = obs.get("depth", None)

    # 1) 保存 RGB（BGR 排列，cv2 直接写）
    rgb_path = None
    if save_rgb and rgb is not None:
        rgb_path = os.path.join(outdir, f"{prefix}_rgb.jpg")
        cv2.imwrite(rgb_path, rgb)

    # 2) 保存 16-bit 深度（单位毫米）
    depth16_path = None
    vis_path = None
    if depth_m is not None and (save_depth_16bit or save_depth_vis):
        d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)  # 清 NaN/Inf
        if save_depth_16bit:
            depth_mm = np.clip(np.round(d * 1000.0), 0, 65535).astype(np.uint16)
            depth16_path = os.path.join(outdir, f"{prefix}_depth_mm.png")
            cv2.imwrite(depth16_path, depth_mm)

        # 3) 保存伪彩（仅用于查看）
        if save_depth_vis:
            d_clip = np.clip(d, 0.0, max_depth_m)
            # 近处亮一些：先归一化到 0~255，再取反做色图
            d_norm = (d_clip / max_depth_m * 255.0).astype(np.uint8)
            depth_color = cv2.applyColorMap(255 - d_norm, cv2.COLORMAP_JET)
            vis_path = os.path.join(outdir, f"{prefix}_depth_vis.png")
            cv2.imwrite(vis_path, depth_color)

    # 4) 元信息
    meta = {
        "timestamp_s": ts,
        "paths": {
            "rgb": rgb_path,
            "depth_mm": depth16_path,
            "depth_vis": vis_path,
        },
        "intrinsics": obs.get("intrinsics", {}),
        "notes": "depth_mm.png 是以毫米存储的 16-bit PNG；depth_vis.png 仅用于可视化。",
    }
    with open(os.path.join(outdir, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta
