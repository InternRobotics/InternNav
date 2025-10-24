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
    max_depth_m: float = 3.0,
    save_rgb: bool = True,
    save_depth_16bit: bool = True,
    save_depth_vis: bool = True,
):
    """
    save obs = {"rgb": HxWx3 uint8 (BGR), "depth": HxW float32 (m), "timestamp_s": float, "intrinsics": {...}}
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    ts = float(obs.get("timestamp_s", time.time()))
    stamp = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S_%f")
    prefix = prefix or f"{stamp}"

    rgb = obs.get("rgb", None)
    depth_m = obs.get("depth", None)

    # 1) save RGB（BGR，cv2）
    rgb_path = None
    if save_rgb and rgb is not None:
        rgb_path = os.path.join(outdir, f"{prefix}_rgb.jpg")
        cv2.imwrite(rgb_path, rgb)

    # 2) save 16-bit depth（unit: mm）
    depth16_path = None
    vis_path = None
    if depth_m is not None and (save_depth_16bit or save_depth_vis):
        d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)  # 清 NaN/Inf
        if save_depth_16bit:
            depth_mm = np.clip(np.round(d * 1000.0), 0, 65535).astype(np.uint16)
            depth16_path = os.path.join(outdir, f"{prefix}_depth_mm.png")
            cv2.imwrite(depth16_path, depth_mm)

        # 3) save depth vis
        if save_depth_vis:
            d_clip = np.clip(d, 0.0, max_depth_m)
            # Brighten the near field: first normalize to 0–255, then invert and apply a colormap.
            d_norm = (d_clip / max_depth_m * 255.0).astype(np.uint8)
            depth_color = cv2.applyColorMap(255 - d_norm, cv2.COLORMAP_JET)
            vis_path = os.path.join(outdir, f"{prefix}_depth_vis.png")
            cv2.imwrite(vis_path, depth_color)

    # 4) meta info
    meta = {
        "timestamp_s": ts,
        "paths": {
            "rgb": rgb_path,
            "depth_mm": depth16_path,
            "depth_vis": vis_path,
        },
        "intrinsics": obs.get("intrinsics", {}),
        "notes": "depth_mm.png 是以毫米存储的 16-bit PNG, depth_vis.png 仅用于可视化。",
    }
    with open(os.path.join(outdir, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# load_obs.py
from glob import glob
from typing import Dict, List, Optional, Tuple


def _resolve(base: str, p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(base, p))


def load_obs_from_meta(meta_path: str, nan_for_zeros: bool = False) -> Dict:
    """
    读取由 save_obs() 生成的 meta.json, 并还原为 obs dict:
      {
        "rgb": uint8[H,W,3]  (BGR),
        "depth": float32[H,W] (meters) 或 None,
        "timestamp_s": float,
        "intrinsics": dict
      }

    Args:
        meta_path: *_meta.json 路径
        nan_for_zeros: 若为 True，则把深度为 0 的像素转为 NaN（便于下游遮挡处理）
    """
    meta_path = os.path.abspath(meta_path)
    base = os.path.dirname(os.path.dirname(meta_path))
    with open(meta_path, "r") as f:
        meta = json.load(f)

    paths = meta.get("paths", {})
    rgb_path = _resolve(base, paths.get("rgb"))
    depth_mm_path = _resolve(base, paths.get("depth_mm"))

    # 读 RGB（保存时就是 BGR，OpenCV 读回来仍是 BGR）
    rgb = None
    if rgb_path and os.path.exists(rgb_path):
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)  # uint8, HxWx3 (BGR)

    # 读深度（16-bit 毫米 PNG → 米）
    depth = None
    if depth_mm_path and os.path.exists(depth_mm_path):
        depth_mm = cv2.imread(depth_mm_path, cv2.IMREAD_UNCHANGED)  # uint16
        if depth_mm is None:
            raise RuntimeError(f"Failed to read depth image: {depth_mm_path}")
        if depth_mm.dtype != np.uint16:
            raise ValueError(f"Depth image must be uint16 (mm). Got {depth_mm.dtype}")
        depth = depth_mm.astype(np.float32) / 1000.0  # meters
        if nan_for_zeros:
            depth[depth_mm == 0] = np.nan

    # 尺寸一致性检查（若两者都有）
    if rgb is not None and depth is not None and depth.shape != rgb.shape[:2]:
        raise ValueError(f"Shape mismatch: rgb {rgb.shape[:2]} vs depth {depth.shape}. " "确保保存前已对齐（align 到 color）。")

    obs = {
        "rgb": rgb,
        "depth": depth,
        "timestamp_s": float(meta.get("timestamp_s", 0.0)),
        "intrinsics": meta.get("intrinsics", {}),
    }
    return obs


def load_all_obs_in_dir(captures_dir: str, pattern: str = "*_meta.json", sort: bool = True) -> List[Tuple[str, Dict]]:
    """
    批量读取目录下所有 meta，返回 [(meta_path, obs), ...]
    """
    metas = glob(os.path.join(captures_dir, pattern))
    if sort:
        metas.sort()
    result = []
    for m in metas:
        try:
            result.append((m, load_obs_from_meta(m)))
        except Exception as e:
            print(f"[warn] skip {m}: {e}")
    return result
