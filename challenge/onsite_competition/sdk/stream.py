# stream_server.py
import time

import cv2
from flask import Flask, Response

app = Flask(__name__)

# 由主程序注入
_env = None


def set_env(env):
    """set env from main to stream server"""
    global _env
    _env = env


def _mjpeg_generator(jpeg_quality: int = 80):
    boundary = b"--frame"
    while True:
        if _env is None:
            time.sleep(0.1)
            continue
        obs = _env.get_observation()
        if obs is None:
            time.sleep(0.01)
            continue
        frame_bgr = obs["rgb"]
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            continue
        yield (boundary + b"\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


@app.route("/stream")
def stream():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")
