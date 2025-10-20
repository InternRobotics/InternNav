# stream_server.py
import threading
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, make_response, stream_with_context

app = Flask(__name__)

_env = None
_stop = threading.Event()  # graceful stop for Ctrl+C


def set_env(env):
    """set env from main to stream server"""
    global _env
    _env = env


_instruction = ""
_instruction_lock = threading.Lock()


def set_instruction(text: str) -> None:
    """
    Set the instruction text that will be displayed on the viewer page.
    Thread-safe; can be called from your main/control loop at any time.
    """
    global _instruction
    with _instruction_lock:
        _instruction = str(text) if text is not None else ""


def get_instruction() -> str:
    """
    Get the current instruction text (thread-safe).
    """
    with _instruction_lock:
        return _instruction


def _encode_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> Optional[bytes]:
    ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpg.tobytes() if ok else None


def _mjpeg_generator(jpeg_quality: int = 80, fps_limit: float = 30.0):
    boundary = b"--frame"
    min_interval = 1.0 / fps_limit if fps_limit > 0 else 0.0

    last = 0.0
    while not _stop.is_set():
        if _env is None:
            time.sleep(0.05)
            continue

        obs = _env.get_observation()
        if not obs or "rgb" not in obs:
            time.sleep(0.01)
            continue

        frame = obs["rgb"]  # Expect BGR for OpenCV; convert if your source is RGB
        if frame is None:
            time.sleep(0.01)
            continue

        # If your env returns RGB, uncomment the next line:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        jpg = _encode_jpeg(frame, quality=jpeg_quality)
        if jpg is None:
            continue

        # multipart chunk
        yield (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Pragma: no-cache\r\n\r\n" + jpg + b"\r\n"
        )

        # simple pacing (prevents 100% CPU + helps Ctrl+C responsiveness)
        now = time.time()
        sleep_t = min_interval - (now - last)
        if sleep_t > 0:
            time.sleep(sleep_t)
        last = now


@app.route("/stream")
def stream():
    # stream_with_context ensures Flask doesn’t buffer the generator
    gen = stream_with_context(_mjpeg_generator())
    resp = Response(gen, mimetype="multipart/x-mixed-replace; boundary=frame", direct_passthrough=True)
    # extra headers to avoid caching
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Connection"] = "keep-alive"
    return resp


@app.route("/instruction")
def instruction():
    text = get_instruction()
    r = make_response(text)
    r.headers["Content-Type"] = "text/plain; charset=utf-8"
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    return r


@app.route("/")
def index():
    # simple viewer page with instruction overlay
    html = """
    <!doctype html>
    <title>MJPEG Stream</title>
    <meta charset="utf-8" />
    <style>
      :root { color-scheme: dark; }
      body {
        margin:0; background:#111; height:100vh; overflow:hidden;
        display:grid; place-items:center; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Apple Color Emoji, Segoe UI Emoji;
      }
      .wrap { position:relative; width:min(100vw, 100vh*16/9); height:min(100vh, 100vw*9/16); }
      .wrap img { width:100%; height:100%; object-fit:contain; display:block; background:#000; }
      #instr {
        position:absolute; left:1rem; right:1rem; bottom:1rem;
        background:rgba(0,0,0,0.55); backdrop-filter: blur(4px);
        border:1px solid rgba(255,255,255,0.15);
        border-radius:12px; padding:0.75rem 1rem;
        color:#eaeaea; line-height:1.4; white-space:pre-wrap; word-break:break-word;
        font-size:clamp(14px, 1.8vw, 18px);
      }
      #instr .label { opacity:0.7; font-size:0.9em; margin-right:0.5rem; }
    </style>
    <div class="wrap">
      <img src="/stream" alt="stream"/>
      <div id="instr"><span class="label">Instruction:</span><span id="instr-text">(none)</span></div>
    </div>
    <script>
      async function refreshInstruction() {
        try {
          const res = await fetch("/instruction", { cache: "no-store" });
          if (!res.ok) return;
          const text = await res.text();
          // Use textContent to avoid injecting HTML
          document.getElementById("instr-text").textContent = text || "(none)";
        } catch (e) {
          // ignore fetch errors (server might be restarting)
        }
      }
      // Initial fetch and periodic refresh
      refreshInstruction();
      setInterval(refreshInstruction, 500); // update 2x/sec; adjust if you want slower
      // Also refresh when page becomes visible again
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible") refreshInstruction();
      });
    </script>
    """
    r = make_response(html)
    r.headers["Cache-Control"] = "no-cache"
    return r


def _handle_sigint(sig, frame):
    _stop.set()


# start the HTTP server in a background thread (non-blocking)
def run_server(env, host, port):
    # make env accessible to the server
    set_env(env)
    # IMPORTANT: disable reloader so it doesn't spawn another process
    app.run(host="0.0.0.0", port=8080, threaded=True, use_reloader=False)


def run(host="0.0.0.0", port=8080, env=None):
    server_thread = threading.Thread(target=run_server, args=(env, host, port), daemon=True)
    server_thread.start()
    time.sleep(0.3)  # tiny delay to let the server bind the port
    print("--- stream app is running on http://0.0.0.0:8080 ---")


if __name__ == "__main__":
    # Example: plug your env here before run()
    # set_env(MyEnv())
    run()
