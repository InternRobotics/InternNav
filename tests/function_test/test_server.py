"""
Test if the server starts successfully and is still alive after sleep.
"""
import subprocess
import sys
import time


def start_server():
    server_cmd = [
        sys.executable,
        "scripts/eval/start_server.py",
    ]

    proc = subprocess.Popen(
        server_cmd,
        stdout=None,
        stderr=None,
        start_new_session=True,
    )
    return proc


def stop_server(proc):
    if proc and proc.poll() is None:
        print("Shutting down server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError("❌ Server failed to shut down within 10 seconds.")


if __name__ == '__main__':
    try:
        proc = start_server()
        time.sleep(5)

        # Raise if process exited
        if proc.poll() is not None:
            raise RuntimeError(f"❌ Server exited too early with code {proc.returncode}")
        print("✅ Server is still alive after 5 seconds.")

        stop_server(proc)

    except Exception as e:
        print(f'exception is {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)
