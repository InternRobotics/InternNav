import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import gradio as gr
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")  # fastapi server
API_ENDPOINTS = {
    "submit_task": f"{BACKEND_URL}/predict/video",
    "query_status": f"{BACKEND_URL}/predict/task",
    "get_result": f"{BACKEND_URL}/predict",
}


SCENE_CONFIGS = {
    "scene_1": {
        "description": "Modern Apartment",
        "name": "17DRP5sb8fy",
        "glb_path": "scene_assets/scene1_no_ceiling.glb",  # PLY file path
    },
    "scene_2": {
        "description": "Office Building",
        "name": "r1Q1Z4BcV1o",
        "glb_path": "scene_assets/scene2_no_ceiling.glb",
    },
    "scene_3": {
        "description": "University Campus",
        "name": "dhjEzFoUFzH",
        "glb_path": "scene_assets/scene3_no_ceiling.glb",
    },
}

EPISODE_CONFIGS = {
    "episode_1": {
        "description": "1",
    },
    "episode_2": {
        "description": "2",
    },
    "episode_3": {
        "description": "3",
    },
    "episode_4": {
        "description": "4",
    },
}

MODEL_CHOICES = []


###############################################################################

SESSION_TASKS = {}
IP_REQUEST_RECORDS = defaultdict(list)
IP_LIMIT = 5


def is_request_allowed(ip: str) -> bool:
    now = datetime.now()
    IP_REQUEST_RECORDS[ip] = [t for t in IP_REQUEST_RECORDS[ip] if now - t < timedelta(minutes=1)]
    if len(IP_REQUEST_RECORDS[ip]) < IP_LIMIT:
        IP_REQUEST_RECORDS[ip].append(now)
        return True
    return False


###############################################################################


# Log directory path
LOG_DIR = "~/logs"
os.makedirs(LOG_DIR, exist_ok=True)
ACCESS_LOG = os.path.join(LOG_DIR, "access.log")
SUBMISSION_LOG = os.path.join(LOG_DIR, "submissions.log")


def log_access(user_ip: str = None, user_agent: str = None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "type": "access",
        "user_ip": user_ip or "unknown",
        "user_agent": user_agent or "unknown",
    }

    with open(ACCESS_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def log_submission(scene: str, prompt: str, model: str, user: str = "anonymous", res: str = "unknown"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "type": "submission",
        "user": user,
        "scene": scene,
        "prompt": prompt,
        "model": model,
        # "max_step": str(max_step),
        "res": res,
    }

    with open(SUBMISSION_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def read_logs(log_type: str = "all", max_entries: int = 50) -> list:
    logs = []

    if log_type in ["all", "access"]:
        try:
            with open(ACCESS_LOG, "r") as f:
                for line in f:
                    logs.append(json.loads(line.strip()))
        except FileNotFoundError:
            pass

    if log_type in ["all", "submission"]:
        try:
            with open(SUBMISSION_LOG, "r") as f:
                for line in f:
                    logs.append(json.loads(line.strip()))
        except FileNotFoundError:
            pass

    # Sorted by timestamp
    logs.sort(key=lambda x: x["timestamp"], reverse=True)
    return logs[:max_entries]


def format_logs_for_display(logs: list) -> str:
    if not logs:
        return "No log record"

    markdown = "### System Access Log\n\n"
    markdown += "| Time | Type | User/IP | Details |\n"
    markdown += "|------|------|---------|----------|\n"

    for log in logs:
        timestamp = log.get("timestamp", "unknown")
        log_type = "Access" if log.get("type") == "access" else "Submission"

        if log_type == "Access":
            user = log.get("user_ip", "unknown")
            details = f"User-Agent: {log.get('user_agent', 'unknown')}"
        else:
            user = log.get("user", "anonymous")
            result = log.get('res', 'unknown')
            if result != "success":
                if len(result) > 40:  # Adjust this threshold as needed
                    result = f"{result[:20]}...{result[-20:]}"
            details = f"Scene: {log.get('scene', 'unknown')}, Prompt: {log.get('prompt', '')}, Model: {log.get('model', 'unknown')}, result: {result}"

        markdown += f"| {timestamp} | {log_type} | {user} | {details} |\n"

    return markdown


def submit_to_backend(
    scene: str,
    prompt: str,
    episode: str,
    user: str = "Gradio-user",
) -> dict:
    job_id = str(uuid.uuid4())

    scene_index = scene.split("_")[-1]
    episode_index = episode.split("_")[-1]

    data = {
        "task_type": "vln_eval",  # Identify task type
        "instruction": prompt,
        "scene_index": scene_index,
        "episode_index": episode_index,
    }

    payload = {"user": user, "task": "robot_navigation", "job_id": job_id, "data": json.dumps(data)}

    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_ENDPOINTS["submit_task"], json=payload, headers=headers, timeout=600)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_task_status(task_id: str) -> dict:
    try:
        response = requests.get(f"{API_ENDPOINTS['query_status']}/{task_id}", timeout=600)
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"status": "error", "message": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_task_result(task_id: str) -> Optional[dict]:
    try:
        response = requests.get(f"{API_ENDPOINTS['get_result']}/{task_id}", timeout=5)
        return response.json()
    except Exception as e:
        print(f"Error fetching result: {e}")
        return None


def run_simulation(scene: str, prompt: str, episode: str, history: list, request: gr.Request) -> dict:
    model = "InternNav-VLA"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scene_desc = SCENE_CONFIGS.get(scene, {}).get("description", scene)

    user_ip = request.client.host if request else "unknown"
    session_id = request.session_hash

    if not is_request_allowed(user_ip):
        log_submission(scene, prompt, model, user_ip, "IP blocked temporarily")
        raise gr.Error("Too many requests from this IP. Please wait and try again one minute later.")

    submission_result = submit_to_backend(scene, prompt, episode)
    print("submission_result: ", submission_result)

    if submission_result.get("status") != "pending":
        log_submission(scene, prompt, model, user_ip, "Submission failed")
        raise gr.Error(f"Submission failed: {submission_result.get('message', 'unknown issue')}")

    try:
        task_id = submission_result["task_id"]
        SESSION_TASKS[session_id] = task_id

        gr.Info(f"Simulation started, task_id: {task_id}")
        time.sleep(5)
        # Get Task Status
        status = get_task_status(task_id)
        print("first status: ", status)
        result_folder = status.get("result", "")
    except Exception as e:
        log_submission(scene, prompt, model, user_ip, str(e))
        raise gr.Error(f"error occurred when parsing submission result from backend: {str(e)}")

    while True:
        status = get_task_status(task_id)
        if status.get("status") == "completed":
            break
        elif status.get("status") == "failed":
            break
        time.sleep(1)
    if status.get("status") == "completed":
        import base64

        video_bytes = base64.b64decode(status.get("video"))
        receive_time = time.time()
        with open(f"received_video_{receive_time}.mp4", "wb") as f:
            f.write(video_bytes)
        video_path = f"received_video_{receive_time}.mp4"
        new_entry = {"timestamp": timestamp, "scene": scene, "model": model, "prompt": prompt, "video_path": video_path}

        updated_history = history + [new_entry]

        if len(updated_history) > 10:
            updated_history = updated_history[:10]

        print("updated_history:", updated_history)
        log_submission(scene, prompt, model, user_ip, "success")
        gr.Info("Simulation completed successfully!")
        yield video_path, updated_history

    elif status.get("status") == "failed":
        log_submission(scene, prompt, model, user_ip, status.get('result', 'backend error'))
        raise gr.Error(f"task execution fails: {status.get('result', 'backend error')}")
        yield None, history

    elif status.get("status") == "terminated":
        log_submission(scene, prompt, model, user_ip, "terminated")
        video_path = os.path.join(result_folder, "output.mp4")
        if os.path.exists(video_path):
            return f" task {task_id} terminated with some results", video_path, history
        else:
            return f" task {task_id} terminated without any results", None, history

    else:
        log_submission(scene, prompt, model, user_ip, "missing task's status from backend")
        yield None, history


###################################################################################################################
def update_history_display(history: list) -> list:
    print("update_history_display")
    updates = []

    for i in range(10):
        if i < len(history):
            entry = history[i]
            updates.extend(
                [
                    gr.update(visible=True),
                    gr.update(
                        visible=True,
                        label=f"Simulation {i+1}  scene: {entry['scene']}, prompt: {entry['prompt']}",
                        open=False,
                    ),
                    gr.update(value=entry['video_path'], visible=True),
                    gr.update(value=f"{entry['timestamp']}"),
                ]
            )
            print(f'update video')
            print(entry['video_path'])
        else:
            updates.extend(
                [
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value=""),
                ]
            )
    print("update_history_display end!!")
    return updates


def update_scene_display(scene: str):
    print(f"update_scene_display {scene}")
    config = SCENE_CONFIGS.get(scene, {})
    glb_path = config.get("glb_path", "")

    # Validate if file path exists
    if not os.path.exists(glb_path):
        return None, None

    return None, glb_path


def update_episode_display(scene: str, episode: str):
    print(f"update_episode_display {scene} {episode}")
    config = SCENE_CONFIGS.get(scene, {})
    scene_name = config.get("name", "")
    episode_id = int(episode[-1])
    image_path = os.path.join("scene_assets", f"{scene_name}_{episode_id-1}.jpg")
    print(f"image_path {image_path}")
    # valid if file path exists
    if not os.path.exists(image_path):
        return None

    return image_path


def update_log_display():
    logs = read_logs()
    return format_logs_for_display(logs)


##############################################################################


def cleanup_session(request: gr.Request):
    session_id = request.session_hash
    task_id = SESSION_TASKS.pop(session_id, None)
    if task_id:
        try:
            requests.post(f"{BACKEND_URL}/predict/terminate/{task_id}", timeout=3)
            print(f"Task Terminated: {task_id}")
        except Exception as e:
            print(f"Task Termination Failed: {task_id}: {e}")


###############################################################################

custom_css = """
#simulation-panel {
    border-radius: 8px;
    padding: 20px;
    background: #f9f9f9;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
#result-panel {
    border-radius: 8px;
    padding: 20px;
    background: #f0f8ff;
}
.dark #simulation-panel { background: #2a2a2a; }
.dark #result-panel { background: #1a2a3a; }

.history-container {
    max-height: 600px;
    overflow-y: auto;
    margin-top: 20px;
}

.history-accordion {
    margin-bottom: 10px;
}

.scene-preview {
    height: 400px;
    border: 1px solid #ddd;
    border-radius: 8px;
    overflow: hidden;
}
"""

with gr.Blocks(title="Robot Navigation Inference", css=custom_css) as demo:
    gr.Markdown(
        """
    # 🧭 Habitat Robot Navigation Demo
    ### Simulation Test Based on Habitat Framework
    """
    )

    history_state = gr.State([])

    with gr.Row():
        with gr.Column(elem_id="simulation-panel"):
            gr.Markdown("### Simulation Task Configuration")
            with gr.Row():
                scene_dropdown = gr.Dropdown(
                    label="Select Scene",
                    choices=list(SCENE_CONFIGS.keys()),
                    value="scene_1",
                    interactive=True,
                )
                episode_dropdown = gr.Dropdown(
                    label="Select Start Position",
                    choices=list(EPISODE_CONFIGS.keys()),
                    value="episode_1",
                    interactive=True,
                )

            with gr.Row():
                scene_preview = gr.Model3D(
                    elem_classes=["scene-preview"],
                    camera_position=(90.0, 120, 20000.0),
                    # display_mode="solid"
                )
                fps_preview = gr.Image(label="FPS Preview")

            scene_description = gr.Markdown("### Scene preview")

            prompt_input = gr.Textbox(
                label="Navigation Instruction",
                value="Exit the bedroom and turn left. Walk straight passing the gray couch and stop near the rug.",
                placeholder="e.g.: 'Exit the bedroom and turn left. Walk straight passing the gray couch and stop near the rug.'",
                lines=2,
                max_lines=4,
            )

            scene_dropdown.change(
                update_scene_display, inputs=scene_dropdown, outputs=[scene_description, scene_preview]
            ).then(update_episode_display, inputs=[scene_dropdown, episode_dropdown], outputs=[fps_preview])

            episode_dropdown.change(
                update_episode_display, inputs=[scene_dropdown, episode_dropdown], outputs=[fps_preview]
            )

            submit_btn = gr.Button("Start Navigation Simulation", variant="primary")

        with gr.Column(elem_id="result-panel"):
            gr.Markdown("### Latest Simulation Result")

            # Video Output
            video_output = gr.Video(
                label="Live",
                interactive=False,
                format="mp4",
                autoplay=True,
                # streaming=True
            )

            with gr.Column() as history_container:
                gr.Markdown("### History")
                gr.Markdown("#### History will be reset after refresh")

                history_slots = []
                for i in range(10):
                    with gr.Column(visible=False) as slot:
                        with gr.Accordion(visible=False, open=False) as accordion:
                            video = gr.Video(interactive=False)
                            detail_md = gr.Markdown()
                    history_slots.append((slot, accordion, video, detail_md))

    with gr.Accordion("View System Log (DEV ONLY)", open=False):
        logs_display = gr.Markdown()
        refresh_logs_btn = gr.Button("Refresh Log", variant="secondary")

        refresh_logs_btn.click(update_log_display, outputs=logs_display)

    gr.Examples(
        examples=[
            [
                "scene_1",
                "Exit the bedroom and turn left. Walk straight passing the gray couch and stop near the rug.",
                "episode_0",
            ],
            ["scene_2", "Go from reception to conference room passing the water cooler.", "episode_1"],
            ["scene_3", "From the classroom, go to the library via the main hall.", "episode_2"],
            ["scene_4", "From emergency room to pharmacy passing nurse station.", "episode_3"],
        ],
        inputs=[scene_dropdown, prompt_input, episode_dropdown],
        label="Navigation Task Example",
    )

    submit_btn.click(
        fn=run_simulation,
        inputs=[scene_dropdown, prompt_input, episode_dropdown, history_state],
        outputs=[video_output, history_state],
        queue=True,
        api_name="run_simulation",
    ).then(
        fn=update_history_display,
        inputs=history_state,
        outputs=[comp for slot in history_slots for comp in slot],
        queue=True,
    ).then(
        fn=update_log_display,
        outputs=logs_display,
    )

    demo.load(fn=lambda: update_scene_display("scene_1"), outputs=[scene_description, scene_preview]).then(
        fn=update_log_display, outputs=logs_display
    )
    demo.load(fn=lambda: update_episode_display("scene_1", "episode_1"), outputs=[fps_preview])

    def record_access(request: gr.Request):
        user_ip = request.client.host if request else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        log_access(user_ip, user_agent)
        return update_log_display()

    demo.load(fn=record_access, inputs=None, outputs=logs_display, queue=False)

    demo.queue(default_concurrency_limit=8)

    demo.unload(fn=cleanup_session)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5750, debug=True, share=True, allowed_paths=["/mnt"])
