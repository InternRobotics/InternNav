      
# Accessing the lerobot dataset using the LMDB interface
import os
import pandas as pd
import numpy as np
import json
import cv2
from PIL import Image, ImageDraw, ImageFont


def get_font(font_size):
    """Get font, prioritize fonts that support multiple languages"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    try:
        for path in font_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, font_size)
        return ImageFont.load_default()
    except:
        return ImageFont.load_default()


def wrap_text(text, font, max_width):
    """Wrap text to fit specified width"""
    lines = []
    words = text.split(' ')
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = font.getbbox(test_line)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines


def calculate_text_height(text, font, max_width, line_spacing=5):
    """Calculate total height of text after wrapping"""
    wrapped_lines = wrap_text(text, font, max_width)
    bbox = font.getbbox("Ag")
    line_height = (bbox[3] - bbox[1]) + line_spacing
    return len(wrapped_lines) * line_height, wrapped_lines


def find_optimal_font_size(text, panel_width, panel_height, margin=10, title_height=35, min_font_size=10, max_font_size=24):
    """Find the maximum font size that allows text to be fully displayed"""
    available_width = panel_width - 2 * margin
    available_height = panel_height - title_height - 2 * margin
    
    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = get_font(font_size)
        text_height, wrapped_lines = calculate_text_height(text, font, available_width, line_spacing=5)
        
        if text_height <= available_height:
            return font_size, font, wrapped_lines
    
    font = get_font(min_font_size)
    _, wrapped_lines = calculate_text_height(text, font, available_width, line_spacing=5)
    return min_font_size, font, wrapped_lines


def create_text_panel(text, panel_width, panel_height, frame_info="", bg_color=(30, 30, 30), text_color=(255, 255, 255)):
    """Create text panel with automatic font size adjustment"""
    panel = Image.new('RGB', (panel_width, panel_height), bg_color)
    draw = ImageDraw.Draw(panel)
    
    margin = 10
    title_height = 35
    
    # Combine frame info with instruction
    full_text = f"{frame_info}\n\n{text}" if frame_info else text
    
    font_size, font, wrapped_lines = find_optimal_font_size(
        full_text, panel_width, panel_height, 
        margin=margin, title_height=title_height,
        min_font_size=10, max_font_size=22
    )
    
    title_font = get_font(min(font_size + 2, 24))
    
    # Draw title
    title = "Instruction:"
    draw.text((margin, margin), title, font=title_font, fill=(100, 200, 255))
    
    # Draw wrapped text
    bbox = font.getbbox("Ag")
    line_height = (bbox[3] - bbox[1]) + 5
    y_position = title_height
    
    for line in wrapped_lines:
        draw.text((margin, y_position), line, font=font, fill=text_color)
        y_position += line_height
    
    return np.array(panel)


class LerobotAsLmdb:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    
    def get_all_keys(self, allow_scan_list=['r2r']):
        keys = []
        for scan in os.listdir(self.dataset_path):
            scan_path = os.path.join(self.dataset_path, scan)
            if not os.path.isdir(scan_path):
                continue
            if scan not in allow_scan_list:
                continue
            for scene_index in os.listdir(scan_path):
                scene_path = os.path.join(scan_path, scene_index)
                if not os.path.isdir(scene_path):
                    continue
                
                data_dir = os.path.join(scene_path, "data")
                if os.path.exists(data_dir):
                    for chunk_dir in os.listdir(data_dir):
                        if chunk_dir.startswith("chunk-"):
                            chunk_path = os.path.join(data_dir, chunk_dir)
                            chunk_idx = int(chunk_dir.split("-")[1])
                            
                            for file in os.listdir(chunk_path):
                                if file.startswith("episode_") and file.endswith(".parquet"):
                                    episode_idx = int(file.split("_")[1].split(".")[0])
                                    keys.append(f"{scan}_{scene_index}_{chunk_idx:03d}_{episode_idx:06d}")
                else:
                    for trajectory in os.listdir(scene_path):
                        trajectory_path = os.path.join(scene_path, trajectory)
                        if not os.path.isdir(trajectory_path):
                            continue
                        keys.append(f"{scan}_{scene_index}_000_{trajectory:06d}")
        return keys
    
    def get_data_by_key(self, key):
        # key: {scan}_{scene_index}_{chunk_index}_{episode_index}
        parts = key.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid key format: {key}")
        
        if parts[1] == 'flash':
            scan = 'r2r_flash'
            parts.pop(1)
        elif parts[1] == 'aliengo':
            scan = 'r2r_aliengo'
            parts.pop(1)
        else:
            scan = parts[0]

        scene_index = parts[1]
        chunk_idx = int(parts[-2])
        episode_idx = int(parts[-1])
        
        base_path = os.path.join(self.dataset_path, scan, scene_index)
        
        chunk_str = f"chunk-{chunk_idx:03d}"
        parquet_path = os.path.join(base_path, "data", chunk_str, f"episode_{episode_idx:06d}.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        df = pd.read_parquet(parquet_path)
        
        stats_path = os.path.join(base_path, "meta", "episodes_stats.jsonl")
        task_min = 0
        task_max = 0
        
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                for line in f:
                    try:
                        stats_data = json.loads(line.strip())
                        if stats_data.get("episode_index") == episode_idx:
                            task_info = stats_data.get("task_index", {})
                            task_min = task_info.get("min", 0)
                            task_max = task_info.get("max", 0)
                            break
                    except json.JSONDecodeError as e:
                        print(f"Error decoding stats JSON: {e}")
        
        tasks_path = os.path.join(base_path, "meta", "tasks.jsonl")
        episodes_in_json = []  
        finish_status_in_json = None
        fail_reason_in_json = None
        
        with open(tasks_path, 'r') as f:
            for line in f:
                try:
                    json_data = json.loads(line.strip())
                    task_index = json_data.get("task_index")
                    
                    if task_index is not None and task_min <= task_index <= task_max:
                        episodes_in_json.append(json_data)

                        finish_status_in_json = json_data.get('finish_status')
                        fail_reason_in_json = json_data.get('fail_reason')
                except json.JSONDecodeError as e:
                    print(f"Error decoding tasks JSON: {e}")
        
        rgb_path = os.path.join(base_path, "videos", chunk_str, "observation.images.rgb", f"episode_{episode_idx:06d}.npy")
        depth_path = os.path.join(base_path, "videos", chunk_str, "observation.images.depth", f"episode_{episode_idx:06d}.npy")
        
        data = {}
        data['episode_data'] = {}
        data['episode_data']['camera_info'] = {}
        data['episode_data']['camera_info']['pano_camera_0'] = {}
        
        data['episode_data']['camera_info']['pano_camera_0']['position'] = np.array(df['observation.camera_position'].tolist())
        data['episode_data']['camera_info']['pano_camera_0']['orientation'] = np.array(df['observation.camera_orientation'].tolist())
        data['episode_data']['camera_info']['pano_camera_0']['yaw'] = np.array(df['observation.camera_yaw'].tolist())
        
        data['episode_data']['robot_info'] = {}
        data['episode_data']['robot_info']['position'] = np.array(df['observation.robot_position'].tolist())
        data['episode_data']['robot_info']['orientation'] = np.array(df['observation.robot_orientation'].tolist())
        data['episode_data']['robot_info']['yaw'] = np.array(df['observation.robot_yaw'].tolist())
        
        data['episode_data']['progress'] = np.array(df['observation.progress'].tolist())
        data['episode_data']['step'] = np.array(df['observation.step'].tolist())
        data['episode_data']['action'] = df['observation.action'].tolist()
        
        data["finish_status"] = finish_status_in_json
        data["fail_reason"] = fail_reason_in_json
        data["episodes_in_json"] = episodes_in_json
        

        data['episode_data']['camera_info']['pano_camera_0']['rgb'] = np.load(rgb_path)
        data['episode_data']['camera_info']['pano_camera_0']['depth'] = np.load(depth_path)

        
        return data

    def save_video(self, key, output_path=None, fps=10, panel_width=400, show_instruction=True):
        """
        Save RGB data as a video file with instruction panel on the right.
        
        Args:
            key: The episode key
            output_path: Output video file path (default: {key}.mp4 in current directory)
            fps: Frames per second (default: 10)
            panel_width: Width of the instruction panel (default: 400)
            show_instruction: Whether to show instruction panel (default: True)
        
        Returns:
            str: Path to the saved video file
        """
        # Get data by key
        data = self.get_data_by_key(key)
        
        # Extract RGB data
        rgb_data = data['episode_data']['camera_info']['pano_camera_0']['rgb']
        
        # Extract instruction
        instruction = ""
        try:
            if data.get('episodes_in_json') and len(data['episodes_in_json']) > 0:
                # Only show the first task
                instructions_list = []
                if 'task' in data['episodes_in_json'][0]:
                    instruction = data['episodes_in_json'][0]['task']
            else:
                instruction = "No instruction"
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not extract instruction: {e}")
            instruction = "No instruction"
        
        # Set default output path if not provided
        if output_path is None:
            output_path = f"{key}.mp4"
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Get video dimensions
        num_frames, height, width, channels = rgb_data.shape
        
        # Calculate output video dimensions
        if show_instruction:
            output_width = width + panel_width
        else:
            output_width = width
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))
        
        print(f"Processing video for key: {key}")
        print(f"  Frames: {num_frames}, Frame size: {width}x{height}")
        print(f"  Output size: {output_width}x{height}, FPS: {fps}")
        if show_instruction:
            print(f"  Instruction: {instruction[:100]}..." if len(instruction) > 100 else f"  Instruction: {instruction}")
        
        # Write frames
        for i in range(num_frames):
            frame = rgb_data[i]
            
            # Ensure data is in uint8 format
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add instruction panel if requested
            if show_instruction:
                # Create frame info text
                frame_text = f"Frame: {i+1}/{num_frames}"
                
                # Create text panel
                text_panel = create_text_panel(
                    instruction, 
                    panel_width, 
                    height,
                    frame_info=frame_text
                )
                text_panel_bgr = cv2.cvtColor(text_panel, cv2.COLOR_RGB2BGR)
                
                # Combine frame and text panel horizontally
                combined_frame = np.hstack([frame_bgr, text_panel_bgr])
            else:
                combined_frame = frame_bgr
            
            video_writer.write(combined_frame)
            
            # Progress indicator
            if (i + 1) % 50 == 0 or i == num_frames - 1:
                print(f"  Progress: {i + 1}/{num_frames} frames ({100*(i+1)/num_frames:.1f}%)")
        
        # Release video writer
        video_writer.release()
        
        print(f"✓ Video saved to: {output_path}")
        
        return output_path


if __name__ == '__main__':
    ds = LerobotAsLmdb('/shared/smartbot/vln-pe-0.5')
    allow_scan_name = 'r2r_flash'
    keys = ds.get_all_keys(allow_scan_list=[allow_scan_name])
    print(f"total keys:{len(keys)}")
    for k in keys[:-5]: 
        try:
            o = ds.get_data_by_key(k)
            # ds.save_video(k, output_path=f"videos/{allow_scan_name}/{k}.mp4")
            print(f"Key: {k}")
            print(f"  Finish status: {o.get('finish_status')}")
            print(f"  Tasks in JSON: {len(o.get('episodes_in_json', []))}")
            print(f"  RGB data: {'loaded' if o['episode_data']['camera_info']['pano_camera_0'].get('rgb') is not None else 'not found'}")
            print(f"  Depth data: {'loaded' if o['episode_data']['camera_info']['pano_camera_0'].get('depth') is not None else 'not found'}")
        except Exception as e:
            print(f"Error processing key {k}: {e}")

    