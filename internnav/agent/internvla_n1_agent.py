import os
import copy
from PIL import Image
import numpy as np
import imageio
import base64
import pickle
import threading
import cv2
from gym import spaces
import time
import torch

from internnav.configs.agent import StepRequest, AgentCfg
from internnav.agent.base import Agent
from internnav.configs.model.base_encoders import ModelCfg
from internnav.model import get_config, get_policy
from internnav.model.utils.misc import set_random_seed
from internnav.model.utils.vln_utils import S2Input, S2Output, S1Input, S1Output


@Agent.register('internvla_n1')
class InternVLAN1Agent(Agent):
    observation_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(256, 256, 1),
        dtype=np.float32,
    )
    
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        set_random_seed(0)
        vln_sensor_config = self.config.model_settings
        self._model_settings = ModelCfg(**vln_sensor_config)
        env_num = getattr(self._model_settings, 'env_num', 1)
        sim_num = getattr(self._model_settings, 'sim_num', 1)
        self.device = torch.device(self._model_settings.device)
        
        policy = get_policy(self._model_settings.policy_name)
        policy_config = get_config(self._model_settings.policy_name)
        model_config = {'model': self._model_settings.model_dump()}
        self.policy = policy(config=policy_config(model_cfg=model_config))
        self.policy.eval()
        
        self.camera_intrinsic = self.get_intrinsic_matrix(
            self._model_settings.width, self._model_settings.height, self._model_settings.hfov
        )
        
        self.episode_step = 0
        self.episode_idx = 0
        self.look_down = False
        
        self.s1_input = S1Input()
        self.s2_input = S2Input()
        self.s2_output = S2Output()
        self.s1_output = S1Output()
        
        # Thread management
        self.s2_thread = None
        
        # Thread locks
        self.s2_input_lock = threading.Lock()
        self.s2_output_lock = threading.Lock()
        self.s2_agent_lock = threading.Lock()
        
        # Start S2 thread
        self._start_s2_thread()
        
        # vis debug
        self.vis_debug = vln_sensor_config['vis_debug']
        if self.vis_debug:
            self.debug_path = vln_sensor_config['vis_debug_path']
            os.makedirs(self.debug_path, exist_ok=True)
            self.fps_writer = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}.mp4", fps=5)
            self.fps_writer2 = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_dp.mp4", fps=5)
            self.output_pixel = None
            
    def reset(self, reset_index=None):
        '''reset_index: [0]'''
        if reset_index is not None:
            self.episode_idx += 1
            if self.vis_debug:
                self.fps_writer.close()
                self.fps_writer2.close()
        else:
            self.episode_idx = -1
            
        self.episode_step = 0
        self.s1_input = S1Input()
        with self.s2_input_lock:
            self.s2_input = S2Input()
        with self.s2_output_lock:
            self.s2_output = S2Output()
        self.s1_output = S1Output()
        
        # Reset s2 agent
        with self.s2_agent_lock:
            self.policy.reset()
        
        if self.vis_debug:
            self.fps_writer = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}.mp4", fps=5)
            self.fps_writer2 = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_dp.mp4", fps=5)
        
    def get_intrinsic_matrix(self, width, height, hfov) -> np.ndarray:
        width = width
        height = height
        fov = hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array([
            [fx,  0.0, cx, 0.0],
            [ 0.0, fy, cy, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]
        ])
        return intrinsic_matrix
        
    def _start_s2_thread(self):
        def s2_thread_func():
            while True:
                # Check if inference is needed
                should_infer = self.s2_input.should_infer
                if should_infer:
                    with self.s2_input_lock:
                        self.s2_input.should_infer = False
                        s2_infer_idx = self.s2_input.idx
                else:
                    time.sleep(0.5)  # Sleep briefly if inference is not needed
                    continue
                
                # Check if currently inferring
                if not self.s2_output.is_infering:
                    with self.s2_output_lock:
                        self.s2_output.is_infering = True
                else:
                    time.sleep(0.5)  # Sleep briefly if already inferring
                    continue
                
                # Execute inference
                success = True
                try:
                    with self.s2_agent_lock:
                        current_s2_output = self.policy.s2_step(self.s2_input.rgb, self.s2_input.depth, self.s2_input.pose, self.s2_input.instruction, self.camera_intrinsic, self.s2_input.look_down)
                except Exception as e:
                    print(f"s2 infer error: {e}")
                    self.s2_output.is_infering = False
                    self.policy.reset()
                    success = False
                if not success:
                    try:
                        current_s2_output = self.policy.s2_step(self.s2_input.rgb, self.s2_input.depth, self.s2_input.pose, self.s2_input.instruction, self.camera_intrinsic, False)
                    except Exception as e:
                        print(f"s2 infer error: {e}")
                        self.s2_output.is_infering = False
                        self.policy.reset()
                        self.s2_output.output_pixel = None
                        self.s2_output.output_action = [0] # finish the inference
                        self.s2_output.output_latent = None
                        continue
                
                print(f"s2 infer finish!!")
                # Update output state
                with self.s2_output_lock:
                    print(f"get s2 output lock")
                    # S2 output
                    self.s2_output.is_infering = False
                    self.s2_output.output_pixel = current_s2_output.output_pixel
                    self.s2_output.output_action = current_s2_output.output_action
                    self.s2_output.output_latent = current_s2_output.output_latent
                    self.s2_output.idx = s2_infer_idx
                    self.s2_output.rgb_memory = self.s2_input.rgb
                    self.s2_output.depth_memory = self.s2_input.depth
                time.sleep(0.01)  # Sleep briefly after completing inference

        self.s2_thread = threading.Thread(target=s2_thread_func)
        self.s2_thread.daemon = True
        self.s2_thread.start()
        
    def should_infer_s2(self, mode="sync"):
        if self.episode_step == 0:
            return True
        
        if self.s2_output.is_infering:
            return False
        
        # 1. Synchronous mode: infer S2 every frame to provide to S1 for execution
        if mode == "sync":
            if self.s2_output.output_action is None:
                return True
            else:
                return False
        # 2. Partial async mode: S2 infers 1 frame while S1 executes 10 frames
        if mode == "partial_async":
            if self.episode_step - self.s2_output.idx >= 8:
                return True
            if self.s2_output.output_action is None and self.s2_output.output_pixel is None and self.s2_output.output_latent is None:
                # This normally only occurs when output is discrete action and discrete action has been fully executed
                return True
            return False
        # 3. Fully async mode: S2 and S1 run completely in parallel, so S2 infers every frame
        if mode == "full_async":
            return True
        raise ValueError("Invalid mode: {}".format(mode))
    
    def should_infer_s1(self, mode="sync"):
        # 1. Synchronous mode: need to wait for S2 current frame inference to complete before inferring S1
        if mode == "sync":
            if not self.s2_output.is_infering and self.s2_output.idx == self.episode_step:
                return True
            return False
        # 2. Partial async mode: S2 infers 1 frame while S1 executes 10 frames
        if mode == "partial_async":
            print(f"self.s2_output.is_infering {self.s2_output.is_infering} self.s2_output.idx {self.s2_output.idx} self.episode_step {self.episode_step}")
            if not self.s2_output.is_infering and self.s2_output.idx - self.episode_step <= 8 and self.s2_output.idx != -1:
                return True
            return False
        # 3. Fully async mode: only need S2 output, no waiting, directly infer S1
        if mode == "full_async":
            if self.s2_output.idx != -1:
                return True
            return False
        raise ValueError("Invalid mode: {}".format(mode))
    
        
    def step(self, obs):
        mode = 'sync'  # 'sync', 'partial_async', 'full_async'
        
        obs = obs[0]    # do not support batch_env currently?
        rgb = obs['rgb']
        depth = obs['depth']
        instruction = obs['instruction']
        pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
        # S2 inference is done in a separate thread
        if self.should_infer_s2(mode) or self.look_down: # The look down frame must be inferred
            print(f"======== Infer S2 at step {self.episode_step}========")
            with self.s2_input_lock:
                self.s2_input.idx = self.episode_step
                self.s2_input.rgb = rgb
                self.s2_input.depth = depth
                self.s2_input.pose = pose
                self.s2_input.instruction = instruction
                self.s2_input.should_infer = True
                self.s2_input.look_down = self.look_down
        else:
            # Even if this frame doesn't do s2 inference, rgb needs to be provided to ensure history is correct
            self.policy.step_no_infer(rgb, depth, pose)
        # S1 inference is done in the main thread
        while self.s2_output.is_infering:
            time.sleep(0.5)
        
        while not self.s2_output.validate():
            time.sleep(0.2)
        
        output = {}
        # Simple branch:
        # 1. If S2 output is full discrete actions, don't execute S1 and return directly
        print('===============', self.s2_output.output_action, '=================')
        if self.s2_output.output_action is not None:
            output['action'] = [self.s2_output.output_action[0]]
   
            with self.s2_output_lock:
                self.s2_output.output_action = self.s2_output.output_action[1:]
                if self.s2_output.output_action == []:
                    self.s2_output.output_action = None
            if output['action'][0] == 5:
                self.look_down = True
                # Clear action list when looking down
                with self.s2_output_lock:
                    self.s2_output.output_action = None
                    self.s2_output.output_pixel = None
                    self.s2_output.output_latent = None
                output['action'] = [-1]
            else:
                self.look_down = False
                
            print('Output action:', output)
            
        else:
            self.look_down = False
            # 2. If output is in latent form, execute latent S1
            if self.s2_output.output_latent is not None:
                self.output_pixel = copy.deepcopy(self.s2_output.output_pixel)
                print(self.output_pixel)
                self.s1_output = self.policy.s1_step_latent(rgb, depth * 10000.0, self.s2_output.output_latent)
            else:
                assert False, f"S2 output should be either action or latent, but got neither!  {self.s2_output}"
            
            if self.s1_output.idx == []:
                output['action'] = [-1]
            else:
                output['action'] = [self.s1_output.idx[0]]
            with self.s2_output_lock:
                if len(self.s1_output.idx) > 1:
                    self.s2_output.output_action = self.s1_output.idx[1:]
                    if self.s2_output.output_action == []:
                        self.s2_output.output_action = None
                else:
                    self.s2_output.output_action = None
                self.s2_output.output_pixel = None
                self.s2_output.output_latent = None
            print('Output discretized traj:', output['action'])
        
        # Visualization  
        if self.vis_debug:
            vis = rgb.copy()
            if 'action' in output:
                vis = cv2.putText(vis, str(output['action'][0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.output_pixel is not None:
                pixel = self.output_pixel
                vis = cv2.putText(vis, f"{pixel[1]}, {pixel[0]} ({self.s2_output.idx})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.circle(vis, (pixel[1], pixel[0]), 5, (0, 255, 0), -1)
                self.output_pixel = None
            self.fps_writer.append_data(vis)
            
            if self.s1_output.vis_image is not None:
                Image.fromarray(self.s1_output.vis_image).save(os.path.join("./vis_debug_pix/", f"ttttt_{self.episode_step}.png"))
                self.fps_writer2.append_data(self.s1_output.vis_image)
                 
        self.episode_step += 1
        if 'action' in output:
            return [{'action': output['action'], 'ideal_flag': True}]
        elif 'velocity' in output:
            return [{'action': output['velocity'], 'ideal_flag': False}]
        else:
            assert False