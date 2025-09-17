from typing import Union
from transformers import PreTrainedModel, AutoTokenizer, AutoProcessor
import torch
from internnav.configs.model.base_encoders import ModelCfg
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM, InternVLAN1ModelConfig
from internnav.model.utils.vln_utils import S2Output, S1Output, traj_to_actions, chunk_token, split_and_clean
from PIL import Image
import numpy as np
import re
import copy
import itertools
from collections import OrderedDict


class InternVLAN1Net(PreTrainedModel):
    config_class = InternVLAN1ModelConfig

    def __init__(self, config: Union[InternVLAN1ModelConfig, ModelCfg]):
        super().__init__(config)
        self.model_config = ModelCfg(**config.model_cfg['model'])
        
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            self.model_config.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map={"": self.model_config.device}
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(self.model_config.model_path)
        self.processor.tokenizer = self.tokenizer
        self.processor.tokenizer.padding_side = 'left'
        
        self.init_prompts()
        
        self.num_frames = self.model_config.num_frames
        self.num_history = self.model_config.num_history
        self.num_future_steps = self.model_config.num_future_steps
        self.continuous_traj = self.model_config.continuous_traj
        self.resize_w = self.model_config.resize_w
        self.resize_h = self.model_config.resize_h
        
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0  # S2's episode idx is different from the system's idx
        self.conversation_history = []  # Multi-turn conversation exists when looking down
        self.llm_output = ""
        
        
    def init_prompts(self):
        self.DEFAULT_IMAGE_TOKEN = "<image>"
        # For absolute pixel goal 
        prompt = f"You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task." 
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        
        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]

        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3],
            "↓": [5],
        })
        
    def reset(self):
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        
    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)
    
    def step_no_infer(self, rgb, depth, pose):
        image = Image.fromarray(rgb).convert('RGB')
        raw_image_size = image.size
        image = image.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        self.episode_idx += 1
        
    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        # Need to be careful: look_down images are not added to rgb_list and won't be selected as history
        # 1. Preprocess input
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:  # Don't add look_down images to rgb_list
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
        
        # 2. Prepare input for the model
        if not look_down:
            # Clear conversation history when not looking down, provide normal image history and instruction
            self.conversation_history = [] 
            # 2.1 instruction
            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            # 2.2 images
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
                placeholder = (self.DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'
            
            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1  # Only increment when not looking down to maintain correspondence with rgb_list idx
        else:
            # Continue conversation based on previous when looking down
            self.input_images.append(image)  # This image should be the look_down image
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append({ 'role': 'assistant', 'content': [{ 'type': 'text', 'text': self.llm_output}]})
            
        prompt = self.conjunctions[0] + self.DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        prompt_instruction = copy.deepcopy(sources[0]["value"])
        parts = split_and_clean(prompt_instruction)
        
        content = []
        for i in range (len(parts)):
            if parts[i] == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
                input_img_id +=1
            else:
                content.append({"type": "text", "text": parts[i]}) 
        
        self.conversation_history.append({'role': 'user', 'content': content})
        
        text = self.processor.apply_chat_template(
            self.conversation_history, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)
        
        # 3. Model inference
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        self.llm_output = self.processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"============ output {self.episode_idx}  {self.llm_output}")
        output = S2Output()
        
        # 4. Post-process results
        if bool(re.search(r'\d', self.llm_output)):  # Output pixel goal 
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
            output.output_pixel = np.array(pixel_goal)
            
            image_grid_thw = torch.cat(
                [thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0
            )
            with torch.no_grad():
                traj_latents = self.model.generate_latents(
                    output_ids, inputs.pixel_values, image_grid_thw
                )
            output.output_latent = traj_latents
            
        else:  # Output action
            action_seq = self.parse_actions(self.llm_output)
            output.output_action = action_seq
            
        return output
    
            
    def s1_step_latent(self, rgb, depth, latent, use_async=False):
        with torch.no_grad():
            if use_async:
                dp_actions = self.model.generate_traj(latent, rgb, depth, use_async)
            else:
                dp_actions = self.model.generate_traj(latent)

        if self.continuous_traj:
            action_list = traj_to_actions(dp_actions)
        else:
            random_choice = np.random.choice(dp_actions.shape[0])
            action_list = chunk_token(dp_actions[random_choice])
            
        action_list = [x for x in action_list if x != 0]
        
        
        ##If the mode is async, S1 just use the part of actions
        if use_async:
            output = S1Output(idx=action_list[:4])
        else:
            output = S1Output(idx=action_list[:8])
        return output