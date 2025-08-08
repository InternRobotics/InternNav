![internnav](pic_06.gif)

  # 🧭 IROS Challenge 2025 Nav Track: Vision-and-Language Navigation in Physical Environments

This track challenges participants to develop **multimodal navigation agents** that can interpret **natural language instructions** and operate within a **realistic physics-based simulation** environment.

Participants will deploy their agents on a **legged humanoid robot** (e.g., **Unitree H1**) to perform complex indoor navigation tasks using **egocentric visual inputs** and **language commands**. Agents must not only understand instructions but also perceive the environment, model trajectory history, and predict navigation actions in real time. 

The system should be capable of handling challenges such as camera shake, height variation, and local obstacle avoidance, ultimately achieving robust and safe vision-and-language navigation.

---

### 🧠 Key Objectives

- **Multimodal Perception & Understanding**: Combine egocentric RGB/depth vision with natural language instructions into a unified understanding framework.
- **Physics-based Robustness**: Ensure stable and safe control on a humanoid robot within a physics simulator, handling:
  - Camera shake and motion blur
  - Dynamic height shifts during walking
  - Close-range obstacle avoidance
- **Human-like Navigation**: Demonstrate smooth and interpretable navigation behavior similar to how a human would follow instructions.

---

### 🧪 Simulation Environment

- **Platform**: Physics-driven simulation using [InternUtopia](https://github.com/InternRobotics/InternUtopia)
- **Robot**: Unitree H1 humanoid robot model  
- **Tasks**: Instruction-based navigation in richly furnished indoor scenes  
- **Evaluation**: Based on success rate, path efficiency, and instruction compliance

---

### 🔍 Evaluation Metrics

- **Success Rate (SR)**: Proportion of episodes where the agent reaches the goal location within 3m  
- **SPL**: Success weighted by Path Length
- **Trajectory Length (TL)**: Total length of the trajectory (m)
- **Navigation Error (NE)**: Euclidean distance between the agent's final position and the goal (m)
- **OS Oracle Success Rate (OSR)**: Whether any point along the predicted trajectory reaches the goal within 3m
- **Fall Rate (FR)**: Frequency of the agent falling during navigation
- **Stuck Rate (StR)**: Frequency of the agent becoming stuck during navigation

---

### 🚨 Challenges to Solve

- ✅ Integrating vision, language, and control into a single inference pipeline  
- ✅ Overcoming sensor instability and actuation delay from simulated humanoid locomotion  
- ✅ Ensuring real-time, smooth, and goal-directed behavior under physics constraints

This track pushes the boundary of embodied AI by combining **natural language understanding**, **3D vision**, and **realistic robot control**, fostering solutions ready for future real-world deployments.

---

# 🚀 Get Started

This guide provides a step-by-step walkthrough for participating in the **IROS 2025 Challenge on Multimodal Robot Learning**—from setting up your environment and developing your model, to evaluating and submitting your results.

---

## 🔗 Useful Links
- 🔍 **Challenge Overview:**  
 [Challenge of Multimodal Robot Learning in InternUtopia and Real World](https://internrobotics.shlab.org.cn/challenge/2025/).

- 📖 **InternUtopia + InternNav Documentation:**  
 [Getting Started](https://internrobotics.github.io/user_guide/internutopia/get_started/index.html)

- 🚀 **Interactive Demo:**  
 [InternNav Model Inference Demo](https://huggingface.co/spaces/InternRobotics/InternNav-Eval-Demo)

---

## 🧩 Environment Setup

### Clone the InternNav repository to any desired location
```bash
$ git clone git@github.com:InternRobotics/InternNav.git
```

### Pull our base Docker image
```bash
$ docker pull crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internnav:v1.0
```

### Run the container
```bash
$ xhost +local:root # Allow the container to access the display

$ cd PATH/TO/INTERNNAV/

$ docker run --name internnav -it --rm --gpus all --network host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -e "DISPLAY=${DISPLAY}" \
  --entrypoint /bin/bash \
  -w /root/InternNav \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v ${PWD}:/root/InternNav \
  -v ${HOME}/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ${HOME}/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ${HOME}/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ${HOME}/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ${HOME}/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ${HOME}/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ${HOME}/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ${HOME}/docker/isaac-sim/documents:/root/Documents:rw \
  crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internnav:v1.0
```

### Download the starter dataset (val_seen + val_unseen splits)
```bash
$ git lfs install
# At /root/InternNav/
$ mkdir kujiale_data
# InteriorAgent scene usd
$ git clone https://huggingface.co/datasets/spatialverse/InteriorAgent kujiale_data/scene_data
# InteriorAgent train and val dataset
$ git clone https://huggingface.co/datasets/spatialverse/InteriorAgent_Nav kujiale_data/raw_data

# Latest InternData (required huggingface token to download, generate one from here https://huggingface.co/settings/tokens)
$ git clone -b v0.1-full https://huggingface.co/datasets/InternRobotics/InternData-N1 data
```

### Suggested Dataset Path
#### InternData-N1
```
data/ 
├── Embodiments/
├── scene_data/
│   ├── mp3d/
│   │   ├──17DRP5sb8fy/
│   │   ├── 1LXtFkjw3qL/
│   │   └── ...
├── vln_pe/
│   ├── raw_data/
│   │   ├── train/
│   │   ├── val_seen/
│   │   │   └── val_seen.json.gz
│   │   └── val_unseen/
│   │       └── val_unseen.json.gz
└── └── traj_data/
        ├── interior_agent/
        │   └── kujiale
        └── mp3d/
            └── trajectory_0/
                ├── data/
                ├── meta/
                └── videos/
```
#### Interior_data/
```
kujiale_data
├── scene_data      
│   ├── kujiale_xxxx/
│   └── ...
└── raw_data        
    ├── train/
    ├── val_seen/
    └── val_unseen/
```


### [Optional] Download the baseline model
```bash
# ddppo-models
$ mkdir -p checkpoints/ddppo-models
$ wget -P checkpoints/ddppo-models https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth
# longclip-B
$ huggingface-cli download --include 'longclip-B.pt' --local-dir-use-symlinks False --resume-download Beichenzhang/LongCLIP-B --local-dir checkpoints/clip-long
# download r2r finetuned baseline checkpoints
$ git clone https://huggingface.co/InternRobotics/VLN-PE && mv VLN-PE/r2r checkpoints/
```

## 🛠️ Local Development & Testing

### Develop & test
- Implement your policy under `internnav/model` and add to `internav/agent`.
- We provide train and eval scripts to quick start.
- Use our train script to train your model:
    ```bash
    $ ./scripts/train/start_train.sh --name train_rdp --model rdp
    ```
- Use our evaluation script for quick checks:
    ```bash
    $ ./scripts/eval/start_eval.sh --config scripts/eval/configs/challenge_cfg.py
    ```
- **Example**: Try to train and evaluate the baseline models. 
We provide default train and eval configs named as `challenge_xxx_cfg.py` under `scripts/.../configs`

## 📦 Packaging & Submission

### ✅ Ensure Trained Weights & Model Are Included

Make sure your trained weights and model are correctly packaged in your submitted Docker image at `/root/InternNav` and that the evaluation configuration is properly set at: `scripts/eval/configs/challenge_cfg.py`. No need to include the `data` directory in your submission. We will handle the test dataset.
```bash
# quick check
$ bash challenge/start_eval_iros.sh --config scripts/eval/configs/challenge_cfg.py
```

### Build Your Submission Docker Image

Write a **Dockerfile** and follow the instructions below to build your submission image:
```bash
# Navigate to the directory
$ cd PATH/TO/INTERNNAV/

# Build the new image
$ docker build -t my-internnav-custom:v1 .
```
Or commit your container as new image: 

```bash
$ docker commit [container_name] my-internnav-with-updates:v1
# Easier to manage custom environment
# May include all changes, making the docker image bloat
```

Push to your public registry
```bash
$ docker tag my-internnav-custom:v1 your-registry/internnav-custom:v1
$ docker push your-registry/internnav-custom:v1
```

### Submit your image URL on Eval.AI

#### Submission Format

Create a JSON file with your Docker image URL and team information. The submission must follow this exact structure:

```json
{
    "url": "registry.cn-hangzhou.aliyuncs.com/yourteam/iros2025:dev",
    "team": {
        "name": "your-team-name",
        "members": [
            {
                "name": "John Doe",
                "affiliation": "University of Example",
                "email": "john.doe@example.com",
                "leader": true
            },
            {
                "name": "Jane Smith",
                "affiliation": "Example Research Lab",
                "email": "jane.smith@example.com",
                "leader": false
            }
        ]
    }
}
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Complete Docker registry URL for your submission image |
| `team.name` | string | Official team name for leaderboard display |
| `team.members` | array | List of all team members with their details |
| `members[].name` | string | Full name of team member |
| `members[].affiliation` | string | University or organization affiliation |
| `members[].email` | string | Valid contact email address |
| `members[].leader` | boolean | Team leader designation (exactly one must be `true`) |

For detailed submission guidelines and troubleshooting, refer to the official Eval.AI platform documentation.

## 📝 Official Evaluation Flow
### DSW Creation
- We use the AliCloud API to instantiate an instance from your image link.
- The system mounts the evaluation config + full dataset (val_seen, val_unseen, test).

### Evaluation Execution
- Via SSH + `screen`, we launch `challenge/start_eval_iros.sh --config scripts/eval/configs/challenge_cfg.py`.
- A polling loop watches for result files.

### Results Collection
- Upon completion, metrics for each split are parsed and pushed to the [EvalAI](https://eval.ai/web/challenges/challenge-page/2627/overview) leaderboard.
- The released results are computed as a weighted sum of the test subsets from VLNPE-R2R (MP3D scenes) and Interior-Agent (Kujiale scenes), with a weighting ratio of 2:1.

---
## 📖 Citation
For more details with in-depth physical analysis results on the VLN task, please refer to our **VLN-PE**:
[Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](https://arxiv.org/pdf/2507.13019).
```
@inproceedings{vlnpe,
  title={Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities},
  author={Wang, Liuyi and Xia, Xinyuan and Zhao, Hui and Wang, Hanqing and Wang, Tai and Chen, Yilun and Liu, Chengju and Chen, Qijun and Pang, Jiangmiao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
