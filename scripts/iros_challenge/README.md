![internnav](demo.gif)

  # 🧭 IROS Challenge 2025 Nav Track: Vision-and-Language Navigation in Physical Environments

This track challenges participants to develop **multimodal navigation agents** that can interpret **natural language instructions** and operate within a **realistic physics-based simulation** environment.

Participants will deploy their agents on a **legged humanoid robot** (e.g., **Unitree H1**) to perform complex indoor navigation tasks using **egocentric visual inputs** and **language commands**. Agents must not only understand instructions but also perceive the environment, model trajectory history, and predict navigation actions in real time.

The system should be capable of handling challenges such as camera shake, height variation, and local obstacle avoidance, ultimately achieving robust and safe vision-and-language navigation.

---
## 🆕 Updates
- [2025/10/09] Real-world challenge phase is released! check onsite_competition part for the details.
- We have fixed possible memory leak inside InternUtopia. Please pull the latest image v1.2 to use.
- For submission, please make sure the image contain `screen`. Quick check: `$ screen --version`.

## 📋 Table of Contents
- [📚 Getting Started](#-get-started)
- [🔗 Useful Links](#-useful-links)
- [🧩 Environment Setup](#-environment-setup)
- [🛠️ Model Training and Testing](#-model-training-and-testing)
- [📦 Packaging and Submission](#-packaging-and-submission)
- [📝 Official Evaluation Flow](#-official-evaluation-flow)
- [📖 About the Challenge](#-about-the-challenge)
- [🔗 Citation](#-citation)
- [👏 Contribution](#-contribution)

## 🚀 Get Started

This guide provides a step-by-step walkthrough for participating in the **IROS 2025 Challenge on Multimodal Robot Learning**—from setting up your environment and developing your model, to evaluating and submitting your results.



## 🔗 Useful Links
- 🔍 **Challenge Overview:**
 [Challenge of Multimodal Robot Learning in InternUtopia and Real World](https://internrobotics.shlab.org.cn/challenge/2025/).

- 📖 **InternUtopia + InternNav Documentation:**
 [Getting Started](https://internrobotics.github.io/user_guide/internutopia/get_started/index.html)

- 🚀 **Interactive Demo:**
 [InternNav Model Inference Demo](https://huggingface.co/spaces/InternRobotics/InternNav-Eval-Demo)



## 🧩 Environment Setup

### Clone the InternNav repository to any desired location
```bash
$ git clone git@github.com:InternRobotics/InternNav.git --recursive
```

### Pull our base Docker image
```bash
$ docker pull crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internnav:v1.2
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
  -v ${PWD}/data/scene_data/mp3d_pe:/isaac-sim/Matterport3D/data/v1/scans:ro \
  crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internnav:v1.2
```

### Download the starter dataset (val_seen + val_unseen splits)
All the datasets are in LeRobot format. Please refer to [Dataset Structure & Format Specification](https://internrobotics.github.io/user_guide/internnav/tutorials/dataset.html).

Download the [**InteriorNav Dataset**](https://huggingface.co/datasets/spatialverse/InteriorAgent)
```bash
$ git lfs install
# At /root/InternNav/
$ mkdir interiornav_data

# InteriorNav scene usd
$ git clone https://huggingface.co/datasets/spatialverse/InteriorAgent interiornav_data/scene_data

# InteriorNav val dataset
$ git clone https://huggingface.co/datasets/spatialverse/InteriorAgent_Nav interiornav_data/raw_data

# train data can be found in next section under IROS-2025-Challenge-Nav
```
Please refer to [document](https://internrobotics.github.io/user_guide/internnav/quick_start/installation.html#interndata-n1-dataset-preparation) for a full guide on InternData-N1 Dataset Preparation. In this challenge, we used test on the VLN-PE part of the [InternData-N1](https://huggingface.co/datasets/InternRobotics/InternData-N1) dataset. Optional: please feel free to download the full dataset to train your model.

- Download the [**IROS-2025-Challenge-Nav Dataset**](https://huggingface.co/datasets/InternRobotics/IROS-2025-Challenge-Nav/tree/main) for the `vln_pe/`,
- Download the [SceneData-N1](https://huggingface.co/datasets/InternRobotics/Scene-N1/tree/main) for the `scene_data/`,
- Download the [Embodiments](https://huggingface.co/datasets/InternRobotics/Embodiments) for the `Embodiments/`

```bash
# InternData-N1 with vln-pe data only
$ git clone https://huggingface.co/datasets/InternRobotics/IROS-2025-Challenge-Nav data

# Scene
$ wget https://huggingface.co/datasets/InternRobotics/Scene-N1/resolve/main/mp3d_pe.tar.gz    # unzip to data/scene_data

# Embodiments
$ git clone https://huggingface.co/datasets/InternRobotics/Embodiments data/Embodiments
```

### Suggested Dataset Directory Structure
#### InternData-N1
```
data/
├── Embodiments/
├── scene_data/
│   └── mp3d_pe/
│       ├──17DRP5sb8fy/
│       ├── 1LXtFkjw3qL/
│       └── ...
└── vln_pe/
    ├── raw_data/                       # JSON files defining tasks, navigation goals, and dataset splits
    │   └── r2r/
    │       ├── train/
    │       ├── val_seen/
    │       │   └── val_seen.json.gz
    │       └── val_unseen/
    └── traj_data/                      # training sample data for two types of scenes
        ├── interiornav/
        │   └── kujiale_xxxx.tar.gz
        └── r2r/
            └── trajectory_0/
                ├── data/
                ├── meta/
                └── videos/
```
#### Interior_data/
```bash
interiornav_data
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

# pulled code need to download longclip and diffusion policy
$ git submodule update --init
```

## 🛠️ Model Training and Testing

Please refer to the [documentation](https://internrobotics.github.io/user_guide/internnav/quick_start/train_eval.html) for a quick-start guide to training or evaluating supported models in InternNav.

For advanced usage, including customizing datasets, models, and experimental settings, see the [tutorial](https://internrobotics.github.io/user_guide/internnav/tutorials/index.html).

### Requirements
For fair comparison in this IROS challenge, the USD file, controller, and observation space must remain consistent with the provided implementation.
- **Robot USD file**: Includes the Unitree H1 assets and an RGB-D camera.
- **Controller**: Supports four discrete actions: move forward 0.25 m, turn left 15°, turn right 15°, and stop.
- **Observation space**: Ego-centric monocular RGB-D input.
- **Technical**: All publicly available datasets and pretrained weights are allowed. The use of large-scale model APIs (e.g., GPT, Claude, Gemini, etc.) is **not** permitted. **Note**: the test server for this challenge has no internet access.

**Note**: Please use our provided camera usd `camera_prim_path='torso_link/h1_pano_camera_0'` as the RGB-D camera, the resolution can be `[640, 480]` or `[256, 256]`.

### Development Overview
The main architecture of the evaluation code adopts a client-server model. In the client, we specify the corresponding configuration (*.cfg), which includes settings such as the scenarios to be evaluated, robots, models, and parallelization parameters. The client sends requests to the server, which then make model to predict and response to the client.

The InternNav project adopts a modular design, allowing developers to easily add new navigation algorithms.
The main components include:

- **Model**: Implements the specific neural network architecture and inference logic

- **Agent**: Serves as a wrapper for the Model, handling environment interaction and data preprocessing

- **Config**: Defines configuration parameters for the model and training

### Example: Train & Evaluate the Baseline Model
- We provide train and eval scripts to quick start.
- Use our train script to train your model:
    ```bash
    $ conda activate internutopia
    $ pip install -r requirements/train.txt --index-url https://pypi.org/simple

    $ ./scripts/train/start_train.sh --name train_rdp --model rdp
    ```
- Use our evaluation script for quick checks:
    ```bash
    $ ./scripts/eval/start_eval.sh --config scripts/eval/configs/challenge_cfg.py
    ```
- Currently supported baseline model: Sequence-to-Sequence  (Seq2Seq), Cross-Modal Attention (CMA), Recurrent Diffusion Policy (RDP). Implementations can be found at:
    - `internnav/agent/`: model agent
    - `internnav/model/`: trained model
    - `scripts/train/configs`: training configs
    - `scripts/eval/configs`: evaluating configs
- The evaluation process now can be viewed at `logs/`. Update `challenge_cfg.py` to get visualization output:
    - Set `eval_settings['vis_output']=True` to see saved frames and video during the evaluation trajectory
    - Set `env_settings['headless']=False` to open isaac-sim interactive window
    <img src="output.gif" alt="output" style="width:50%;">

### Create Your Model & Agent
#### Custom Model
A Model is the concrete implementation of your algorithm. For each step, the model should expect an observation from the ego-centric camera.
```
action = self.agent.step(obs)
```
**obs** has format:
```
obs = [{
    'globalgps': [X, Y, Z]              # robot location
    'globalrotation': [X, Y, Z, W]      # robot orientation in quaternion
    'rgb': np.array(256, 256, 3)        # rgb camera image
    'depth': np.array(256, 256, 1)      # depth image
}]
```
**action** has format:
```
action = List[int]                      # action for each environments
# 0: stop
# 1: move forward
# 2: turn left
# 3: turn right
```
#### Create a Custom Config Class

In the model file, define a `Config` class that inherits from `PretrainedConfig`.
A reference implementation is `CMAModelConfig` in [`cma_model.py`](../internnav/model/cma/cma_policy.py).

#### Registration and Integration

In [`internnav/model/__init__.py`](../internnav/model/__init__.py):
- Add the new model to `get_policy`.
- Add the new model's configuration to `get_config`.

#### Create a Custom Agent

The Agent handles interaction with the environment, data preprocessing/postprocessing, and calls the Model for inference.
A custom Agent usually inherits from [`Agent`](../internnav/agent/base.py) and implements the following key methods:

- `reset()`: Resets the Agent's internal state (e.g., RNN states, action history). Called at the start of each episode.
- `inference(obs)`: Receives environment observations `obs`, performs preprocessing (e.g., tokenizing instructions, padding), calls the model for inference, and returns an action.
- `step(obs)`: The external interface, usually calls `inference`, and can include logging or timing.

Example: [`CMAAgent`](../internnav/agent/cma_agent.py)

#### Create a Trainer

The Trainer manages the training loop, including data loading, forward pass, loss calculation, and backpropagation.
A custom trainer usually inherits from the [`Base Trainer`](../internnav/trainer/base.py) and implements:

- `train_epoch()`: Runs one training epoch (batch iteration, forward pass, loss calculation, parameter update).
- `eval_epoch()`: Evaluates the model on the validation set and records metrics.
- `save_checkpoint()`: Saves model weights, optimizer state, and training progress.
- `load_checkpoint()`: Loads pretrained models or resumes training.

Example: [`CMATrainer`](../internnav/trainer/cma_trainer.py) shows how to handle sequence data, compute action loss, and implement imitation learning.

#### Training Data

The training data is under `data/vln_pe/traj_data`. Our dataset provides trajectory data collected from the H1 robot as it navigates through the task environment.
Each observation in the trajectory is paired with its corresponding action.

You may also incorporate external datasets to improve model generalization.

#### Evaluation Data
In `raw_data/val`, for each task, the model should guide the robot at the start position and rotation to the target position with language instruction.

#### Set the Corresponding Configuration

Refer to existing **training** configuration files for customization:

- **CMA Model Config**: [`cma_exp_cfg`](../scripts/train/configs/cma.py)

Configuration files should define:
- `ExpCfg` (experiment config)
- `EvalCfg` (evaluation config)
- `IlCfg` (imitation learning config)

Ensure your configuration is imported and registered in [`__init__.py`](../scripts/train/configs/__init__.py).

Key parameters include:
- `name`: Experiment name
- `model_name`: Must match the name used during model registration
- `batch_size`: Batch size
- `lr`: Learning rate
- `epochs`: Number of training epochs
- `dataset_*_root_dir`: Dataset paths
- `lmdb_features_dir`: Feature storage path

Refer to existing **evaluation** config files for customization:

- **CMA Model Evaluation Config**: [`h1_cma_cfg.py`](../scripts/eval/configs/h1_cma_cfg.py)

Main fields:
- `name`: Evaluation experiment name
- `model_name`: Must match the name used during training
- `ckpt_to_load`: Path to the model checkpoint
- `task`: Define the tasks settings, number of env, scene, robots
- `dataset`: Load r2r or interiornav dataset
- `split`: Dataset split (`val_seen`, `val_unseen`, `test`, etc.)

## 📦 Packaging and Submission

### ✅ Run the benchmark locally (same entrypoint as EvalAI)

Use this to evaluate your model on the validation split locally. The command is identical to what EvalAI runs, so it’s also a good sanity check before submitting.

- Make sure your trained weights and model code are correctly packaged in your submitted Docker image at `/root/InternNav`.
- The evaluation configuration is properly set at: `scripts/eval/configs/challenge_cfg.py`.
- No need to include the `data` directory in your submission.
```bash
# Run local benchmark on the validation set
$ bash challenge/start_eval_iros.sh --config scripts/eval/configs/challenge_cfg.py --split [val_seen/val_unseen]

```

### Build Your Submission Docker Image

Write your **Dockerfile** and follow the instructions below to build your submission image:
```bash
# Navigate to the directory
$ cd PATH/TO/INTERNNAV/

# Build the new image
$ docker build -t my-internnav-custom:v1 .
```
Or commit your container as new image:

```bash
$ docker commit internnav my-internnav-with-updates:v1
# Easier to manage custom environment
# May include all changes, making the docker image bloat. Please delete cache and other operations to reduce the image size.
```

Push to your public registry. You can follow the following [aliyun document](https://help.aliyun.com/zh/acr/user-guide/create-a-repository-and-build-images?spm=a2c4g.11186623.help-menu-60716.d_2_15_4.75c362cbMywaYx&scm=20140722.H_60997._.OR_help-T_cn~zh-V_1) or [Quay document](https://quay.io/tutorial/) to create a free personal image registry. During the creation of the repository, please set it to public access.

```bash
$ docker tag my-internnav-custom:v1 your-registry/internnav-custom:v1
$ docker push your-registry/internnav-custom:v1
```

[Optional] quick test your image with a mini split in r2r dataset, 10 episodes should be done. This also tests whether you have set the image to public access.
```bash
$ docker logout
$ docker run --name internnav-test -it --gpus all --network host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -e "DISPLAY=${DISPLAY}" \
  --entrypoint /bin/bash \
  -w /root/InternNav \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v ${PWD}/data:/root/InternNav/data \
  -v ${PWD}/interiornav_data:/root/InternNav/interiornav_data \
  your-registry/internnav-custom:v1 \
  -c "challenge/start_eval_iros.sh --config scripts/eval/configs/challenge_cfg.py --split mini; exec /bin/bash"
```

### Submit your image URL on Eval.AI

After creating an account and team on [eval.ai](https://eval.ai/web/challenges/challenge-page/2627/overview), please submit your entry here. In the "Make Submission" column at the bottom, you can select phase. Please select Upload file as the submission type and upload the JSON file shown below. If you select private for your submission visibility, the results will not be published on the leaderboard. You can select public again on the subsequent result viewing page.

#### Submission Format

Create a JSON file with your Docker image URL and team information. The submission must follow this exact structure:

```json
{
    "url": "your-registry/internnav-custom:v1",
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

## 📖 About the Challenge

### 🧠 Key Objectives

- **Multimodal Perception & Understanding**: Combine egocentric RGB/depth vision with natural language instructions into a unified understanding framework.
- **Physics-based Robustness**: Ensure stable and safe control on a humanoid robot within a physics simulator, handling:
  - Camera shake and motion blur
  - Dynamic height shifts during walking
  - Close-range obstacle avoidance
- **Human-like Navigation**: Demonstrate smooth and interpretable navigation behavior similar to how a human would follow instructions.


### 🧪 Simulation Environment

- **Platform**: Physics-driven simulation using [InternUtopia](https://github.com/InternRobotics/InternUtopia)
- **Robot**: Unitree H1 humanoid robot model
- **Tasks**: Instruction-based navigation in richly furnished indoor scenes
- **Evaluation**: Based on success rate, path efficiency, and instruction compliance



### 🔍 Evaluation Metrics

- **Success Rate (SR)**: Proportion of episodes where the agent reaches the goal location within 3m
- **SPL**: Success weighted by Path Length
- **Trajectory Length (TL)**: Total length of the trajectory (m)
- **Navigation Error (NE)**: Euclidean distance between the agent's final position and the goal (m)
- **OS Oracle Success Rate (OSR)**: Whether any point along the predicted trajectory reaches the goal within 3m
- **Fall Rate (FR)**: Frequency of the agent falling during navigation
- **Stuck Rate (StR)**: Frequency of the agent becoming stuck during navigation



### 🚨 Challenges to Solve

- ✅ Integrating vision, language, and control into a single inference pipeline
- ✅ Overcoming sensor instability and actuation delay from simulated humanoid locomotion
- ✅ Ensuring real-time, smooth, and goal-directed behavior under physics constraints

This track pushes the boundary of embodied AI by combining **natural language understanding**, **3D vision**, and **realistic robot control**, fostering solutions ready for future real-world deployments.



## 🔗 Citation
For more details with in-depth physical analysis results on the VLN task, please refer to **VLN-PE**:
[Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](https://arxiv.org/pdf/2507.13019).
```
@inproceedings{vlnpe,
  title={Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities},
  author={Wang, Liuyi and Xia, Xinyuan and Zhao, Hui and Wang, Hanqing and Wang, Tai and Chen, Yilun and Liu, Chengju and Chen, Qijun and Pang, Jiangmiao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## 👏 Contribution
- **Organizer**: Shanghai AI Lab
- **Co-organizers**: ManyCore Tech, University of Adelaide
- **Data Contributions**: Online test data provided by Prof. Qi Wu's team; Kujiale scenes provided by ManyCore Tech
- **Sponsors** (in no particular order): ByteDance, HUAWEI, ENGINEAI, HONOR, ModelScope, Alibaba Cloud, AGILEX, DOBOT
