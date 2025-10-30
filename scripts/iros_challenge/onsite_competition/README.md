# 🧭 IROS On-site Challenge

Welcome to the **IROS Vision-Language Navigation On-site Challenge**!
In this phase, participants’ models will be deployed on **a real robot** to evaluate performance in real-world conditions.

---

## Robot
The robot uses a wheeled chassis. The chassis and camera control code can be found in the provided SDK. The camera’s default resolution is 640×480, and the depth and color images are already aligned.

The robot is an agilex [RANGER MINI 3.0](https://www.agilex.ai/solutions/1) with RGB camera and a LiDAR sensor.

## ⚙️ Installation

First, install the `InternNav` package:

```bash
cd /InternNav
pip install -e .
```

## 🚀 Running Your Agent
### 1. Start the Agent Server
Launch your agent server with the desired configuration file:

```bash
python -m internnav.agent.utils.server --config path/to/cfg.py
```

### 2. Test the Agent with Robot Captures
You can locally test your model using previously recorded observations from the robot (stored under ./captures):

```bash
python challenge/onsite_competition/sdk/test_agent.py  # you may need to modify the path to your agent
```

### 3. Actual Competition Execution
During the on-site evaluation, the organizers will run:

```bash
python challenge/onsite_competition/sdk/main.py
```

for each episode, paired with its corresponding natural language instruction.

## 🧩 Data Format
Action
```python
action = [{'action': [int], 'ideal_flag': bool}]
```
Observation
```python
obs = {
    "rgb": rgb,           # RGB image from the robot
    "depth": depth,       # Depth image (aligned with RGB)
    "instruction": str    # Natural language navigation
}
```

## 📋 Rules
Please check out the [onsite competition rules](./onsite_competition_rules_en-US.md) .


## 🚀 Code Submission
Submit a Docker image with your agent server preconfigured and ready to run. During the competition, the robot will connect to a local server over the network.

Please also bring a local copy of your image for the day of onsite testing (10.18), in case there is any issue with the network connection. Contact us if you need any assistance.
