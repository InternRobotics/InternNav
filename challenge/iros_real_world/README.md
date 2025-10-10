# 🧭 IROS On-site Challenge

Welcome to the **IROS Vision-Language Navigation On-site Challenge**!
In this phase, participants’ models will be deployed on **a real robot** to evaluate performance in real-world conditions.

---

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
python test_agent.py --config path/to/cfg.py
```

### 3. Actual Competition Execution
During the on-site evaluation, the organizers will run:

```bash
python main.py
```

for each episode, paired with its corresponding natural language instruction.

## 🧩 Data Format
Action
```python
action = [{'action': int, 'ideal_flag': bool}]
```
Observation
```python
obs = {
    "rgb": rgb,           # RGB image from the robot
    "depth": depth,       # Depth image (aligned with RGB)
    "instruction": str    # Natural language navigation
}
```
