# IROS On-site Challenge

In this phase, the model will be tested with real robot.

Install internnav
```
cd /InternNav
pip install -e .
```

start your agent server:

```
python -m internnav.agent.utils.server --config path/to/cfg.py
```

Test the agent with observation that taken from the robot (`./captures`)
```
python test_agent.py --config path/to/cfg.py
```

In the actual test, we will run `main.py` for each episode with corresponding instruction.

## format
```
action = [int]

obs = {
    "rgb": rgb,
    "depth": depth,
    "instruction": str
}
```
