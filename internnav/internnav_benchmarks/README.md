# InternNav Benchmarks

## Utopia Evaluator

Evaluator for InternUtopia VLN episodes. It orchestrates end-to-end evaluation by:
(1) materializing episodes, (2) managing environment/agent interaction, (3) logging
per-trajectory progress and aggregate results, and (4) optionally saving visualization
frames and JSON outputs.

This evaluator also defines the **communication protocol** between the agent and the
environment and adapts it to the simulator’s schema. Concretely, the agent consumes
observation batches in a task-level VLN format and produces action batches in a
task-level VLN format; the evaluator then converts those actions to the simulator/
robot-specific format via `env.transform_action_batch(...)`.

### Registration
Registered as ``'utopia'`` via ``@Evaluator.register('utopia')``.

### Supported Benchmarks


**Common:** batched per env slot; strip env-only keys: `{'finish_action','metrics','render','current_pose','fail_reason'}` before calling the agent.

- **VLN-Discrete**
  - **obs:** `{'rgb': H×W×3 np.ndarray, 'depth': H×W(×1) np.ndarray}`
  - **act:** `[{'action': [int], 'ideal_flag': bool}, ...]`

- **VLN-Continuous**
  - **obs:** `{'rgb','depth'}`
  - **act:** `[{'vel': [vx, vy, w], 'stop': bool}, ...]`
