# Habitat VL-LN in InternNav

Vision-Language-and-Language Navigation (VL-LN) is a new [benchmark](https://0309hws.github.io/VL-LN.github.io/) built upon VLN in Habitat, which refers to the setting that models take the vision and language as input and output language and navigation actions. In contrast to VLN, where agents only take navigation actions, agents in VL-LN could ask questions and engage in dialogue with users to complete tasks better with language interaction.
This package adapts [Meta AI Habitat](https://aihabitat.org) for VL-LN within InternNav. It wraps Habitat environments that expose semantic masks, registers dialog-aware datasets and metrics, and provides evaluators that coordinate agent actions, NPC interactions, and logging.

Install our benchmark [dataset](https://huggingface.co/datasets/InternRobotics/VL-LN-Bench) and the [latest checkpoints](https://huggingface.co/InternRobotics/VL-LN-Bench-basemodel) from HuggingFace.
Place the downloaded benchmark under `InternNav/projects/VL-LN-Bench` to match the default path expected by the code.

## Package structure

```
InternNav
├── assets/
├── internnav/
│   ├── habitat_vlln_extensions
│   │   ├── simple_npc
│   │   │   ├── api_key.txt
│   │   ├── measures.py
│   │   ├── habitat_dialog_evaluator.py
│   │   ├── habitat_vlln_env.py
│   ... ... ...
...
├── projects
│   ├── VL-LN-Bench/
│   │   ├── base_model/
│   │   ├── raw_data/
│   │   ├── scene_datasets/
│   │   │   └── mp3d/
│   │   │       └── 17DRP5sb8fy/
│   │   │       ├── 1LXtFkjw3qL/
│   │   │       ...
│   │   ├── traj_data/
...
```

* `__init__.py` re-exports the public entry points so callers can import
  `HabitatVllnEnv`, `HabitatDialogEvaluator`, and `DialogDatasetV1` directly from
  `internnav.habitat_extensions`.
* `habitat_vlln_env.py` extends the shared `HabitatEnv` wrapper to compute
  semantic masks for goal instances and expose depth camera intrinsics needed by
  dialog tasks.
* `habitat_dialog_evaluator.py` implements a dialog-capable evaluator that ties
  together Habitat configuration, InternNav agents, environment rollout, NPC
  interactions, and metric collection.
* `measures.py` registers additional Habitat measurements (path length,
  oracle-style success, SPL variants, step counts) needed by the evaluator.
* `simple_npc/` contains a lightweight NPC backed by an OpenAI chat model plus
  prompt templates for different dialog modes.
* `utils/` hosts dataset loaders (`dialog_dataset.py`), episode dataclasses
  (`dialog_episodes.py`), and navigation/path utilities (`dialog_utils.py`).

## Habitat environment wrapper

`HabitatVllnEnv` is registered under the key `"habitat_vlln"` via the shared
`Env.register` decorator. It subclasses the generic Habitat wrapper to add
VLLN-specific capabilities:

1. During initialization it reads the underlying simulator's depth sensor
   settings (FOV, intrinsics, height) and bootstraps a ground-truth perception
   helper (`MP3DGTPerception`) for Manhattan-world segmentation masks.
2. `reset()` and `step()` defer to the base `HabitatEnv` but, when the current
   task name contains "instance", they also attach a `semantic` mask derived
   from the current depth frame and episode goal bounding boxes.
3. Utility helpers expose the episodic-to-global transform and assemble the
   camera-to-point-cloud transform so segmentation can be projected correctly in
   world coordinates.

This keeps Habitat-specific math encapsulated while exposing extra observations
needed by dialog agents.

## Evaluation pipeline

`HabitatDialogEvaluator` extends the shared `DistributedEvaluator` to orchestrate
interactive VLLN evaluation:

* **Configuration:** It merges a Habitat baseline YAML with the experiment config
  (`get_config`), sets the task name/split, and enables map and collision
  measurements. The Habitat config and output directory are injected back into
  the `EnvCfg` so the base evaluator can spin up `HabitatVllnEnv`.
* **Agent and NPC setup:** InternNav agent settings are primed with the task and
  sensor intrinsics. A `SimpleNPC` instance is constructed to answer questions
  using an OpenAI chat model, enforcing a maximum number of dialog turns.
* **Episode loop:** For each Habitat episode the evaluator restores progress,
  prepares output directories, and resets the agent. When dialog is enabled it
  loads per-scene object/region summaries to support NPC responses. At each step
  it forwards observations to the agent, translates actions (navigation, look
  down, ask NPC), updates semantic masks as needed, and logs actions/paths.
* **Dialog handling:** When the agent issues an ask action the evaluator
  assembles context (episode instruction, path description, semantic mask) and
  queries the NPC. NPC answers are appended to the action log and included in
  observations so the agent can react.
* **Metrics and output:** After an episode ends it records success, SPL, oracle
  success, navigation error, step counts, and serialized trajectories to
  `result.json`. `calc_metrics` aggregates tensors across ranks for global
  reporting.

## Datasets and utilities

* `DialogDatasetV1` registers a Habitat dataset named `dialog`, deserializing
  dialog episodes/goals, normalizing scene paths, and filtering by split.
* `dialog_episodes.py` defines attr-based dataclasses for dialog-specific goals,
  view locations, and instructions used by the dataset.
* `dialog_utils.py` merges Habitat configs, computes shortest paths, and
  generates path descriptions/object summaries that inform NPC answers or agent
  prompts. It also includes visualization helpers for depth and top-down maps.


## Dialog agent

The evaluator drives the `DialogAgent` defined in `internnav/agent/dialog_agent.py`.
Key behaviors:

* **Sensor handling:** `convert_input` filters depth via Habitat's
  `filter_depth`, tracks camera intrinsics/FOV, and maintains RGB/depth history
  plus look-down frames for pixel-goal projection.
* **Prompt construction:** `inference` builds chat-style prompts that mix the
  current instruction, historical observations (as `<image>` tokens), and prior
  NPC exchanges. It uses the Hugging Face Qwen2.5-VL model/processor to decode
  the next action string.
* **Action decoding:** `convert_output` parses the language model output to
  detect dialog requests (`<talk>`), look-down actions, or pixel goal
  coordinates. Pixel goals are converted to world GPS points, validated for
  navigability, and passed to Habitat's `ShortestPathFollower` for motion.
* **Step loop:** `step` alternates between model inference and executing queued
  navigation actions, handles NPC answers, manages goal retries, and logs model
  outputs to the evaluator's episode transcript.

## Simple NPC module

The `simple_npc` package implements a minimal in-environment character for
answering clarification questions:

* `simple_npc.py` wraps the OpenAI chat client, loads API keys, and exposes
  `answer_question` with one- or two-turn prompt flows that can request
  disambiguation or path descriptions.
* `prompt.py` houses the templated prompts and disambiguation responses used by
  `SimpleNPC`.
* `get_description.py` builds natural-language descriptions of navigation paths
  and regions to support NPC answers when the agent asks for help.

By centralizing Habitat VLLN adaptations here, InternNav can run dialog-driven
navigation experiments without modifying the rest of the training or evaluation
stack.
