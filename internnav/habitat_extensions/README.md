# Habitat in InternNav

This package adapts [Meta AI Habitat](https://aihabitat.org) environments and
metrics so they can be used from InternNav's evaluation framework. It provides
an environment wrapper, custom measurements, and evaluator implementations that
bridge Habitat simulations with InternNav agents and distributed evaluation
utilities.

## Package structure

```
habitat_extensions/
├── __init__.py
├── habitat_env.py
├── habitat_default_evaluator.py
├── habitat_vln_evaluator.py
└── measures.py
```

* `__init__.py` re-exports the public entry points for the environment and the
  VLN evaluator so they can be imported as
  `from internnav.habitat_extensions import HabitatEnv`.
* `habitat_env.py` implements the `Env` subclass that wraps Habitat's
  `Env` object. It bootstraps episodes, handles sharding across distributed
  ranks, and adapts Habitat's observations to InternNav's expectations.
* `habitat_default_evaluator.py` contains a lightweight evaluator that runs a
  conventional Habitat agent inside the InternNav evaluator loop.
* `habitat_vln_evaluator.py` is the task-specific evaluator used for Vision-
  and-Language Navigation (VLN). It loads InternNav vision-language models,
  orchestrates inference, and logs results during distributed evaluation.
* `measures.py` registers additional Habitat measurements (path length,
  oracle metrics, step counts) that are required by the evaluators.


## Habitat environment wrapper

`HabitatEnv` is registered under the key `"habitat"` via the shared
`Env.register` decorator. When InternNav builds an environment from an
`EnvCfg`, the wrapper:

1. Imports and instantiates the Habitat `Env` using the configuration object
   provided in `env_settings['habitat_config']`.
2. Stores the distributed context (`local_rank`, `world_size`) and any output
   directory override (`output_path`).
3. Pre-computes the episode list by grouping Habitat episodes by scene,
   filtering completed episodes via `progress.json`, and sharding the remaining
   work by rank.
4. Implements the standard reset/step/close/render accessors expected by the
   InternNav `Env` base class while delegating to the underlying Habitat
   simulator.

This design keeps the Habitat-specific logic isolated from the rest of the
framework and ensures that distributed evaluation proceeds deterministically
across ranks.

## Evaluation pipeline

InternNav evaluators extend the shared `DistributedEvaluator` base class, which
handles distributed initialization, environment instantiation, metric
aggregation, and result logging. The Habitat integration provides two
implementations:

### `HabitatVlnEvaluator`

The VLN evaluator (`habitat_vln_evaluator.py`) is responsible for coordinating
model inference in Habitat scenes.

* **Configuration:** During initialization the evaluator reads an `EvalCfg`
  whose `env.env_settings['config_path']` points to a Habitat YAML file. The
  config is loaded with Habitat's baseline utilities, sensor intrinsics are
  cached, and custom measurements (`top_down_map`, `collisions`) are enabled.
* **Environment binding:** The Habitat configuration is injected back into the
  `EnvCfg` so the shared `DistributedEvaluator` base class can create the
  `HabitatEnv` wrapper with the correct settings.
* **Model loading:** Depending on `cfg.agent.model_settings.mode`, the evaluator
  loads either the InternVLA dual-system model or a Qwen2.5-VL model using
  Hugging Face Transformers. The processor is configured with left padding and
  the model is moved to the rank-local GPU.
* **Episode loop:**
  1. `HabitatEnv.reset()` advances to the next episode and returns the first
     observation.
  2. The evaluator reads episode metadata (scene, instruction) from Habitat,
     constructs prompt messages, and collects RGB/depth history for the
     language model.
  3. Visual inputs are prepared (resizing, optional look-down depth capture) and
     depth maps are filtered through `filter_depth` to remove sensor noise.
  4. The evaluator queries the loaded model for the next action sequence,
     translates model tokens to Habitat actions via `traj_to_actions`, and
     steps the environment.
  5. Per-episode metrics (`success`, `SPL`, oracle success, navigation error)
     are appended and checkpointed to `progress.json` for resumability.
* **Aggregation:** After all ranks finish, inherited utilities gather per-rank
  tensors, compute global averages, and write `result.json` in
  `output_path`.

### `HabitatVlnEvaluator` (baseline)

The default evaluator in `habitat_default_evaluator.py` offers a simpler loop
where a pre-built InternNav agent interacts with the Habitat environment.
InternNav's agent abstraction is reset with each new Habitat episode, and
per-step actions are produced via `agent.act()`. The evaluator records the same
metrics as the VLN evaluator, making it useful for baselines or sanity checks.

## Custom Habitat measurements

`measures.py` registers a suite of metrics with Habitat's registry so that they
are available in the Habitat configuration:

* `PathLength`: cumulative Euclidean distance traveled by the agent.
* `OracleNavigationError`: minimum geodesic distance to the goal along the
  trajectory.
* `OracleSuccess`: binary success metric derived from oracle navigation error
  relative to a goal radius (default 3.0 meters).
* `OracleSPL`: best Success weighted by Path Length value observed during the
  trajectory.
* `StepsTaken`: number of actions issued by the agent, including STOP.

These metrics complement Habitat's built-in success and SPL scores, allowing
InternNav to report a richer set of statistics.

## Extending the integration

* **Adding evaluators:** Subclass `DistributedEvaluator`, supply
  Habitat-specific initialization similar to `HabitatVlnEvaluator`, and
  implement `eval_action` and `calc_metrics`.
* **Custom sensors or observations:** Augment the Habitat YAML configuration and
  update `HabitatEnv` or the evaluator to consume the new observation keys.
* **Additional metrics:** Register new measures in `measures.py` and enable them
  in the Habitat config via `config.habitat.task.measurements.update(...)`.

By centralizing Habitat-specific logic in this package, InternNav can swap in
other simulators or extend Habitat support without touching the rest of the
training and evaluation stack.
