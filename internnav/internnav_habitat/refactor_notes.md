# Refactoring `habitat_vln_evaluator`

This note explains how to split the current `VLNEvaluator` implementation into the
framework's `Env` and `Agent` abstractions.

## 1. Construction and configuration (lines 45-135)
* **Environment**: Habitat config loading, dataset split selection, sensor parameter
  extraction, and measurement registration belong to the environment setup. These
  responsibilities configure and own the simulator state and therefore should be
  moved into a `HabitatVLNEnv` class that extends `Env`.【F:internnav/evaluator/habitat_vln_evaluator.py†L45-L103】
* **Agent**: Model handles, prompt bootstrapping, conversation history, action
  vocabulary, and instruction templates are part of the policy logic and should be
  carried by a dedicated `HabitatVLNAgent` subclass. These fields initialize the
  reasoning model rather than the simulator.【F:internnav/evaluator/habitat_vln_evaluator.py†L104-L135】

## 2. Perception utilities (lines 137-236)
Depth pre-processing, intrinsic matrix computation, coordinate transforms, and GPS
projection are tied to the simulator sensor geometry. They should move into the
`HabitatVLNEnv` so that observation tensors returned to the agent are already in a
consistent world frame.【F:internnav/evaluator/habitat_vln_evaluator.py†L137-L236】

## 3. Visualization helper (lines 238-309)
The dot-matrix overlay operates purely on rendered frames and can stay as an
environment utility. The helper should become a method of the environment (or a
separate visualization module) so evaluators can call it regardless of the agent.
【F:internnav/evaluator/habitat_vln_evaluator.py†L238-L309】

## 4. Low-level point navigation (lines 311-347)
The `_pointnav` helper controls a waypoint-following controller that consumes
processed observations and outputs low-level actions. Because it interacts with the
robot's state (goal resets, depth resizing, point-goal calculation), it fits inside
the environment. The agent can request point-goal actions through a method such as
`HabitatVLNEnv.pointnav(goal, depth, ...)`.【F:internnav/evaluator/habitat_vln_evaluator.py†L311-L347】

## 5. Main evaluation loop (lines 349-520)
* **Environment**: Episode iteration, resetting, stepping, intrinsic assembly, and
  metric gathering should be owned by the environment. Wrapping Habitat's episode
  lifecycle in `HabitatVLNEnv` keeps the evaluator thin and deterministic.
* **Agent**: Generating waypoint predictions, maintaining conversation turns, and
  deciding discrete actions are policy responsibilities. The evaluator should ask
  the new agent for an action by passing observations (RGB, depth, state metadata)
  returned by the environment wrapper.【F:internnav/evaluator/habitat_vln_evaluator.py†L349-L520】

## 6. Language and action parsing (lines 522-680)
Instruction processing (`split_and_clean`, dynamic prompt assembly) and action string
parsing convert model text into executable commands. These should be encapsulated in
`HabitatVLNAgent` so the evaluator only receives structured actions (e.g., STOP,
MOVE, LOOK).【F:internnav/evaluator/habitat_vln_evaluator.py†L522-L680】

## 7. Metric aggregation and exports (lines 682-745)
Writing JSON lines, aggregating SPL/OS/NE, and optional video dumping can remain in
the evaluator, but the raw metrics originate from the environment through
`HabitatVLNEnv.get_metrics()` and rendering helpers. The evaluator should simply
post-process the aggregated numbers.【F:internnav/evaluator/habitat_vln_evaluator.py†L682-L745】

## Resulting structure
1. **`internnav/env/habitat_vln_env.py`**: wraps Habitat configuration, episode
   control, sensor processing, point-nav helper, and visualization utilities.
2. **`internnav/agent/habitat_vln_agent.py`**: encapsulates the vision-language
   model, prompt management, observation parsing, and action decoding.
3. **`internnav/evaluator/habitat_vln_evaluator.py`**: becomes a thin coordinator
   that instantiates the env/agent via the registry, loops over episodes, and logs
   metrics.

This split brings the Habitat evaluator in line with the existing framework while
keeping domain-specific functionality in focused components.
