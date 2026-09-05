# KV-cache continuation: validation and scope

The existing Habitat DualVLN entry point accepts one optional flag:

```bash
PYTHONPATH=. python scripts/eval/eval.py \
  --config scripts/eval/configs/habitat_dual_system_cfg.py \
  --kv-cache-continuation
```

Use the normal checkpoint/data setup. Omitting the flag keeps full replay.
`scripts/eval/bash/eval_dual_system.sh` also forwards this flag; its existing
Slurm setup is still required. No default configuration or dependency changes.

## Automated regression checks

```bash
PYTHONPATH=. USE_TF=0 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python -m pytest -q tests/unit_test/test_kv_cache_continuation.py
```

23 CPU tests passed without checkpoints or a simulator. The CLI tests exercise
real parsing/configuration, replacing only evaluator initialization: default
configuration unchanged, explicit opt-in, and rejection of unsupported mode.
Tiny-model FP32 tests cover eager/SDPA, one/multiple images, EOS, complete or
missing-final-token caches, stale position state, repeated readout, cache-prefix
preservation, invalid inputs, skipped vision/logits and exception cleanup.
Numerical tolerances are `atol=2e-6, rtol=2e-5`.

The supported cache path is eval-only, unpadded batch one, non-beam generation,
standard `DynamicCache`, full attention, and image input from the same generation.
It processes the uncached final text token and four latent queries in one forward
pass and restores the cache afterward. The original `generate_latents` is unchanged.

## Habitat measurement

Measured on 2026-09-05 against upstream main `7a5c62400ac45b313d9b709c740b64191556a242`
plus this implementation. The submission base is dev
`5e287ed395f0ad4996b7839e9ccd0f48c36870c4`; the DualVLN evaluation, model,
configuration and dependency files used here are identical between those bases.
CPU regressions were also rerun after changing the submission base.

- GPU: one NVIDIA RTX A6000, 48 GB; BF16, FlashAttention 2.7.4.post1.
- Python 3.9, PyTorch 2.6.0+cu124, Transformers 4.51.0, Habitat-Sim 0.2.4.
- Local `InternVLA-N1` checkpoint; complete vision/language/System1 loading,
  with no missing, unexpected or mismatched keys. No quantization or retraining.
- Real R2R `val_unseen`, scene `2azQ1b91cZZ`, fixed episodes 10 and 11.
  Python/NumPy/Torch/Habitat seed `100 + episode_id` at each reset.
- Normal evaluator, observations, prompts and actions. History size 8;
  front/history images 384 x 384, look-down image 640 x 480.
  Original `nextdit_async` System1: 10 diffusion steps, 32 samples.
- Only local paths, episode selection, seeding and the step cap were overridden
  in memory. `max_steps_per_episode=40`; the existing loop reports 41 steps
  for each episode. Auxiliary camera actions are counted separately below.

### Paired latent readout (not the whole evaluation)

At each of the seven actual pixel-goal calls, compare the two readouts using the
same generation. Warm each path once, then time three repetitions per path in
alternating order, with CUDA synchronization before and after each readout.
Take each call's median, then summarize these seven medians. All seven calls
have 10 images and one uncached final token. Generation, preprocessing, System1
and simulator time are excluded; continuation includes final-token catch-up.

| Episode / generation index | Prompt tokens | Full replay (ms) | Continuation (ms) |
| --- | ---: | ---: | ---: |
| 10 / 3 | 2297 | 553.608 | 31.095 |
| 10 / 8 | 2295 | 556.985 | 31.399 |
| 10 / 12 | 2296 | 551.997 | 31.116 |
| 11 / 16 | 2272 | 529.133 | 31.177 |
| 11 / 18 | 2275 | 531.158 | 31.209 |
| 11 / 23 | 2276 | 529.865 | 31.308 |
| 11 / 25 | 2276 | 529.031 | 31.275 |
| Mean | | 540.25 | 31.23 |
| Median | | 531.16 | 31.21 |

Ratio of means: **17.30x latent-readout speedup**. Generation indices are
zero-based across both episodes, including text-action calls.

Mean per-query cosine similarity is 0.999700; minimum query cosine is 0.998746.
Mean relative L2 error is 0.02576 (maximum 0.03628); maximum absolute latent
difference is 2.0. These BF16 outputs are close, **not bitwise equal**.
With the same System1 RNG state, all seven resulting discrete action lists match.
Continuous trajectory values are not identical (maximum raw-output difference
0.277344). Action comparison uses cloned trajectories because `traj_to_actions`
modifies its input in place; shadow calls restore RNG state before navigation resumes.

### Separate bounded closed-loop runs

Run the same two episodes once per mode through the normal eval entry point,
without paired/shadow model calls. Only continuation mode adds the flag.
Measure the evaluator wall time with CUDA synchronization at its boundaries.
This includes episode resets, environment steps, preprocessing, generation and
System1, but excludes evaluator/model initialization. Per-stage synchronization
adds instrumentation overhead. This is a small integration check, not a repeated
statistical end-to-end benchmark.

| Metric | Full replay | Continuation |
| --- | ---: | ---: |
| Two-episode evaluation time | 39.63 s | 35.91 s |
| Generation / latent-readout calls | 26 / 7 | 26 / 7 |
| Actual Habitat actions (including camera probes) | 433 | 433 |
| SR / SPL / oracle success under this cap | 0 / 0 / 0 | 0 / 0 / 0 |
| Mean navigation error | 6.65148 m | 6.65148 m |

Observed wall-time ratio: **1.10x**, or 9.39% less time. All 433 actions match
exactly (215 in episode 10, 218 in episode 11). The paired shadow run also
preserves the baseline action stream. Latent readout accounted for approximately
9.5% of baseline wall time, so its 17.30x local improvement does not imply 17.30x
faster navigation.

## Limits and reproducibility

Repeat the unit command for inexpensive correctness checks. For Habitat A/B,
use the same checkpoint, software/hardware, fixed episode subset and seeds,
and compare full action streams, not just aggregate metrics. For paired timing,
retain `generate(..., return_dict_in_generate=True, use_cache=True)` and compare
`generate_latents(output.sequences, inputs.pixel_values, inputs.image_grid_thw)`
with `generate_latents_from_cache(output, inputs.image_grid_thw, inputs.attention_mask)`
before releasing that generation's cache, using the protocol above. Timing and
episode-selection instrumentation was local; no benchmark framework is added.

Both bounded episodes hit the step cap without success. Identical actions here
do not establish unchanged full-dataset SR/SPL. Neither a complete navigation
benchmark, robot deployment nor full upstream CI was validated. The default-off
scope is intentional; BF16 numerical drift can affect other trajectories.
