# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

Upcoming changes will be tracked in this section.

## Changelog of v0.3.0 (2026/01/05)
### Highlights
- Support training of InternVLA-N1 and evaluation on RxR (#184)
- Support training and evaluation for the [VL-LN benchmark](https://arxiv.org/html/2512.22342v2) (#193, #198)
- Add a new flash without collision controller (#189)

### New Features
- Add training code for InternVLA-N1 (#184)
- Support evaluation on the RxR dataset (#184)
- Add training code for the VL-LN benchmark baseline (#198)
- Support evaluation on VL-LN benchmark (#193)
- Add a Flash-without-Collisoin controller (#189)

### Improvements
- Decouple System 2 and Dual-System evaluation functions in the Habitat evaluator for better readability (#184)
- Update InternVLA-N1 agent in VLN-PE to align with the updated InternVLA-N1 policy interface (#184)
- Enhance the Habitat evaluation pipeline to handle NaN values in results (#217)
- Update the README to include community tutorials (#217)

### Bug Fixes
- Fix the version of diffusers in the requirements (#184)
- Fix the result JSON saving path in VLN-PE (#217)
- Fix a bug in RxR evaluation result collection (#217)
- Removed legacy code in scripts/demo (#217)

### Contributors
@kellyiss @DuangZhu @0309hws @kew6688

Full Changelog: https://github.com/InternRobotics/InternNav/compare/release/v0.2.0...release/v0.3.0

## Changelog of v0.2.0 (2025/12/04)
### Highlights
- Support distributed evaluation for VLN-PE, reducing full benchmark runtime to ~1.6 hours using 16 GPUs (≈13× speedup over single-GPU eval) (#168)
- Enhance Habitat evaluation flow with `DistributedEvaluator` and `HabitatEnv` integrated into the InternNav framework (#168)
- Support install flags for dependency isolation: `[habitat]`, `[isaac]`, `[model]` (#135)

### New Features
- Support distributed evaluation for VLN-PE (#168)
- Support a unified evaluation script `eval.py`, with new Habitat evaluation configs in `scripts/eval/configs` (#168)
- Support install flags for dependency isolation (#168)

### Improvements
- Add `HabitatEnv` with episode pool management (#168)
- Update `InternUtopiaEnv` for distributed execution and episode pool management (#168)
- Enhance `episode_loader` in VLN-PE with new distributed mode compatibility (#168)
- Update `data_collector` to support progress checkpointing and incremental result aggregation in distributed evaluation. (#168)

### Bug Fixes
- Fix logger disabled after Isaac Sim initialization during evaluator bootstrap (#168)
- Fix dataloader bug where `revise_one_data()` was incorrectly applied to all datasets (#168)
- Fix visualization images dimension mismatch during InternVLA-N1 evaluation (#168)
- Fix distributed evaluation crash in rdp policy (#168)
- Fix GitHub CI tests (#168)

### Contributors
A total of 3 developers contributed to this release.
@kew6688, @Gariscat, @yuqiang-yang

Full changelog: https://github.com/InternRobotics/InternNav/compare/release/v0.1.0...release/v0.2.0
