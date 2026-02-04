# Compatibility

## v0.3.1

### InternData-N1 update to v0.5

The InternData-N1 VLN-PE trajectory training dataset has been upgraded from `v0.1` to `v0.5`. This update introduces minor structural changes in the dataset layout and updates the LeRobot-to-LMDB conversion logic to match the new `v0.5` data structure.

The training pipeline now uses the new key name:
- `instruction_text` → `task`

The updated conversion logic is **not compatible** with InternData-N1 `v0.1`.

