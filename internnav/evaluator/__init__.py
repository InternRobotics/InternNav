from internnav.evaluator.base import Evaluator
from internnav.evaluator.distributed_base import DistributedEvaluator
from internnav.evaluator.vln_distributed_evaluator import VLNDistributedEvaluator

# register habitat
try:
    import internnav.habitat_extensions.vlln  # noqa: F401 # isort: skip
except Exception as e:
    print(f"Warning: ({e}), Habitat vlln is not loaded in this runtime. Ignore this if not using Habitat vlln.")

try:
    import internnav.habitat_extensions.vln  # noqa: F401 # isort: skip
except Exception as e:
    print(f"Warning: ({e}), Habitat vln is not loaded in this runtime. Ignore this if not using Habitat vln.")


__all__ = [
    'Evaluator',
    'DistributedEvaluator',
    'VLNDistributedEvaluator',
    'HabitatVLNEvaluator',
    'HabitatDialogEvaluator',
]
