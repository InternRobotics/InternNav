from internnav.evaluator.base import Evaluator
from internnav.evaluator.distributed_base import DistributedEvaluator

# from internnav.evaluator.vln_multi_evaluator import VlnMultiEvaluator
from internnav.evaluator.vln_multi_distributed_evaluator import (
    VlnMultiDistributedEvaluator,
)

# register habitat
try:
    import internnav.internnav_habitat  # noqa: F401 # isort: skip
except Exception as e:
    print(f"Warning: ({e}), Habitat Evaluation is not loaded in this runtime. Ignore this if not using Habitat.")


__all__ = ['Evaluator', 'DistributedEvaluator']
