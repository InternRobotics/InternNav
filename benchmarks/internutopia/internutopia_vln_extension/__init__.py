from internutopia_extension import import_extensions as import_internutopia_extension


def import_extensions():
    import_internutopia_extension()
    from . import controllers, robots, sensors, tasks  # noqa: F401
    from .metrics.vln_pe_metrics import VLNPEMetrics
    from .tasks.vln_eval_task import VLNEvalTask  # noqa: F401
