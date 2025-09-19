from internutopia.core.config.task import MetricCfg


class VLNPEMetricCfg(MetricCfg):
    type: str = 'VLNPEMetric'
    shortest_to_goal_distance: float
    success_distance: float
