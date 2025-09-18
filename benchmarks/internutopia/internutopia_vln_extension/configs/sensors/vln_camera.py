from typing import Optional, Tuple

from internutopia.core.config.robot import SensorCfg


class VLNCameraCfg(SensorCfg):
    type: Optional[str] = 'VLNCamera'
    enable: Optional[bool] = True
    resolution: Optional[Tuple[int, int]]
