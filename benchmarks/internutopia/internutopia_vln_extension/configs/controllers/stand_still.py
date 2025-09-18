from typing import Optional

from internutopia.core.config.robot import ControllerCfg


class StandStillControllerCfg(ControllerCfg):
    type: Optional[str] = 'StandStillController'
