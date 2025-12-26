#!/usr/bin/env python3
from typing import List, Optional, Union

import attr
import numpy as np
from habitat.core.dataset import Episode
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationGoal


@attr.s(auto_attribs=True)
class AgentPosition:
    position: Union[None, List[float], np.ndarray]


@attr.s(auto_attribs=True)
class DialogViewLocation:
    agent_state: AgentPosition


@attr.s(auto_attribs=True, kw_only=True)
class DialogGoal(NavigationGoal):
    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None
    bbox: Optional[List[float]] = None
    view_points: Optional[List[DialogViewLocation]] = None


@attr.s(auto_attribs=True, kw_only=True)
class DialogEpisode(Episode):
    object_category: Optional[str] = None
    goals: List[DialogGoal] = attr.ib(
        default=None,
        validator=not_none_validator,
        on_setattr=Episode._reset_shortest_path_cache_hook,
    )
    instruction: Optional[dict] = []
    frames: Optional[int] = []
