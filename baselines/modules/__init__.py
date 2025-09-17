import os

import numpy as np
import torch
from gym import spaces

from internnav.utils.common_log_util import common_logger as logger

from .basemodel.cma.cma_clip_policy import CMA_CLIP_Net, CMACLIPModelConfig
from .basemodel.cma.cma_policy import CMAModelConfig, CMANet
from .basemodel.internvla_n1.internvla_n1_policy import (
    InternVLAN1ModelConfig,
    InternVLAN1Net,
)
from .basemodel.navdp.navdp_policy import NavDPModelConfig, NavDPNet
from .basemodel.rdp.rdp_policy import RDPModelConfig, RDPNet
from .basemodel.seq2seq.seq2seq_policy import Seq2SeqModelConfig, Seq2SeqNet
from .utils.misc import set_cuda, set_random_seed, wrap_model
from .utils.save import load_checkpoint


def get_policy(policy_name):
    if policy_name == 'CMA_CLIP_Policy':
        return CMA_CLIP_Net
    elif policy_name == 'RDP_Policy':
        return RDPNet
    elif policy_name == 'CMA_Policy':
        return CMANet
    elif policy_name == 'Seq2Seq_Policy':
        return Seq2SeqNet
    elif policy_name == 'InternVLAN1_Policy':
        return InternVLAN1Net
    elif policy_name == 'NavDP_Policy':
        return NavDPNet
    else:
        raise ValueError(f'Policy {policy_name} not found')


def get_config(policy_name):
    if policy_name == 'CMA_CLIP_Policy':
        return CMACLIPModelConfig
    elif policy_name == 'RDP_Policy':
        return RDPModelConfig
    elif policy_name == 'CMA_Policy':
        return CMAModelConfig
    elif policy_name == 'Seq2Seq_Policy':
        return Seq2SeqModelConfig
    elif policy_name == 'InternVLAN1_Policy':
        return InternVLAN1ModelConfig
    elif policy_name == 'NavDP_Policy':
        return NavDPModelConfig
    else:
        raise ValueError(f'Policy {policy_name} not found')
