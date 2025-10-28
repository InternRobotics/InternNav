def get_policy(policy_name):
    if policy_name == 'CMA_CLIP_Policy':
        from .basemodel.cma.cma_clip_policy import CMA_CLIP_Net, CMACLIPModelConfig

        return CMA_CLIP_Net
    elif policy_name == 'RDP_Policy':
        from .basemodel.rdp.rdp_policy import RDPNet

        return RDPNet
    elif policy_name == 'CMA_Policy':
        from .basemodel.cma.cma_policy import CMAModelConfig, CMANet

        return CMANet
    elif policy_name == 'Seq2Seq_Policy':
        from .basemodel.seq2seq.seq2seq_policy import Seq2SeqModelConfig, Seq2SeqNet

        return Seq2SeqNet
    elif policy_name == 'InternVLAN1_Policy':
        from .basemodel.internvla_n1.internvla_n1_policy import (
            InternVLAN1ModelConfig,
            InternVLAN1Net,
        )

        return InternVLAN1Net
    elif policy_name == 'NavDP_Policy':
        from .basemodel.navdp.navdp_policy import NavDPNet

        return NavDPNet
    else:
        raise ValueError(f'Policy {policy_name} not found')


def get_config(policy_name):
    if policy_name == 'CMA_CLIP_Policy':
        from .basemodel.cma.cma_clip_policy import CMA_CLIP_Net, CMACLIPModelConfig

        return CMACLIPModelConfig
    elif policy_name == 'RDP_Policy':
        from .basemodel.rdp.rdp_policy import RDPModelConfig

        return RDPModelConfig
    elif policy_name == 'CMA_Policy':
        from .basemodel.cma.cma_policy import CMAModelConfig, CMANet

        return CMAModelConfig
    elif policy_name == 'Seq2Seq_Policy':
        from .basemodel.seq2seq.seq2seq_policy import Seq2SeqModelConfig, Seq2SeqNet

        return Seq2SeqModelConfig
    elif policy_name == 'InternVLAN1_Policy':
        from .basemodel.internvla_n1.internvla_n1_policy import (
            InternVLAN1ModelConfig,
            InternVLAN1Net,
        )

        return InternVLAN1ModelConfig
    elif policy_name == 'NavDP_Policy':
        from .basemodel.navdp.navdp_policy import NavDPModelConfig

        return NavDPModelConfig
    else:
        raise ValueError(f'Policy {policy_name} not found')
