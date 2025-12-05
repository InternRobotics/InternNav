import gzip
import json
import os
from collections import defaultdict
from numbers import Number
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.distributions import constraints
from torch.distributions.normal import Normal


class TemperatureTanh(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        """The hyperbolic tangent with an optional temperature."""
        super().__init__()
        assert temperature != 0.0, 'temperature must be nonzero.'
        self._T = temperature
        self.tanh = torch.nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.tanh(x / self._T)


class TruncatedNormal(nn.Module):
    """The truncated normal distribution is derived from the normal
    distribution and is bounded above, below, or by both. It is parameterized
    by the mean and variance of the untruncated normal distribution. This is
    a custom implementation because it doesn't exist in pytorch.
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    https://en.wikipedia.org/wiki/Truncated_normal_distribution
    """

    def __init__(
        self,
        loc: Tensor,
        scale: Union[float, Tensor],
        smin: float = -np.inf,
        smax: float = np.inf,
        validate_args: Optional[Any] = None,
    ) -> None:
        super().__init__()
        assert smin < smax, 'smin must be less than smax'
        assert np.isfinite(smin) and np.isfinite(
            smax
        ), 'two-sided truncation is required for now. Set both `smin` and `smax`.'
        assert (loc >= smin).all() and (loc <= smax).all(), f'loc is out of range ({smin}, {smax}): {loc}'
        if isinstance(scale, Number):
            assert scale >= 0.0, 'scale is negative'
        else:
            assert (scale >= 0.0).all(), 'scale is negative'

        self._normal = Normal(loc, scale, validate_args=validate_args)
        self._loc = loc
        self._scale = scale
        self._smin = smin
        self._smax = smax
        self._unbounded = self._smin == -np.inf and self._smax == np.inf
        self.A = 1 / (self._scale * np.sqrt(2 * np.pi))
        try:
            self.Z = self._normal.cdf(self._smax) - self._normal.cdf(self._smin)
        except Exception:
            device = loc.device
            smax_tensor = torch.tensor([self._smax], device=device)
            smin_tensor = torch.tensor([self._smin], device=device)
            self.Z = self._normal.cdf(smax_tensor) - self._normal.cdf(smin_tensor)
            del smax_tensor
            del smin_tensor
        self.support = constraints.interval(self._smin, self._smax)
        self._init_mean_variance_entropy()

    def _init_mean_variance_entropy(self) -> None:
        """References for entropy:
        https://github.com/olmjo/RcppTN
        https://en.wikipedia.org/wiki/Truncated_normal_distribution
        """
        standard_normal = Normal(0.0, 1.0)
        standard_normal.pdf = lambda x: (np.e ** (-0.5 * (x**2))) / np.sqrt(2 * np.pi)
        alpha = (self._smin - self._loc) / self._scale
        beta = (self._smax - self._loc) / self._scale

        alpha_pdf = standard_normal.pdf(alpha)
        beta_pdf = standard_normal.pdf(beta)

        alpha_cdf = standard_normal.cdf(alpha)
        beta_cdf = standard_normal.cdf(beta)
        standard_Z = beta_cdf - alpha_cdf

        self._mean = self._loc - self._scale * ((beta_pdf - alpha_pdf) / standard_Z)

        t1 = (beta * beta_pdf - alpha * alpha_pdf) / standard_Z
        t2 = ((beta_pdf - alpha_pdf) / standard_Z) ** 2
        self._variance = (self._scale**2) * (1 - t1 - t2)

        self._entropy = 0.5 * np.log(2 * np.pi * np.e)
        self._entropy += torch.log(self._scale * standard_Z)
        self._entropy += (alpha * alpha_pdf - beta * beta_pdf) / (2 * standard_Z)

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        return self._variance

    def sample(self, resample_limit: int = 10000) -> Tensor:
        if self._unbounded:
            return self._normal.sample()

        samples = self._normal.sample()
        do_resample = (samples < self._smin).logical_or(samples > self._smax)
        num_resamples = 0
        while do_resample.any():
            assert (
                num_resamples < resample_limit
            ), f'Hit resample limit of {resample_limit} for bounds [{self._smin}, {self._smax}]'
            num_resamples += 1

            samples[do_resample] = self._normal.sample()[do_resample]
            do_resample = (samples < self._smin).logical_or(samples > self._smax)

        return samples

    def log_prob(self, value: Union[float, Tensor]) -> Tensor:
        if self._unbounded:
            return self._normal.log_prob(value)

        msg = 'value is out of truncation range and has an undefined log_prob.'
        if isinstance(value, Number):
            assert value >= self._smin and value <= self._smax, msg
        else:
            assert (value >= self._smin).all() and (value <= self._smax).all(), msg

        normal_prob_density = self.A * np.e ** (-0.5 * ((value - self._loc) / self._scale) ** 2)
        truncated_prob_density = normal_prob_density / self.Z

        if isinstance(truncated_prob_density, Number):
            return np.log(truncated_prob_density)
        else:
            return truncated_prob_density.log()

    def mode(self):
        return self._loc

    def entropy(self):
        return self._entropy


class DotProductAttention(nn.Module):
    def __init__(self, key_dimension: int) -> None:
        super().__init__()
        self.scale = torch.tensor(1.0 / ((key_dimension) ** 0.5))
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Scaled dot-product attention with an optional mask.
        2X speed improvement over `torch.einsum`.
        Args:
            query: [Batch, Dk]
            key: [Batch, Dk, P]
            value: [Batch, Dv, P]
        Returns:
            tensor of dimension [Batch, Dv]
        """
        energy = torch.bmm(Q.unsqueeze(1), K)
        if mask is not None:
            energy *= mask.unsqueeze(1).float()

        attn = self.softmax(energy * self.scale)
        return torch.bmm(attn, V.permute(0, 2, 1)).squeeze(1)


class MultiHeadDotProductAttention(nn.Module):
    def __init__(
        self,
        d_q_in: int,
        d_k_in: int,
        d_v_in: int,
        d_qk: int,
        d_v: int,
        num_heads: int,
        d_out: int,
        normalize: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        """The residual connection of Vaswani et al is not used here. The
        residual makes sense if self-attention is being used.
        Args:
            d_q_in (int): dimension of the query vector input
            d_k_in (int): dimension of the key vector input
            d_v_in (int): dimension of the value vector input
            d_qk (int): dimension to map queries & keys to prior to attention
            d_v (int): dimension to map values to prior to attention
            num_heads (int): number of attention heads
            d_out (int): output dimension of this module (final linear layer)
        """
        super().__init__()
        self.num_heads = num_heads
        self.normalize = normalize
        self.q_linear = nn.Linear(d_q_in, d_qk * num_heads, bias=False)
        self.k_linear = nn.Linear(d_k_in, d_qk * num_heads, bias=False)
        self.v_linear = nn.Linear(d_v_in, d_v * num_heads, bias=False)

        self.attn = DotProductAttention(d_qk)
        self.final_linear = nn.Linear(d_v * num_heads, d_out, bias=False)

        self.dropout = None
        if dropout_p > 0.0:
            self.dropout = nn.Dropout(dropout_p)

        if self.normalize:
            self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: None = None) -> Tensor:
        """Performs multihead scaled dot product attention for some Q, K, V.
        Args:
            Q: [Batch, d_q_in]
            K: [Batch, d_k_in, P]
            V: [Batch, d_v_in, P]
        """
        assert K.shape[2] == V.shape[2], 'keys must be the same size as values'

        Q = self.q_linear(Q)
        K = self.k_linear(K.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        V = self.v_linear(V.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        Q = Q.view(Q.shape[0] * self.num_heads, Q.shape[1] // self.num_heads)
        K = K.view(
            K.shape[0] * self.num_heads,
            K.shape[1] // self.num_heads,
            K.shape[2],
        )
        V = V.view(
            V.shape[0] * self.num_heads,
            V.shape[1] // self.num_heads,
            V.shape[2],
        )

        attended_V = self.attn(Q, K, V, mask=mask)

        attended_V = attended_V.view(
            attended_V.shape[0] // self.num_heads,
            self.num_heads,
            attended_V.shape[1],
        )

        attended_V = attended_V.view(attended_V.shape[0], attended_V.shape[1] * attended_V.shape[2])

        out = self.final_linear(attended_V)
        if self.dropout is not None:
            out = self.dropout(out)
        if self.normalize:
            out = self.layer_norm(out)
        return out


class CustomFixedCategorical(torch.distributions.Categorical):
    """Same as the CustomFixedCategorical in hab-lab, but renames log_probs
    to log_prob. All the torch distributions use log_prob.
    """

    def sample(self, sample_shape: Size = torch.Size()) -> Tensor:  # noqa: B008
        return super().sample(sample_shape).unsqueeze(-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


def batched_index_select(x: torch.Tensor, dim: int, index: torch.LongTensor) -> torch.Tensor:
    """A batched index_select where each batch selects different indices.

    Args:
        x: size [B, d0, d1, ..., dn]
        dim: int where 0 <= dim < len(x.size())
        index: size [B, d0, d1, ..., dn]

    Returns:
        torch.Tensor where the selected dimension has been squeezed.

    Example:
        >>> x = torch.randn(2,3,4)
        >>> index = torch.randint(0,3, (2,))
        >>> result = batched_index_select(x, 1, index)  # size: [2, 4]
    """
    views = [x.shape[0]] + [1 if i != dim else -1 for i in range(1, len(x.shape))]
    expanse = list(x.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(x, dim, index).squeeze(dim)


def get_delta(actions):
    if isinstance(actions, torch.Tensor):
        # append zeros to first action
        device = actions.device  # Get the device of the input tensor
        ex_actions = torch.cat(
            [
                torch.zeros((actions.shape[0], 1, actions.shape[-1]), device=device),
                actions,
            ],
            dim=1,
        )
        delta = ex_actions[:, 1:] - ex_actions[:, :-1]
    elif isinstance(actions, np.ndarray):
        if len(actions.shape) == 2:
            ex_actions = np.concatenate([np.zeros((1, actions.shape[-1])), actions], axis=0)
            delta = ex_actions[1:] - ex_actions[:-1]
        else:
            ex_actions = np.concatenate(
                [np.zeros((actions.shape[0], 1, actions.shape[-1])), actions],
                axis=1,
            )
            delta = ex_actions[:, 1:] - ex_actions[:, :-1]

    return delta


def map_action_to_2d(delta_actions):
    actions_2d = np.zeros((delta_actions.shape[0], 2))
    for a_idx, action in enumerate(delta_actions):
        if action[2] > 0:
            # turn right
            actions_2d[a_idx] = [0, 1]
        elif action[2] < 0:
            # turn left
            actions_2d[a_idx] = [0, -1]
        elif action[0] == action[1] == action[2] == 0:
            # stop
            actions_2d[a_idx] = [0, 0]
        else:
            # forward
            actions_2d[a_idx] = [1, 0]
    return actions_2d


def get_action(diffusion_output, action_stats, cumsum=True):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)

    ndeltas = diffusion_output

    ndeltas = unnormalize_data(ndeltas, action_stats)
    if cumsum:
        import torch

        torch.use_deterministic_algorithms(False)
        actions = torch.cumsum(ndeltas, dim=1)  # This get the relative actions (not delta) from the diffusion output
    else:
        actions = ndeltas
    return actions.float()


# normalize data
def get_data_stats(data):
    data_xy = data[:, :2]
    data_xy = data_xy.reshape(-1, data_xy.shape[-1])
    stats = {'min': np.min(data_xy, axis=0), 'max': np.max(data_xy, axis=0)}
    return stats


def normalize_data(data, stats, device=None):
    if device is not None:
        if isinstance(stats['min'], np.ndarray):
            stats['min'] = torch.from_numpy(stats['min'])
            stats['max'] = torch.from_numpy(stats['max'])
            stats['min'] = stats['min'].to(device)
            stats['max'] = stats['max'].to(device)

    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    # else:
    #     ndata = (data[:, :2] - stats['min'][:2]) / (stats['max'][:2] - stats['min'][:2])
    #     ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    if isinstance(ndata, torch.Tensor):
        device = ndata.device

    if len(ndata.shape) == 3:
        ndata_part = (ndata + 1) / 2
    else:
        ndata_part = (ndata[:, :2] + 1) / 2
    try:
        data = ndata_part * (stats['max'].to(device) - stats['min'].to(device)) + stats['min'].to(device)
    except Exception:
        data = ndata_part * (stats.max.to(device) - stats.min.to(device)) + stats.min.to(device)

    return data


def action_reduce(action_mask, unreduced_loss: torch.Tensor):
    # Reduce over non-batch dimensions to get loss per batch element
    while unreduced_loss.dim() > 1:
        unreduced_loss = unreduced_loss.mean(dim=-1)
    assert unreduced_loss.shape == action_mask.shape, f'{unreduced_loss.shape} != {action_mask.shape}'
    return (unreduced_loss * action_mask).mean() / (action_mask.float().mean() + 1e-2)


def load_dataset(dataset_root_dir, split, logger=None, dataset_type='r2r'):
    load_data = defaultdict(list)
    with gzip.open(os.path.join(dataset_root_dir, f'{split}', f'{split}.json.gz'), 'rt', encoding='utf-8') as f:
        data = json.load(f)
        for item in data['episodes']:
            if 'scene_id' in item.keys():
                if '/' in item['scene_id']:  # for 3dgs dataset
                    item['scan'] = item['scene_id'].split('/')[1]
                else:  # for vlnce dataset
                    item['scan'] = item['scene_id']
            elif 'scan' in item.keys():  # for grutopia10 dataset
                item['scan'] = item['scan']
            else:
                raise ValueError(f'No scan id found in {item}')

            if dataset_type == 'r2r':
                item['start_position'] = [
                    item['start_position'][0],
                    -item['start_position'][2],
                    item['start_position'][1],
                ]
                item['start_rotation'] = [
                    -item['start_rotation'][3],
                    item['start_rotation'][0],
                    item['start_rotation'][2],
                    -item['start_rotation'][1],
                ]  # [x,y,z,-w] => [w,x,y,z]
                item['c_reference_path'] = []
                if 'reference_path' in item.keys():
                    for path in item['reference_path']:
                        item['c_reference_path'].append([path[0], -path[2], path[1]])
                    item['reference_path'] = item['c_reference_path']
                    del item['c_reference_path']

            if dataset_type == 'kujiale':
                load_data[f'{str(item["trajectory_id"])}_{str(item["episode_id"])}'].append(item)
            else:
                load_data[str(item['trajectory_id'])].append(item)
    if logger is not None:
        logger.info(f'Loaded data with a total of {len(load_data)} items from {split}')
    return load_data
