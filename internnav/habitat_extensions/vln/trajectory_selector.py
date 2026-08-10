"""Lightweight candidate selection and risk estimates for S1 trajectories."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectorySelection:
    selected_index: int
    trajectory: np.ndarray
    dispersion: float
    chunk_size: int
    bimodal: bool
    cluster_sizes: tuple
    clearance: float
    smoothness: float
    depth_risk: bool


class TrajectorySelector:
    """Select a real candidate using consensus, clustering, and cheap safety scores."""

    def __init__(
        self,
        cluster_min_fraction=0.25,
        cluster_lateral_gap=0.25,
        depth_lookahead=1.25,
        unsafe_clearance=0.12,
        near_clearance=0.35,
        horizontal_fov_deg=79.0,
    ):
        self.cluster_min_fraction = float(cluster_min_fraction)
        self.cluster_lateral_gap = float(cluster_lateral_gap)
        self.depth_lookahead = float(depth_lookahead)
        self.unsafe_clearance = float(unsafe_clearance)
        self.near_clearance = float(near_clearance)
        self.horizontal_fov_rad = np.deg2rad(float(horizontal_fov_deg))

    @staticmethod
    def _as_numpy(value):
        value = value.detach() if hasattr(value, "detach") else value
        value = value.float().cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
        return np.asarray(value, dtype=np.float32)

    @classmethod
    def reconstruct(cls, dp_actions):
        deltas = cls._as_numpy(dp_actions).copy()
        if deltas.ndim != 3 or deltas.shape[-1] < 2 or deltas.shape[0] == 0:
            raise ValueError("dp_actions must have shape [candidates, steps, >=2]")
        deltas[:, :, :2] /= 4.0
        positions = np.cumsum(deltas[:, :, :2], axis=1)
        origin = np.zeros((positions.shape[0], 1, 2), dtype=positions.dtype)
        return np.concatenate((origin, positions), axis=1)

    def _two_medoid_clusters(self, pairwise, trajectories):
        count = pairwise.shape[0]
        if count < 4:
            return False, (count, 0)

        medoids = np.unravel_index(np.argmax(pairwise), pairwise.shape)
        assignments = None
        for _ in range(3):
            assignments = np.argmin(pairwise[:, medoids], axis=1)
            updated = []
            for cluster in range(2):
                members = np.flatnonzero(assignments == cluster)
                if members.size == 0:
                    updated.append(medoids[cluster])
                else:
                    costs = pairwise[np.ix_(members, members)].sum(axis=1)
                    updated.append(int(members[np.argmin(costs)]))
            medoids = tuple(updated)

        assignments = np.argmin(pairwise[:, medoids], axis=1)
        sizes = tuple(int(np.sum(assignments == cluster)) for cluster in range(2))
        minimum_size = max(2, int(np.ceil(count * self.cluster_min_fraction)))
        endpoints = trajectories[list(medoids), -1, 1]
        lateral_gap = float(abs(endpoints[0] - endpoints[1]))
        opposite_sides = bool(endpoints[0] * endpoints[1] < 0)
        within = []
        for cluster in range(2):
            members = np.flatnonzero(assignments == cluster)
            within.extend(pairwise[members, medoids[cluster]].tolist())
        within_distance = float(np.mean(within)) if within else 0.0
        separation = float(pairwise[medoids])
        bimodal = (
            min(sizes) >= minimum_size
            and opposite_sides
            and lateral_gap >= self.cluster_lateral_gap
            and separation >= max(0.12, 1.5 * within_distance)
        )
        return bimodal, sizes

    @staticmethod
    def _smoothness(trajectories):
        segments = np.diff(trajectories, axis=1)
        headings = np.unwrap(np.arctan2(segments[:, :, 1], segments[:, :, 0]), axis=1)
        turns = np.diff(headings, axis=1)
        return np.mean(np.abs(turns), axis=1)

    def _depth_clearance(self, trajectories, depth):
        if depth is None:
            return np.full(trajectories.shape[0], np.inf, dtype=np.float32)
        depth = self._as_numpy(depth).squeeze()
        if depth.ndim != 2:
            raise ValueError("depth must be a 2-D image in metres")

        height, width = depth.shape
        row_start, row_stop = int(height * 0.30), max(int(height * 0.82), 1)
        clearance = np.full(trajectories.shape[0], np.inf, dtype=np.float32)
        for candidate_index, trajectory in enumerate(trajectories):
            candidate_clearance = []
            for forward, lateral in trajectory[1:]:
                radial = float(np.hypot(forward, lateral))
                if forward <= 0.05 or radial > self.depth_lookahead:
                    continue
                bearing = float(np.arctan2(lateral, forward))
                if abs(bearing) >= self.horizontal_fov_rad / 2:
                    continue
                column = int(round((0.5 - bearing / self.horizontal_fov_rad) * (width - 1)))
                left, right = max(0, column - 3), min(width, column + 4)
                values = depth[row_start:row_stop, left:right]
                valid = values[np.isfinite(values) & (values > 0.05)]
                if valid.size:
                    ray_depth = float(np.quantile(valid, 0.15))
                    candidate_clearance.append(ray_depth - float(forward))
            if candidate_clearance:
                clearance[candidate_index] = min(candidate_clearance)
        return clearance

    def select(self, dp_actions, depth=None, recent_failure=False):
        trajectories = self.reconstruct(dp_actions)
        flattened = trajectories.reshape(trajectories.shape[0], -1)
        pairwise = np.sqrt(np.mean((flattened[:, None] - flattened[None, :]) ** 2, axis=2))
        centrality = pairwise.mean(axis=1)
        medoid_index = int(np.argmin(centrality))
        dispersion = float(np.sqrt(np.mean((trajectories - trajectories[medoid_index]) ** 2)))
        bimodal, cluster_sizes = self._two_medoid_clusters(pairwise, trajectories)
        smoothness = self._smoothness(trajectories)
        clearance = self._depth_clearance(trajectories, depth)

        selected_index = medoid_index
        if depth is not None:
            centrality_scale = max(float(np.median(centrality)), 1e-6)
            smoothness_scale = max(float(np.median(smoothness)), 1e-6)
            clearance_penalty = np.maximum(self.near_clearance - clearance, 0.0) / self.near_clearance
            scores = centrality / centrality_scale + 0.20 * smoothness / smoothness_scale + 4.0 * clearance_penalty
            selected_index = int(np.argmin(scores))

        selected_clearance = float(clearance[selected_index])
        depth_risk = selected_clearance < self.unsafe_clearance
        if recent_failure or bimodal or depth_risk:
            chunk_size = 1
        elif selected_clearance < self.near_clearance:
            chunk_size = 2
        else:
            chunk_size = 4
        return TrajectorySelection(
            selected_index=selected_index,
            trajectory=trajectories[selected_index],
            dispersion=dispersion,
            chunk_size=chunk_size,
            bimodal=bimodal,
            cluster_sizes=cluster_sizes,
            clearance=selected_clearance,
            smoothness=float(smoothness[selected_index]),
            depth_risk=depth_risk,
        )
