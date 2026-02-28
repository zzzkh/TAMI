from __future__ import annotations

from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F


def _as_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError(f'Unsupported type: {type(x)}')


class TimeGapTracker:
    def __init__(self) -> None:
        self._last_t: Dict[int, float] = {}

    def reset(self) -> None:
        self._last_t.clear()

    def get_gaps(self, node_ids: np.ndarray, times: np.ndarray) -> np.ndarray:
        node_ids = _as_numpy(node_ids).astype(np.int64, copy=False)
        times = _as_numpy(times).astype(np.float64, copy=False)

        gaps = np.zeros_like(times, dtype=np.float64)
        for i, (n, t) in enumerate(zip(node_ids, times)):
            last = self._last_t.get(int(n), None)
            if last is None:
                gaps[i] = 0.0
            else:
                dt = float(t) - float(last)
                gaps[i] = dt if dt > 0 else 0.0
        return gaps

    def update(self, src_ids: np.ndarray, dst_ids: np.ndarray, times: np.ndarray) -> None:
        src_ids = _as_numpy(src_ids).astype(np.int64, copy=False)
        dst_ids = _as_numpy(dst_ids).astype(np.int64, copy=False)
        times = _as_numpy(times).astype(np.float64, copy=False)

        for u, v, t in zip(src_ids, dst_ids, times):
            tt = float(t)
            self._last_t[int(u)] = tt
            self._last_t[int(v)] = tt


class AdaptiveTemperature:
    def __init__(
        self,
        tracker: TimeGapTracker,
        base_tau: float = 0.07,
        tau_alpha: float = 0.25,
        tau_min: float = 0.03,
        tau_max: float = 0.20,
        device: str | torch.device = 'cpu',
    ) -> None:
        assert base_tau > 0
        assert tau_min > 0
        assert tau_max >= tau_min
        self.tracker = tracker
        self.base_tau = float(base_tau)
        self.tau_alpha = float(tau_alpha)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.device = device

    @torch.no_grad()
    def __call__(self, src_node_ids: np.ndarray, times: np.ndarray) -> torch.Tensor:
        gaps = self.tracker.get_gaps(src_node_ids, times)
        gap_feat = np.log1p(gaps)
        tau = self.base_tau * (1.0 + self.tau_alpha * gap_feat)
        tau = np.clip(tau, self.tau_min, self.tau_max).astype(np.float32)
        return torch.from_numpy(tau).to(self.device)


class TemporalNeighborNegativeSampler:
    def __init__(
        self,
        base_negative_sampler,
        neighbor_sampler,
        num_negatives: int = 10,
        hard_ratio: float = 0.5,
        neighbor_k: int = 20,
        max_resample: int = 20,
    ) -> None:
        assert num_negatives >= 1
        assert 0.0 <= hard_ratio <= 1.0
        assert neighbor_k >= 1

        self.base_negative_sampler = base_negative_sampler
        self.neighbor_sampler = neighbor_sampler
        self.num_negatives = int(num_negatives)
        self.hard_ratio = float(hard_ratio)
        self.neighbor_k = int(neighbor_k)
        self.max_resample = int(max_resample)

        self._unique_dst = getattr(base_negative_sampler, 'unique_dst_node_ids', None)
        if self._unique_dst is None:
            raise ValueError('base_negative_sampler must expose unique_dst_node_ids.')

        self._rng = getattr(base_negative_sampler, 'random_state', np.random)

    def _sample_random_dst(self, size: int) -> np.ndarray:
        idx = self._rng.randint(0, len(self._unique_dst), size=size)
        return self._unique_dst[idx].astype(np.int64, copy=False)

    def sample(
        self,
        batch_src_node_ids: np.ndarray,
        batch_dst_node_ids: np.ndarray,
        batch_node_interact_times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_src_node_ids = _as_numpy(batch_src_node_ids).astype(np.int64, copy=False)
        batch_dst_node_ids = _as_numpy(batch_dst_node_ids).astype(np.int64, copy=False)
        batch_node_interact_times = _as_numpy(batch_node_interact_times).astype(np.float64, copy=False)

        B = len(batch_src_node_ids)
        K = self.num_negatives
        num_hard = int(round(K * self.hard_ratio))
        num_hard = min(max(num_hard, 0), K)
        num_rand = K - num_hard

        neighbor_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(
            node_ids=batch_src_node_ids,
            node_interact_times=batch_node_interact_times,
            num_neighbors=self.neighbor_k,
        )
        neighbor_ids = neighbor_ids.astype(np.int64, copy=False)

        neg_dst_mat = np.full((B, K), fill_value=-1, dtype=np.int64)

        for i in range(B):
            u = int(batch_src_node_ids[i])
            pos_v = int(batch_dst_node_ids[i])

            row = neighbor_ids[i]
            seen = set()
            hard_list = []
            for v in row[::-1]:
                vv = int(v)
                if vv == 0:
                    continue
                if vv == pos_v or vv == u:
                    continue
                if vv in seen:
                    continue
                seen.add(vv)
                hard_list.append(vv)
                if len(hard_list) >= num_hard:
                    break

            take = min(len(hard_list), num_hard)
            if take > 0:
                neg_dst_mat[i, :take] = hard_list[:take]

        need_missing_hard = int((neg_dst_mat[:, :num_hard] == -1).sum()) if num_hard > 0 else 0
        need_rand_slots = B * num_rand
        need_total = need_missing_hard + need_rand_slots

        if need_total > 0:
            rand_pool = self._sample_random_dst(size=need_total)
            ptr = 0

            for i in range(B):
                pos_v = int(batch_dst_node_ids[i])

                if num_hard > 0:
                    miss_idx = np.where(neg_dst_mat[i, :num_hard] == -1)[0]
                    for j in miss_idx:
                        v = int(rand_pool[ptr]); ptr += 1
                        tries = 0
                        while v == pos_v and tries < self.max_resample:
                            v = int(self._sample_random_dst(size=1)[0])
                            tries += 1
                        neg_dst_mat[i, j] = v

                if num_rand > 0:
                    for j in range(num_hard, K):
                        v = int(rand_pool[ptr]); ptr += 1
                        tries = 0
                        while v == pos_v and tries < self.max_resample:
                            v = int(self._sample_random_dst(size=1)[0])
                            tries += 1
                        neg_dst_mat[i, j] = v

        if np.any(neg_dst_mat == -1):
            miss = np.where(neg_dst_mat == -1)
            neg_dst_mat[miss] = self._sample_random_dst(size=len(miss[0]))

        for i in range(B):
            self._rng.shuffle(neg_dst_mat[i])

        neg_src_mat = np.repeat(batch_src_node_ids.reshape(B, 1), K, axis=1)
        neg_t_mat = np.repeat(batch_node_interact_times.reshape(B, 1), K, axis=1)

        neg_src_flat = neg_src_mat.reshape(-1).astype(np.int64, copy=False)
        neg_dst_flat = neg_dst_mat.reshape(-1).astype(np.int64, copy=False)
        neg_t_flat = neg_t_mat.reshape(-1).astype(np.float64, copy=False)

        return neg_src_flat, neg_dst_flat, neg_t_flat, neg_dst_mat


class HALT:
    def __init__(
        self,
        base_negative_sampler,
        neighbor_sampler,
        num_negatives: int = 10,
        hard_ratio: float = 0.5,
        neighbor_k: int = 20,
        base_tau: float = 0.07,
        tau_alpha: float = 0.25,
        tau_min: float = 0.03,
        tau_max: float = 0.20,
        device: str | torch.device = 'cpu',
    ) -> None:
        self.device = device
        self.tracker = TimeGapTracker()
        self.temperature = AdaptiveTemperature(
            tracker=self.tracker,
            base_tau=base_tau,
            tau_alpha=tau_alpha,
            tau_min=tau_min,
            tau_max=tau_max,
            device=device,
        )
        self.neg_sampler = TemporalNeighborNegativeSampler(
            base_negative_sampler=base_negative_sampler,
            neighbor_sampler=neighbor_sampler,
            num_negatives=num_negatives,
            hard_ratio=hard_ratio,
            neighbor_k=neighbor_k,
        )

    def reset_state(self) -> None:
        self.tracker.reset()

    def sample_negatives(self, batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times):
        return self.neg_sampler.sample(batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times)

    @torch.no_grad()
    def compute_temperature(self, batch_src_node_ids, batch_node_interact_times) -> torch.Tensor:
        return self.temperature(batch_src_node_ids, batch_node_interact_times)

    def listwise_loss(
        self,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if negative_logits.dim() == 1:
            B = positive_logits.shape[0]
            K = negative_logits.shape[0] // B
            negative_logits = negative_logits.view(B, K)

        B = positive_logits.shape[0]
        if temperature is None:
            temperature = torch.ones((B,), device=positive_logits.device, dtype=positive_logits.dtype)
        elif temperature.dim() == 0:
            temperature = temperature.expand(B)

        logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
        logits = logits / temperature.unsqueeze(1)
        labels = torch.zeros((B,), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def update_state(self, batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times) -> None:
        self.tracker.update(batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times)
