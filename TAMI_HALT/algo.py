from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _as_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported type: {type(x)}")


class TimeGapTracker:
    def __init__(self) -> None:
        self._last_t: Dict[int, float] = {}

    def reset(self) -> None:
        self._last_t.clear()

    def get_gaps(self, node_ids: np.ndarray, times: np.ndarray) -> np.ndarray:
        node_ids = _as_numpy(node_ids).astype(np.int64, copy=False)
        times = _as_numpy(times).astype(np.float64, copy=False)
        gaps = np.zeros_like(times, dtype=np.float64)
        for idx, (node_id, ts) in enumerate(zip(node_ids, times)):
            last_t = self._last_t.get(int(node_id), None)
            if last_t is None:
                gaps[idx] = 0.0
            else:
                dt = float(ts) - float(last_t)
                gaps[idx] = dt if dt > 0 else 0.0
        return gaps

    def update(self, src_ids: np.ndarray, dst_ids: np.ndarray, times: np.ndarray) -> None:
        src_ids = _as_numpy(src_ids).astype(np.int64, copy=False)
        dst_ids = _as_numpy(dst_ids).astype(np.int64, copy=False)
        times = _as_numpy(times).astype(np.float64, copy=False)
        for src_id, dst_id, ts in zip(src_ids, dst_ids, times):
            t = float(ts)
            self._last_t[int(src_id)] = t
            self._last_t[int(dst_id)] = t


class AdaptiveTemperature:
    def __init__(
        self,
        tracker: TimeGapTracker,
        base_tau: float = 0.07,
        tau_alpha: float = 0.25,
        tau_min: float = 0.03,
        tau_max: float = 0.20,
        device: str | torch.device = "cpu",
    ) -> None:
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
        self.base_negative_sampler = base_negative_sampler
        self.neighbor_sampler = neighbor_sampler
        self.num_negatives = int(num_negatives)
        self.hard_ratio = float(hard_ratio)
        self.neighbor_k = int(neighbor_k)
        self.max_resample = int(max_resample)

        self._unique_dst = getattr(base_negative_sampler, "unique_dst_node_ids", None)
        if self._unique_dst is None:
            raise ValueError("base_negative_sampler must expose unique_dst_node_ids")

        self._rng = getattr(base_negative_sampler, "random_state", np.random)

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

        batch_size = len(batch_src_node_ids)
        num_negatives = self.num_negatives
        num_hard = int(round(num_negatives * self.hard_ratio))
        num_hard = min(max(num_hard, 0), num_negatives)
        num_rand = num_negatives - num_hard

        neighbor_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(
            node_ids=batch_src_node_ids,
            node_interact_times=batch_node_interact_times,
            num_neighbors=self.neighbor_k,
        )
        neighbor_ids = neighbor_ids.astype(np.int64, copy=False)

        neg_dst_mat = np.full((batch_size, num_negatives), fill_value=-1, dtype=np.int64)

        for row_idx in range(batch_size):
            src_id = int(batch_src_node_ids[row_idx])
            pos_dst_id = int(batch_dst_node_ids[row_idx])
            row_neighbors = neighbor_ids[row_idx]
            seen = set()
            hard_list = []
            for dst_id in row_neighbors[::-1]:
                cand_dst_id = int(dst_id)
                if cand_dst_id == 0:
                    continue
                if cand_dst_id == pos_dst_id or cand_dst_id == src_id:
                    continue
                if cand_dst_id in seen:
                    continue
                seen.add(cand_dst_id)
                hard_list.append(cand_dst_id)
                if len(hard_list) >= num_hard:
                    break

            take = min(len(hard_list), num_hard)
            if take > 0:
                neg_dst_mat[row_idx, :take] = hard_list[:take]

        need_missing_hard = int((neg_dst_mat[:, :num_hard] == -1).sum()) if num_hard > 0 else 0
        need_rand_slots = batch_size * num_rand
        need_total = need_missing_hard + need_rand_slots

        if need_total > 0:
            rand_pool = self._sample_random_dst(size=need_total)
            ptr = 0
            for row_idx in range(batch_size):
                pos_dst_id = int(batch_dst_node_ids[row_idx])
                if num_hard > 0:
                    miss_idx = np.where(neg_dst_mat[row_idx, :num_hard] == -1)[0]
                    for col_idx in miss_idx:
                        dst_id = int(rand_pool[ptr])
                        ptr += 1
                        tries = 0
                        while dst_id == pos_dst_id and tries < self.max_resample:
                            dst_id = int(self._sample_random_dst(size=1)[0])
                            tries += 1
                        neg_dst_mat[row_idx, col_idx] = dst_id
                if num_rand > 0:
                    for col_idx in range(num_hard, num_negatives):
                        dst_id = int(rand_pool[ptr])
                        ptr += 1
                        tries = 0
                        while dst_id == pos_dst_id and tries < self.max_resample:
                            dst_id = int(self._sample_random_dst(size=1)[0])
                            tries += 1
                        neg_dst_mat[row_idx, col_idx] = dst_id

        if np.any(neg_dst_mat == -1):
            miss = np.where(neg_dst_mat == -1)
            neg_dst_mat[miss] = self._sample_random_dst(size=len(miss[0]))

        for row_idx in range(batch_size):
            self._rng.shuffle(neg_dst_mat[row_idx])

        neg_src_mat = np.repeat(batch_src_node_ids.reshape(batch_size, 1), num_negatives, axis=1)
        neg_t_mat = np.repeat(batch_node_interact_times.reshape(batch_size, 1), num_negatives, axis=1)

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
        device: str | torch.device = "cpu",
    ) -> None:
        self.neg_sampler = TemporalNeighborNegativeSampler(
            base_negative_sampler=base_negative_sampler,
            neighbor_sampler=neighbor_sampler,
            num_negatives=num_negatives,
            hard_ratio=hard_ratio,
            neighbor_k=neighbor_k,
        )
        self.tracker = TimeGapTracker()
        self.temperature = AdaptiveTemperature(
            tracker=self.tracker,
            base_tau=base_tau,
            tau_alpha=tau_alpha,
            tau_min=tau_min,
            tau_max=tau_max,
            device=device,
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
            batch_size = positive_logits.shape[0]
            num_negatives = negative_logits.shape[0] // batch_size
            negative_logits = negative_logits.view(batch_size, num_negatives)

        batch_size = positive_logits.shape[0]
        if temperature is None:
            temperature = torch.ones((batch_size,), device=positive_logits.device, dtype=positive_logits.dtype)
        elif temperature.dim() == 0:
            temperature = temperature.expand(batch_size)

        logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
        logits = logits / temperature.unsqueeze(1)
        labels = torch.zeros((batch_size,), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def update_state(self, batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times) -> None:
        self.tracker.update(batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times)

