import os

import numpy as np
import torch
import torch.nn as nn

from models.modules import HistEmbAggregatorWeightedSum, TRCMemory


class UVProjectionCache(nn.Module):
    def __init__(self, cache_path: str):
        super().__init__()

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f'Cannot find UV projection cache at {cache_path}.')

        cache = np.load(cache_path)
        self.register_buffer('time_values', torch.from_numpy(cache['time_values']).float())
        self.register_buffer('pair_count_hist', torch.from_numpy(cache['pair_count_hist']).float())
        self.register_buffer('node_popularity_hist', torch.from_numpy(cache['node_popularity_hist']).float())
        self.register_buffer('node_profile_hist', torch.from_numpy(cache['node_profile_hist']).float())
        self.register_buffer('global_summary_hist', torch.from_numpy(cache['global_summary_hist']).float())

        self.time_values_numpy = cache['time_values'].astype(np.float64)
        self.time_to_index = {float(time_value): idx for idx, time_value in enumerate(self.time_values_numpy.tolist())}

    def get_time_bucket_indices(self, node_interact_times):
        if torch.is_tensor(node_interact_times):
            flat_times = node_interact_times.detach().cpu().view(-1).tolist()
        else:
            flat_times = np.asarray(node_interact_times).reshape(-1).tolist()

        indices = []
        for interact_time in flat_times:
            interact_time = float(interact_time)
            if interact_time in self.time_to_index:
                indices.append(self.time_to_index[interact_time])
            else:
                indices.append(int(np.searchsorted(self.time_values_numpy, interact_time, side='right') - 1))

        indices = np.clip(indices, 0, len(self.time_values_numpy) - 1)
        return torch.as_tensor(indices, dtype=torch.long, device=self.time_values.device)

    def lookup(self, src_ids, dst_ids, node_interact_times):
        src_ids = torch.as_tensor(src_ids, dtype=torch.long, device=self.time_values.device)
        dst_ids = torch.as_tensor(dst_ids, dtype=torch.long, device=self.time_values.device)
        time_bucket_indices = self.get_time_bucket_indices(node_interact_times=node_interact_times)

        return {
            'time_bucket_indices': time_bucket_indices,
            'pair_count': self.pair_count_hist[time_bucket_indices, src_ids, dst_ids].unsqueeze(dim=1),
            'src_popularity': self.node_popularity_hist[time_bucket_indices, src_ids].unsqueeze(dim=1),
            'dst_popularity': self.node_popularity_hist[time_bucket_indices, dst_ids].unsqueeze(dim=1),
            'src_profile': self.node_profile_hist[time_bucket_indices, src_ids],
            'dst_profile': self.node_profile_hist[time_bucket_indices, dst_ids],
            'global_summary': self.global_summary_hist[time_bucket_indices],
        }


class CoalitionRegimeAdapter(nn.Module):
    def __init__(self, src_emb_dim: int, dst_emb_dim: int, hidden_dim: int, uv_cache_path: str, num_slots: int,
                 profile_rank: int, regime_dim: int, adapter_hidden_dim: int, enable_coalition: bool = True,
                 enable_regime: bool = True):
        super().__init__()

        self.enable_coalition = enable_coalition
        self.enable_regime = enable_regime
        self.projection_cache = UVProjectionCache(cache_path=uv_cache_path)

        self.profile_encoder = nn.Sequential(
            nn.Linear(profile_rank, adapter_hidden_dim),
            nn.ReLU(),
            nn.Linear(adapter_hidden_dim, adapter_hidden_dim),
            nn.ReLU(),
        )
        self.slot_projection = nn.Linear(adapter_hidden_dim, num_slots)

        regime_input_dim = self.projection_cache.global_summary_hist.shape[1]
        self.regime_encoder = nn.GRU(input_size=regime_input_dim, hidden_size=regime_dim, batch_first=True)

        coalition_feature_dim = 0
        if self.enable_coalition:
            coalition_feature_dim = 2 * num_slots + 2 * adapter_hidden_dim

        residual_input_dim = src_emb_dim + dst_emb_dim
        if self.enable_coalition:
            residual_input_dim += coalition_feature_dim
        if self.enable_regime:
            residual_input_dim += regime_dim

        self.use_residual = self.enable_coalition or self.enable_regime
        self.delta_head = nn.Sequential(
            nn.Linear(residual_input_dim, adapter_hidden_dim),
            nn.ReLU(),
            nn.Linear(adapter_hidden_dim, hidden_dim),
        )
        self.bias_head = nn.Linear(3, 1)

    def forward(self, src_ids, dst_ids, node_interact_times, src_emb: torch.Tensor, dst_emb: torch.Tensor):
        cache_outputs = self.projection_cache.lookup(src_ids=src_ids, dst_ids=dst_ids, node_interact_times=node_interact_times)

        stats = torch.log1p(torch.cat(
            [cache_outputs['pair_count'], cache_outputs['src_popularity'], cache_outputs['dst_popularity']],
            dim=1,
        ))

        if self.use_residual:
            residual_inputs = [src_emb, dst_emb]

            if self.enable_coalition:
                src_profile = self.profile_encoder(cache_outputs['src_profile'])
                dst_profile = self.profile_encoder(cache_outputs['dst_profile'])
                src_slots = torch.softmax(self.slot_projection(src_profile), dim=1)
                dst_slots = torch.softmax(self.slot_projection(dst_profile), dim=1)

                coalition_features = torch.cat(
                    [
                        src_slots * dst_slots,
                        torch.abs(src_slots - dst_slots),
                        src_profile * dst_profile,
                        torch.abs(src_profile - dst_profile),
                    ],
                    dim=1,
                )
                residual_inputs.append(coalition_features)

            if self.enable_regime:
                regime_states, _ = self.regime_encoder(self.projection_cache.global_summary_hist.unsqueeze(dim=0))
                regime_features = regime_states.squeeze(dim=0)[cache_outputs['time_bucket_indices']]
                residual_inputs.append(regime_features)

            delta_uv = self.delta_head(torch.cat(residual_inputs, dim=1))
        else:
            delta_uv = torch.zeros_like(src_emb)

        bias_uv = self.bias_head(stats)
        return delta_uv, bias_uv


class CoRAHistoricalDecoder(nn.Module):
    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int, device='cpu', gamma=0.9,
                 uv_cache_path: str = './processed_data/UNvote/uv_projection_cache.npz', num_slots: int = 8,
                 profile_rank: int = 16, regime_dim: int = 32, adapter_hidden_dim: int = 172,
                 symmetric_pairs: bool = True, enable_coalition: bool = True, enable_regime: bool = True):
        super().__init__()

        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)
        self.act = nn.ReLU()

        self.device = device
        self.symmetric_pairs = symmetric_pairs

        self.aggregate_hist_emb = HistEmbAggregatorWeightedSum(gamma=gamma)
        self.hist_emb_aggregator = self.aggregate_hist_emb.aggregator_name

        self.historical_interaction_memory = TRCMemory(dim=input_dim1, device=device)
        self.adapter = CoalitionRegimeAdapter(
            src_emb_dim=input_dim1,
            dst_emb_dim=input_dim2,
            hidden_dim=hidden_dim,
            uv_cache_path=uv_cache_path,
            num_slots=num_slots,
            profile_rank=profile_rank,
            regime_dim=regime_dim,
            adapter_hidden_dim=adapter_hidden_dim,
            enable_coalition=enable_coalition,
            enable_regime=enable_regime,
        )

    def get_memory_keys(self, src_ids, dst_ids):
        if torch.is_tensor(src_ids):
            src_ids = src_ids.detach().cpu().tolist()
        else:
            src_ids = np.asarray(src_ids).tolist()

        if torch.is_tensor(dst_ids):
            dst_ids = dst_ids.detach().cpu().tolist()
        else:
            dst_ids = np.asarray(dst_ids).tolist()

        keys = []
        for src_id, dst_id in zip(src_ids, dst_ids):
            src_id, dst_id = int(src_id), int(dst_id)
            if self.symmetric_pairs:
                src_id, dst_id = min(src_id, dst_id), max(src_id, dst_id)
            keys.append((src_id, dst_id))
        return keys

    def forward(self, src_ids, dst_ids, src_emb: torch.Tensor, dst_emb: torch.Tensor, node_interact_times=None, update_memories=False):
        if node_interact_times is None:
            raise ValueError('CoRAHistoricalDecoder requires node_interact_times for UV projection cache lookup.')

        x = torch.cat([src_emb, dst_emb], dim=1)
        h_current = self.act(self.fc1(x))

        delta_uv, bias_uv = self.adapter(
            src_ids=src_ids,
            dst_ids=dst_ids,
            node_interact_times=node_interact_times,
            src_emb=src_emb,
            dst_emb=dst_emb,
        )
        h_current = h_current + delta_uv

        keys = self.get_memory_keys(src_ids=src_ids, dst_ids=dst_ids)
        hist_emb_most_recent_one = self.historical_interaction_memory.get_memories(keys)
        h_most_recent_emb = torch.stack(hist_emb_most_recent_one, dim=0).to(self.device)

        h_historical = self.aggregate_hist_emb(current_emb=h_current, hist_emb=h_most_recent_emb)

        if self.hist_emb_aggregator == 'weighted_sum':
            updated_weighted_hist_emb = h_historical
            h_historical = h_most_recent_emb
        else:
            raise Exception('Invalid TRC aggregator')

        h = self.fc2(torch.cat((h_current, h_historical), dim=1)) + bias_uv

        if update_memories:
            if self.hist_emb_aggregator == 'weighted_sum':
                self.historical_interaction_memory.update_memories(keys, updated_weighted_hist_emb.detach().cpu())
            else:
                raise Exception('Invalid TRC aggregator when updating TRC memory')
        return h
