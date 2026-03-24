import argparse
import csv
import os

import numpy as np


DATASET_CONFIGS = {
    'UNvote': {
        'symmetric_pairs': True,
        'global_topk': 4,
        'cache_filename': 'uv_projection_cache.npz',
    },
    'USLegis': {
        'symmetric_pairs': True,
        'global_topk': 4,
        'cache_filename': 'us_projection_cache.npz',
    },
    'CanParl': {
        'symmetric_pairs': True,
        'global_topk': 4,
        'cache_filename': 'cp_projection_cache.npz',
    },
    'mooc': {
        'symmetric_pairs': True,
        'global_topk': 4,
        'cache_filename': 'mo_projection_cache.npz',
        'default_num_time_buckets': 16,
        'scalable_mode': True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser('Build history-only projection cache for supported datasets.')
    parser.add_argument('--dataset_name', type=str, default='UNvote', choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument('--input_csv_path', type=str, default=None)
    parser.add_argument('--input_node_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--global_topk', type=int, default=None)
    parser.add_argument(
        '--num_time_buckets',
        type=int,
        default=None,
        help='Optional number of temporal buckets. Required for large datasets such as mooc.',
    )
    return parser.parse_args()


def get_default_paths(args):
    dataset_dir = os.path.join('processed_data', args.dataset_name)
    dataset_config = DATASET_CONFIGS[args.dataset_name]
    input_csv_path = args.input_csv_path or os.path.join(dataset_dir, f'ml_{args.dataset_name}.csv')
    input_node_path = args.input_node_path or os.path.join(dataset_dir, f'ml_{args.dataset_name}_node.npy')
    output_path = args.output_path or os.path.join(dataset_dir, dataset_config['cache_filename'])
    return input_csv_path, input_node_path, output_path


def load_interactions(csv_path):
    interactions = []
    with open(csv_path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            interactions.append((float(row['ts']), int(row['idx']), int(row['u']), int(row['i'])))

    interactions.sort(key=lambda item: (item[0], item[1]))
    return interactions


def build_low_rank_profile(adj_hist: np.ndarray, rank: int):
    if rank <= 0:
        return np.zeros((adj_hist.shape[0], 0), dtype=np.float32)

    features = np.log1p(adj_hist).astype(np.float64, copy=False)
    row_sums = features.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(features)
    np.divide(features, np.clip(row_sums, a_min=1.0, a_max=None), out=normalized, where=row_sums > 0)

    if not np.any(normalized):
        return np.zeros((adj_hist.shape[0], rank), dtype=np.float32)

    try:
        left_singular_vectors, singular_values, _ = np.linalg.svd(normalized, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((adj_hist.shape[0], rank), dtype=np.float32)

    effective_rank = min(rank, left_singular_vectors.shape[1], singular_values.shape[0])
    profile = np.zeros((adj_hist.shape[0], rank), dtype=np.float32)
    if effective_rank > 0:
        profile[:, :effective_rank] = (
            left_singular_vectors[:, :effective_rank] * singular_values[:effective_rank]
        ).astype(np.float32)
    return profile


def build_global_summary(adj_hist: np.ndarray, topk: int):
    binary_adj = (adj_hist > 0).astype(np.float32)
    log_adj = np.log1p(adj_hist).astype(np.float64, copy=False)
    num_nodes = adj_hist.shape[0]

    upper_triangle_edges = float(np.triu(binary_adj, k=1).sum())
    possible_edges = max(num_nodes * (num_nodes - 1) / 2.0, 1.0)
    degrees = binary_adj.sum(axis=1)

    try:
        singular_values = np.linalg.svd(log_adj, compute_uv=False)
    except np.linalg.LinAlgError:
        singular_values = np.zeros(topk, dtype=np.float64)

    summary = np.zeros(4 + topk, dtype=np.float32)
    summary[0] = upper_triangle_edges / possible_edges
    summary[1] = degrees.mean() if len(degrees) > 0 else 0.0
    summary[2] = degrees.std() if len(degrees) > 0 else 0.0
    summary[3] = np.log1p(upper_triangle_edges)

    effective_topk = min(topk, singular_values.shape[0])
    if effective_topk > 0:
        summary[4: 4 + effective_topk] = singular_values[:effective_topk].astype(np.float32)
    return summary


def build_static_node_profiles(node_raw_features: np.ndarray, rank: int):
    if rank <= 0:
        return np.zeros((node_raw_features.shape[0], 0), dtype=np.float32)

    features = np.asarray(node_raw_features, dtype=np.float32)
    if features.ndim == 1:
        features = features[:, None]

    features = features - features.mean(axis=0, keepdims=True)
    if not np.any(features):
        return np.zeros((features.shape[0], rank), dtype=np.float32)

    try:
        left_singular_vectors, singular_values, _ = np.linalg.svd(features, full_matrices=False)
        effective_rank = min(rank, left_singular_vectors.shape[1], singular_values.shape[0])
        profile = np.zeros((features.shape[0], rank), dtype=np.float32)
        if effective_rank > 0:
            profile[:, :effective_rank] = (
                left_singular_vectors[:, :effective_rank] * singular_values[:effective_rank]
            ).astype(np.float32)
        return profile
    except np.linalg.LinAlgError:
        effective_rank = min(rank, features.shape[1])
        profile = np.zeros((features.shape[0], rank), dtype=np.float32)
        profile[:, :effective_rank] = features[:, :effective_rank]
        return profile


def build_global_summary_fast(adj_hist: np.ndarray, topk: int):
    binary_adj = (adj_hist > 0).astype(np.float32)
    num_nodes = adj_hist.shape[0]
    upper_triangle_edges = float(np.triu(binary_adj, k=1).sum())
    possible_edges = max(num_nodes * (num_nodes - 1) / 2.0, 1.0)
    degrees = binary_adj.sum(axis=1)

    summary = np.zeros(4 + topk, dtype=np.float32)
    summary[0] = upper_triangle_edges / possible_edges
    summary[1] = degrees.mean() if len(degrees) > 0 else 0.0
    summary[2] = degrees.std() if len(degrees) > 0 else 0.0
    summary[3] = np.log1p(upper_triangle_edges)

    if topk > 0 and len(degrees) > 0:
        top_values = np.sort(degrees)[-topk:][::-1]
        summary[4: 4 + len(top_values)] = top_values.astype(np.float32) / max(num_nodes - 1, 1)
    return summary


def update_pair_counts(adj_matrix: np.ndarray, src_node_id: int, dst_node_id: int, symmetric_pairs: bool):
    if symmetric_pairs:
        src_node_id, dst_node_id = sorted((src_node_id, dst_node_id))

    if src_node_id == dst_node_id:
        return

    adj_matrix[src_node_id, dst_node_id] += 1.0
    if symmetric_pairs:
        adj_matrix[dst_node_id, src_node_id] += 1.0


def build_exact_cache(interactions, node_raw_features, args, dataset_config, global_topk):
    max_node_id = max(max(src_node_id, dst_node_id) for _, _, src_node_id, dst_node_id in interactions)
    num_nodes = max(max_node_id + 1, node_raw_features.shape[0])
    unique_times = sorted({interaction_time for interaction_time, _, _, _ in interactions})

    pair_count_hist = np.zeros((len(unique_times), num_nodes, num_nodes), dtype=np.float32)
    node_popularity_hist = np.zeros((len(unique_times), num_nodes), dtype=np.float32)
    node_profile_hist = np.zeros((len(unique_times), num_nodes, args.rank), dtype=np.float32)
    global_summary_hist = np.zeros((len(unique_times), 4 + global_topk), dtype=np.float32)

    historical_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    interaction_pointer = 0
    for time_index, current_time in enumerate(unique_times):
        pair_count_hist[time_index] = historical_adj
        node_popularity_hist[time_index] = historical_adj.sum(axis=1)
        node_profile_hist[time_index] = build_low_rank_profile(historical_adj, rank=args.rank)
        global_summary_hist[time_index] = build_global_summary(historical_adj, topk=global_topk)

        while interaction_pointer < len(interactions) and interactions[interaction_pointer][0] == current_time:
            _, _, src_node_id, dst_node_id = interactions[interaction_pointer]
            update_pair_counts(
                historical_adj,
                src_node_id=src_node_id,
                dst_node_id=dst_node_id,
                symmetric_pairs=dataset_config['symmetric_pairs'],
            )
            interaction_pointer += 1

    return np.array(unique_times, dtype=np.float64), pair_count_hist, node_popularity_hist, node_profile_hist, global_summary_hist


def build_bucketed_cache(interactions, node_raw_features, args, dataset_config, global_topk):
    max_node_id = max(max(src_node_id, dst_node_id) for _, _, src_node_id, dst_node_id in interactions)
    num_nodes = max(max_node_id + 1, node_raw_features.shape[0])
    bucket_count = args.num_time_buckets or dataset_config.get('default_num_time_buckets')
    if bucket_count is None or bucket_count <= 0:
        raise ValueError('num_time_buckets must be a positive integer for scalable_mode datasets.')

    min_time = float(interactions[0][0])
    max_time = float(interactions[-1][0])
    if max_time <= min_time:
        bucket_starts = np.array([min_time], dtype=np.float64)
    else:
        bucket_edges = np.linspace(min_time, max_time + 1e-6, num=bucket_count + 1, dtype=np.float64)
        bucket_starts = bucket_edges[:-1]

    pair_count_hist = np.zeros((len(bucket_starts), num_nodes, num_nodes), dtype=np.float16)
    node_popularity_hist = np.zeros((len(bucket_starts), num_nodes), dtype=np.float32)
    static_profiles = build_static_node_profiles(node_raw_features=node_raw_features, rank=args.rank)
    node_profile_hist = np.repeat(static_profiles[None, :, :], len(bucket_starts), axis=0).astype(np.float32)
    global_summary_hist = np.zeros((len(bucket_starts), 4 + global_topk), dtype=np.float32)

    historical_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    interaction_pointer = 0
    for bucket_index, bucket_start in enumerate(bucket_starts):
        while interaction_pointer < len(interactions) and interactions[interaction_pointer][0] < bucket_start:
            _, _, src_node_id, dst_node_id = interactions[interaction_pointer]
            update_pair_counts(
                historical_adj,
                src_node_id=src_node_id,
                dst_node_id=dst_node_id,
                symmetric_pairs=dataset_config['symmetric_pairs'],
            )
            interaction_pointer += 1

        pair_count_hist[bucket_index] = historical_adj.astype(np.float16, copy=False)
        node_popularity_hist[bucket_index] = historical_adj.sum(axis=1, dtype=np.float32)
        global_summary_hist[bucket_index] = build_global_summary_fast(historical_adj, topk=global_topk)

    return bucket_starts, pair_count_hist, node_popularity_hist, node_profile_hist, global_summary_hist


def main():
    args = parse_args()
    dataset_config = DATASET_CONFIGS[args.dataset_name]
    input_csv_path, input_node_path, output_path = get_default_paths(args)

    interactions = load_interactions(input_csv_path)
    node_raw_features = np.load(input_node_path)

    if len(interactions) == 0:
        raise ValueError(f'No interactions were loaded from {input_csv_path}!')

    global_topk = args.global_topk if args.global_topk is not None else dataset_config['global_topk']

    if dataset_config.get('scalable_mode', False):
        time_values, pair_count_hist, node_popularity_hist, node_profile_hist, global_summary_hist = build_bucketed_cache(
            interactions=interactions,
            node_raw_features=node_raw_features,
            args=args,
            dataset_config=dataset_config,
            global_topk=global_topk,
        )
    else:
        time_values, pair_count_hist, node_popularity_hist, node_profile_hist, global_summary_hist = build_exact_cache(
            interactions=interactions,
            node_raw_features=node_raw_features,
            args=args,
            dataset_config=dataset_config,
            global_topk=global_topk,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        time_values=np.array(time_values, dtype=np.float64),
        pair_count_hist=pair_count_hist,
        node_popularity_hist=node_popularity_hist,
        node_profile_hist=node_profile_hist,
        global_summary_hist=global_summary_hist,
    )

    print(f'Saved projection cache to {output_path}.')
    print(
        f'Dataset: {args.dataset_name}, Time buckets: {len(time_values)}, rank: {args.rank}, '
        f'symmetric_pairs: {dataset_config["symmetric_pairs"]}, scalable_mode: {dataset_config.get("scalable_mode", False)}.',
    )


if __name__ == '__main__':
    main()
