import numpy as np


def build_snapshot_groups(interact_times: np.ndarray):
    """
    Group edge indices by identical interaction timestamps.
    """
    uniq_times = np.unique(interact_times)
    groups = []
    for timestamp in uniq_times:
        indices = np.where(interact_times == timestamp)[0]
        groups.append(indices)
    return uniq_times, groups


def split_snapshot_group(indices: np.ndarray, max_size: int = -1):
    """
    Split a timestamp group into chunks to avoid OOM.
    """
    if max_size is None or max_size <= 0 or len(indices) <= max_size:
        return [indices]
    return [indices[i: i + max_size] for i in range(0, len(indices), max_size)]


def build_snapshot_batches(interact_times: np.ndarray, max_snapshot_batch_size: int = -1):
    """
    Build snapshot batches with optional chunking.
    """
    uniq_times, groups = build_snapshot_groups(interact_times=interact_times)
    batches = []
    for timestamp, group in zip(uniq_times, groups):
        batches.append({
            "time": timestamp,
            "full_indices": group,
            "chunks": split_snapshot_group(group, max_snapshot_batch_size),
        })
    return batches
