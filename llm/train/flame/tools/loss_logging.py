import torch
import torch.distributed as dist
from torchtitan.distributed import utils as dist_utils


def _mean_over_mesh(tensor: torch.Tensor, mesh) -> torch.Tensor:
    if mesh is None or mesh.size() <= 1:
        return tensor
    mean = tensor.clone()
    dist.all_reduce(mean, op=dist.ReduceOp.SUM, group=mesh.get_group())
    return mean / mesh.size()


def _sum_over_mesh(tensor: torch.Tensor, mesh) -> torch.Tensor:
    if mesh is None or mesh.size() <= 1:
        return tensor
    summed = tensor.clone()
    dist.all_reduce(summed, op=dist.ReduceOp.SUM, group=mesh.get_group())
    return summed


def _data_parallel_mean_max(
    metric_loss: torch.Tensor, parallel_dims, world_mesh
) -> tuple[float, float]:
    if (
        parallel_dims.dp_replicate_enabled
        or parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
    ):
        return (
            dist_utils.dist_mean(metric_loss, world_mesh["dp_cp"]),
            dist_utils.dist_max(metric_loss, world_mesh["dp_cp"]),
        )

    value = metric_loss.item()
    return value, value


def global_loss_metrics_for_logging(
    loss: torch.Tensor, parallel_dims, world_mesh
) -> tuple[float, float]:
    """Return TP-invariant per-token CE metrics for comparing runs."""
    metric_loss = loss.detach()
    if getattr(parallel_dims, "tp_enabled", False):
        metric_loss = _sum_over_mesh(metric_loss, world_mesh["tp"])

    return _data_parallel_mean_max(metric_loss, parallel_dims, world_mesh)
