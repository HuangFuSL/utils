'''
`utils.ctorch.metrics` - Utility functions for computing metrics in PyTorch
'''

import abc
from typing import Generator, Tuple
import torch

from ..cprint import cformat

def auc_score(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    Compute the Area Under the Curve (AUC) for binary classification.

    Args:
        y_true (torch.Tensor): Ground truth binary labels (0 or 1).
        y_score (torch.Tensor): Predicted scores or probabilities.

    Returns:
        float: The computed AUC value.
    """
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape.")

    y_true = y_true.view(-1)
    y_score = y_score.view(-1)

    total_pairs = torch.sum(y_true == 1) * torch.sum(y_true == 0)
    if total_pairs == 0:
        return 0.0

    sorted_indices = torch.argsort(y_score)
    y_true_sorted = y_true[sorted_indices]

    cum_negative = torch.cumsum(1 - y_true_sorted, dim=0)
    true_indices = y_true_sorted == 1
    return (torch.sum(cum_negative[true_indices]) / total_pairs).item()

def hit_rate(y_true: torch.Tensor, y_score: torch.Tensor, k: int | None = None) -> float:
    """
    Compute the hit rate for retrieval tasks.

    Args:
        y_true (torch.Tensor): Shape (N, C), ground truth binary labels (0 or 1).
        y_score (torch.Tensor): Shape (N, C), predicted scores or probabilities.
        k (int): The number of top predictions to consider.

    Returns:
        float: The computed hit rate.
    """
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape.")
    if k is None:
        k = y_score.shape[1]
    if k <= 0 or k > y_score.shape[1]:
        raise ValueError(f"k must be in the range [1, {y_score.shape[1]}].")
    if torch.any(torch.sum(y_true, dim=1) != 1):
        raise ValueError("Each sample in y_true must have exactly one positive label.")

    _, top_k_indices = torch.topk(y_score, k, dim=1)
    hits = torch.gather(y_true, 1, top_k_indices)

    return torch.mean(hits.sum(dim=1) > 0).item()

def _dcg(y_true: torch.Tensor, y_score: torch.Tensor, k: int | None = None) -> float:
    """
    Compute the discounted cumulative gain (DCG) for retrieval tasks.

    Args:
        y_true (torch.Tensor): Shape (N, C), ground truth binary labels (0 or 1).
        y_score (torch.Tensor): Shape (N, C), predicted scores or probabilities.
        k (int): The number of top predictions to consider.

    Returns:
        float: The computed DCG value.
    """
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape.")
    if k is None:
        k = y_score.shape[1]
    if k <= 0 or k > y_score.shape[1]:
        raise ValueError(f"k must be in the range [1, {y_score.shape[1]}].")

    _, top_k_indices = torch.topk(y_score, k, dim=1)
    gains = torch.gather(y_true, 1, top_k_indices)

    discounts = torch.log2(torch.arange(2, k + 2, device=y_true.device).float())
    return torch.sum(gains / discounts).item()

def ndcg_score(y_true: torch.Tensor, y_score: torch.Tensor, k: int | None = None) -> float:
    """
    Compute the normalized discounted cumulative gain (NDCG) for retrieval tasks.

    Args:
        y_true (torch.Tensor): Shape (N, C), ground truth binary labels (0 or 1).
        y_score (torch.Tensor): Shape (N, C), predicted scores or probabilities.
        k (int): The number of top predictions to consider.

    Returns:
        float: The computed NDCG value.
    """
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape.")
    if k is None:
        k = y_score.shape[1]
    if k <= 0 or k > y_score.shape[1]:
        raise ValueError(f"k must be in the range [1, {y_score.shape[1]}].")

    ideal_dcg = _dcg(y_true, y_true, k)
    if ideal_dcg == 0:
        return 0.0

    actual_dcg = _dcg(y_true, y_score, k)
    return actual_dcg / ideal_dcg

def auuc_score(
    y: torch.Tensor, t: torch.Tensor, tau: torch.Tensor,
    qini: bool = True, normalize: bool = True
) -> float:
    '''
    Compute the Area Under the Uplift Curve (AUUC).

    Args:
        y (torch.Tensor): Binary outcome (0 or 1).
        t (torch.Tensor): Binary treatment assignment (0 or 1).
        tau (torch.Tensor): Uplift scores.
        qini (bool): If True, compute Qini instead of AUUC (subtracts the random baseline).

    Returns:
        float: The computed AUUC value.
    '''
    if y.dim() != 1 or t.dim() != 1 or tau.dim() != 1:
        raise ValueError("y, t, and uplift must be 1-dimensional tensors.")
    if y.shape != t.shape or y.shape != tau.shape:
        raise ValueError("y, t, and uplift must have the same shape.")
    if not torch.all((t == 0) | (t == 1)):
        raise ValueError("t must be a binary tensor (0 or 1).")
    if not torch.all((y == 0) | (y == 1)):
        raise ValueError("y must be a binary tensor (0 or 1).")
    if not torch.any(t) or not torch.any(1 - t):
        raise ValueError("t must have at least one positive and one negative sample.")

    N, = y.shape
    n_t = t.sum()
    n_c = N - n_t

    uplift_desc_idx = torch.argsort(tau, descending=True)

    y_sorted = y[uplift_desc_idx]
    t_sorted = t[uplift_desc_idx]

    cumsum_y_t = (y_sorted * t_sorted).cumsum(0) / n_t
    cumsum_y_c = (y_sorted * (1 - t_sorted)).cumsum(0) / n_c

    v_k = cumsum_y_t - cumsum_y_c
    if normalize:
        v_k /= v_k[-1].clone()  # Normalize ATE range
    if qini:
        # Random baseline is linear
        v_k -= v_k[-1] * torch.arange(1, N + 1, device=y.device) / N

    auuc = v_k.mean().item()
    return auuc


class BatchedMetric(abc.ABC):
    def __init__(self, **kwargs):
        """
        Initialize the BatchedMetric instance.
        This class is intended to be used as a base class for metrics that can be computed in batches.

        Args:
            **kwargs: Additional keyword arguments that can be used to configure the metric.
        """
        self.kwargs = kwargs
        self.reset()

    def reset(self):
        """
        Reset the metric accumulator to start a new computation.
        """
        self._accumulator = self.accumulator(**self.kwargs)
        self._accumulator.send(None)

    @abc.abstractmethod
    def accumulator(self) -> Generator[None, Tuple[torch.Tensor, torch.Tensor] | None, float]:
        """
        Create a generator that accumulates metric values from batches of (y_true, y_score).

        Returns:
            Generator that yields None to receive batches and returns the computed metric value.
        """
        pass

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor):
        self._accumulator.send((y_true, y_score))

    def finalize(self) -> float:
        '''
        Finalize the metric computation and return the accumulated value.

        Returns:
            float: The computed metric value.
        '''
        try:
            if self._accumulator is None:
                raise RuntimeError("Metric accumulator is not initialized.")
            self._accumulator.send(None)
        except StopIteration as e:
            self.reset()  # Reset the accumulator for the next computation
            return e.value
        raise RuntimeError("Accumulator not initialized.")

class BatchedAUC(BatchedMetric):
    """
    Batched AUC metric for binary classification tasks.
    This class computes the AUC in a batch-wise manner using a generator.

    .. code-block:: python

        auc_metric = BatchedAUC(nbins=1000, device='cpu', logit=False)
        for batch in data_loader:
            y_true, y_score = batch
            auc_metric(y_true, y_score)
        auc_value = auc_metric.finalize()

    """
    def __init__(self, nbins: int = 1000, device: str | torch.device = 'cpu', logit: bool = False):
        super().__init__(nbins=nbins, device=device, logit=logit)

    def accumulator(
        self, nbins: int = 1000, device: str | torch.device = 'cpu',
        logit: bool = False
    ):
        '''
        Compute the AUC in a batch-wise manner using a generator.

        Args:
            nbins (int): Number of bins to use for the histogram.
            device (str | torch.device): Device to use for computation.
            logit (bool): If True, apply sigmoid to the scores before computing AUC.
        '''
        binned = torch.linspace(0, 1, nbins + 1, device=device)
        pos_count = torch.zeros_like(binned, dtype=torch.long, device=device)
        neg_count = torch.zeros_like(binned, dtype=torch.long, device=device)
        auc = 0.0

        while True:
            batch = yield
            if batch is None:
                break
            y_true, y_score = batch
            if y_true.shape != y_score.shape:
                raise ValueError("y_true and y_score must have the same shape.")
            y_true = y_true.view(-1)
            y_score = y_score.view(-1)

            if logit:
                y_score = torch.sigmoid(y_score)
            y_score_binned = torch.bucketize(y_score, binned, right=True)
            pos_count.scatter_add_(0, y_score_binned, y_true.long())
            neg_count.scatter_add_(0, y_score_binned, (1 - y_true).long())

            neg_cumsum = torch.cumsum(neg_count, dim=0)
            auc_num = -0.5 * (pos_count * neg_count).sum()
            auc_num += (pos_count * neg_cumsum).sum()
        auc = (auc_num / (pos_count.sum() * neg_count.sum())).item()
        return auc

class BatchedHitRate(BatchedMetric):
    def __init__(self, k: int | None = None):
        super().__init__(k=k)

    def accumulator(self, k: int | None = None):
        '''
        Compute the hit rate in a batch-wise manner using a generator.

        Args:
            k (int): The number of top predictions to consider.
        '''
        hits = 0
        total = 0

        while True:
            batch = yield
            if batch is None:
                break
            y_true, y_score = batch
            if y_true.shape != y_score.shape:
                raise ValueError("y_true and y_score must have the same shape.")
            if torch.any(torch.sum(y_true, dim=1) != 1):
                raise ValueError("Each sample in y_true must have exactly one positive label.")

            _, top_k_indices = torch.topk(
                y_score, k=k if k is not None else y_score.shape[1], dim=1
            )
            hits += torch.sum(torch.gather(y_true, 1, top_k_indices).sum(dim=1) > 0).item()
            total += y_true.shape[0]

        return hits / total if total > 0 else 0.0

class BatchedNDCG(BatchedMetric):
    def __init__(self, k: int | None = None):
        super().__init__(k=k)

    def accumulator(self, k: int | None = None):
        '''
        Compute the NDCG in a batch-wise manner using a generator.

        Args:
            k (int): The number of top predictions to consider.
        '''
        if k is None:
            k = self.kwargs.get('k', None)
        running_ndcg = 0.0
        n_seen = 0

        while True:
            batch = yield
            if batch is None:
                break
            y_true, y_score = batch
            if y_true.shape != y_score.shape:
                raise ValueError("y_true and y_score must have the same shape.")

            ndcg_value = ndcg_score(y_true, y_score, k=k)
            m = y_true.shape[0]
            new_seen = m + n_seen
            running_ndcg += (ndcg_value - running_ndcg) * (m / new_seen)
            n_seen = new_seen
        return running_ndcg


class MetricFormatter():
    '''
    Formats the output of metrics during training.
    '''
    def __init__(
        self, name: str, starting_epoch: int = 0,
        larger_better: bool = True, eps: float = 5e-4
    ):
        '''
        Initialize the MetricFormatter.

        Args:
            name (str): Name of the metric.
            starting_epoch (int): Starting epoch for the metric.
            larger_better (bool): If True, larger values are better.
            eps (float): Small epsilon value to determine significant changes.
        '''
        self.name = name
        self.epoch = starting_epoch
        self.larger_better = larger_better
        self.best = -1e6 if larger_better else 1e6
        self.best_epoch = -1
        self.last = self.best
        self.current = self.best
        self.eps = eps

    def update(self, value: float):
        '''
        Update the metric with a new value.

        Args:
            value (float): The new value to update the metric with.
        '''
        self.last = self.current
        self.current = value
        if (self.larger_better and value > self.best) or \
                (not self.larger_better and value < self.best):
            self.best = value
            self.best_epoch = self.epoch
        self.epoch += 1

    def format(self) -> str:
        '''
        Format the metric for display.

        Returns:
            str: Formatted string representation of the metric.
        '''
        if self.larger_better:
            color_dict = {True: 'green', False: 'red'}
        else:
            color_dict = {True: 'red', False: 'green'}
        if abs(self.current - self.last) < self.eps:
            current_value = cformat(f'{self.current:.4f}', bf=True)
        else:
            current_value = cformat(
                f'{self.current:.4f}',
                fg=color_dict[self.current > self.last],
                bf=True
            )
        return f'{self.name}: {current_value} (Best: {self.best:.4f}@{self.best_epoch})'

    def __str__(self) -> str:
        return self.format()
