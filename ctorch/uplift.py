'''
uplift.py - Module for uplift modeling functionalities in the ctorch library.
'''
import enum
import functools
import warnings
from typing import Any, List, NamedTuple

import torch


class ATEMethod(enum.StrEnum):
    '''
    The standard estimators for ATE are the Hajek estimator and the Horvitz-Thompson estimator. See https://arxiv.org/abs/2106.07695 for details.
    '''
    HAJEK = 'hajek'
    HORVITZ_THOMPSON = 'horvitz-thompson'


class BucketMethod(enum.StrEnum):
    '''
    The method to use for discretization a series of values. The 'quantile' method creates buckets based on quantiles of the predicted uplift, while the 'uniform' method creates buckets of equal width.
    '''
    INDEX = 'index'
    QUANTILE = 'quantile'
    UNIFORM = 'uniform'


class XAxis(enum.StrEnum):
    '''
    The standard x-axis for uplift curve is the number of samples. Another common choice is the percentage of samples.
    '''
    NUM_SAMPLES = 'num_samples'
    PERCENTAGE = 'percentage'


class YAxis(enum.StrEnum):
    '''
    The standard y-axis for uplift curve is the average treatment effect. Another common choice is the cumulative gain.
    '''
    AVERAGE = 'average'
    CUMULATIVE = 'cumulative'

def validate_propensity(propensity: torch.Tensor | None) -> None:
    '''
    Verify that the propensity scores are valid.

    Args:
        propensity (torch.Tensor): The propensity scores to verify.

    Raises:
        ValueError: If the propensity scores contain NaN values or if any value is not in the range (0, 1).
    '''
    if propensity is None:
        return
    if propensity.isnan().any().item():
        raise ValueError('Propensity scores must not contain NaN values.')
    if torch.any((propensity <= 0) | (propensity >= 1)):
        raise ValueError('Propensity scores must be in the range (0, 1).')

def validate_device(*args: torch.Tensor) -> torch.device:
    '''
    Validate that all input tensors are on the same device.

    Args:
        *args (torch.Tensor): Tensors to validate.

    Returns:
        torch.device: The common device of the input tensors.

    Raises:
        ValueError: If the tensors are on different devices.
    '''
    device = None
    for tensor in args:
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise ValueError('All input tensors must be on the same device.')
    if device is None:
        raise ValueError('No tensors were provided for device validation.')
    return device

def validate_shape(*args: torch.Tensor) -> int:
    '''
    Validate that all input tensors have the same shape on the batch dimension.

    Args:
        *args (torch.Tensor): Tensors to validate.

    Returns:
        int: The size of the batch dimension.

    Raises:
        ValueError: If the shapes of the tensors do not match in the batch dimension.
    '''
    B = None
    for tensor in args:
        if tensor.dim() != 1:
            raise ValueError('All input tensors must be 1-dimensional.')
        if B is None:
            B = tensor.size(0)
        elif tensor.size(0) != B:
            raise ValueError('All input tensors must have the same size in the batch dimension.')
    if B is None:
        raise ValueError('No tensors were provided for shape validation.')
    return B

def validate_binary(*args: torch.Tensor) -> None:
    '''
    Validate that all input tensors are binary (contain only 0s and 1s).

    Args:
        *args (torch.Tensor): Tensors to validate.

    Raises:
        ValueError: If any tensor contains values other than 0 or 1.
    '''
    for tensor in args:
        if not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError('All input tensors must be binary (contain only 0s and 1s).')


def _validate_tensor(*args: torch.Tensor, dtype: torch.dtype | None = None) -> List[torch.Tensor]:
    '''
    Convert input arguments to PyTorch tensors of a specified data type.
    Args:
        *args: Input arguments to convert.
        dtype (torch.dtype): Desired data type for the tensors.
    Raises:
        TypeError: If any argument cannot be converted to a tensor.
    '''
    if dtype is None:
        dtype = torch.get_default_dtype()
    return [torch.as_tensor(arg, dtype=dtype) for arg in args]

def sort_by_tau(*args, key: int, descending: bool):
    '''
    Sort multiple tensors based on the values of a key tensor.

    Args:
        *args (torch.Tensor): Tensors to sort. All tensors must have the same shape in the batch dimension.
        key (int): The index of the tensor to use as the sorting key.
        descending (bool): Whether to sort in descending order.

    Returns:
        List[torch.Tensor]: The sorted tensors in the same order as the input.

    Raises:
        ValueError: If the key index is out of range or if the tensors do not have the same shape in the batch dimension.
    '''
    if key < 0 or key >= len(args):
        raise ValueError(f'Key index {key} is out of range for {len(args)} tensors.')
    validate_shape(*args)
    sorted_indices = torch.argsort(
        args[key], dim=0, descending=descending, stable=True
    )
    return [tensor[sorted_indices] for tensor in args]

nan_to_zero = functools.partial(torch.nan_to_num, nan=0, posinf=0, neginf=0)
cum_sum = functools.partial(torch.cumsum, dim=0)

def batch_indexing(*args: torch.Tensor, index: Any) -> List[torch.Tensor]:
    '''
    Perform slicing on multiple tensors based on a given index.

    Args:
        *args (torch.Tensor): Tensors to slice. All tensors must have the same shape in the batch dimension.
        index (torch.Tensor): A tensor containing the indices for slicing. Must have the same shape in the batch dimension as the input tensors.

    Returns:
        List[torch.Tensor]: The sliced tensors in the same order as the input.

    Raises:
        ValueError: If the index is not a tensor of integers or if the tensors do not have the same shape in the batch dimension.
    '''
    validate_shape(*args)
    return [tensor[index] for tensor in args]

class GroupPOResult(NamedTuple):
    response: torch.Tensor  # Cumulative weighted response for the group
    sample: torch.Tensor    # Cumulative sample weight for the group

class MeanPOResult(NamedTuple):
    neg: GroupPOResult  # GroupPOResult for the control group
    pos: GroupPOResult  # GroupPOResult for the treatment group
    n: torch.Tensor     # Number of samples considered in the calculation (after removing ties if applicable)

def calculate_mean_po(
    y: torch.Tensor, t: torch.Tensor, a: torch.Tensor,
    propensity: torch.Tensor | None = None,
    k: int | float | None = None,
    descending: bool = True,
    remove_ties: bool = False,
    weight_sample: bool = True,
    weight_response: bool = True,
) -> MeanPOResult:
    '''
    Compute cumulative sample weights and weighted responses for treatment and control groups.

    Args:
        y (torch.Tensor): Correct (true) target values.
        t (torch.Tensor): Predicted uplift, as returned by a model.
        a (torch.Tensor): Treatment labels.
        propensity (torch.Tensor, optional): Propensity scores. If provided, inverse propensity score weighting is applied.
        k (int | float | None, optional): If provided, return the ATE curve up to k. If k is a float in the range (0, 1], it is treated as a percentage of the total number of samples (rounded down). If None, return the ATE curve for all samples.
        descending (bool): Whether to sort the predicted uplift in descending order. Default is True, which means higher predicted uplift will be ranked higher.
        remove_ties (bool): Whether to remove ties in predicted uplift when sorting. Should be used when calculating curves and not when calculating ATE at a specific k.

    Returns:
        MeanPOResult: A named tuple containing:
            - neg: GroupPOResult for the control group, with response and sample weights.
            - pos: GroupPOResult for the treatment group, with response and sample weights.
            - n: The number of samples considered in the calculation (after removing ties if applicable).
    '''
    # Sanity checks and input validation
    y, t, a = _validate_tensor(y, t, a)
    if propensity is not None:
        e, = _validate_tensor(propensity)
    else:
        treat_ratio = a.mean()
        e = a.new_ones(a.size()) * treat_ratio
    validate_propensity(e)
    validate_device(y, t, a, e)
    validate_shape(y, t, a, e)
    validate_binary(a)
    match k:
        case None: pass
        case int() if k > 0 and k <= y.size(0): pass
        case float() if 0 < k and k <= 1 and int(k * y.size(0)):
            k = int(k * y.size(0))
        case _: raise ValueError(f'Invalid value for k: {k}. Must be a positive integer or a float in the range (0, 1]. If k is a float, the rounded (floored) value must be at least 1.')
    if weight_sample and not weight_response:
        raise ValueError('weight_sample cannot be True when weight_response is False, as it would lead to undefined behavior in the ATE calculation.')

    # Sort and truncate
    y, t, a, e = sort_by_tau(y, t, a, e, key=1, descending=descending)
    if k is not None:
        y, t, a, e = batch_indexing(y, t, a, e, index=slice(0, k))

    # Check identifiability
    if torch.sum(a) == 0 or torch.sum(1 - a) == 0:
        raise ValueError('Both treatment and control groups must have at least one sample for identifiability.')
    elif torch.mean(a) < 0.05 or torch.mean(a) > 0.95:
        warnings.warn(f'Treatment assignment is highly imbalanced (treatment ratio: {torch.mean(a).item():.4f}). The ATE estimation may be unreliable.')

    # Numeric stability
    eps = torch.finfo(e.dtype).eps
    e = torch.clamp(e, eps, 1 - eps)

    # Prepare IPW weights and corresponding weighted outcomes
    # ATE-IPW = Y * T / p - Y * (1 - T) / (1 - p)
    s_1, s_0 = y * a, y * (1 - a)
    if weight_response:
        s_1, s_0 = s_1 / e, s_0 / (1 - e)
    d_1, d_0 = a, 1 - a
    if weight_sample:
        d_1, d_0 = d_1 / e, d_0 / (1 - e)

    # Calculate cumulative mean potential outcomes
    D_1, D_0 = cum_sum(d_1), cum_sum(d_0)
    S_1, S_0 = cum_sum(s_1), cum_sum(s_0)

    # Re-scale
    if not weight_sample and weight_response:
        D_1, D_0 = D_1 + D_0, D_1 + D_0

    n = torch.arange(1, y.size(0) + 1, device=y.device)
    if remove_ties:
        t_diff = t[1:] != t[:-1]
        tie_mask = torch.cat([t_diff, t_diff.new_tensor([True])])
    else:
        tie_mask = torch.ones_like(t, dtype=torch.bool)

    D_1, D_0 = D_1[tie_mask], D_0[tie_mask]
    S_1, S_0 = S_1[tie_mask], S_0[tie_mask]
    n = n[tie_mask]

    return MeanPOResult(
        neg=GroupPOResult(response=S_0, sample=D_0),
        pos=GroupPOResult(response=S_1, sample=D_1),
        n=n,
    )

def lift_at_k(
    y: torch.Tensor,
    t: torch.Tensor,
    a: torch.Tensor,
    e: torch.Tensor | None = None,
    k: int | float | None = None,
    descending: bool = True,
    method: ATEMethod = ATEMethod.HAJEK,
):
    """Compute uplift at first k observations by uplift of the total sample.

    Args:
        y (1d array-like): Correct (true) target values.
        t (1d array-like): Predicted uplift, as returned by a model.
        a (1d array-like): Treatment labels.
        e (1d array-like, optional): Propensity scores. Will be inferred as uniform if not provided.
        k (int | float | None, optional): If provided, return the uplift at first k observations. If None, return the uplift at all observations.
        descending (bool): Whether to sort the predicted uplift in descending order. Default is True.
        method (ATEMethod): The method to use for ATE estimation. Default is ATEMethod.HAJEK.

    Returns:
        float: Uplift at first k observations.
    """
    if method == ATEMethod.HAJEK:
        weight_sample = True
    elif method == ATEMethod.HORVITZ_THOMPSON:
        weight_sample = False
    else:
        raise ValueError(f'Unsupported method: {method}')
    (S_0, D_0), (S_1, D_1), n = calculate_mean_po(
        y, t, a, e, k=k, remove_ties=False, descending=descending,
        weight_sample=weight_sample, weight_response=True
    )
    valid = ~torch.isnan(S_1 / D_1 - S_0 / D_0)
    if not valid[-1].item():
        raise ValueError('The value at k is not identifiable due to lack of samples in treatment or control group.')
    return (S_1 / D_1 - S_0 / D_0)[-1].item()

def uplift_curve(
    y: torch.Tensor,
    t: torch.Tensor,
    a: torch.Tensor,
    propensity: torch.Tensor | None = None,
    descending: bool = True,
    method: ATEMethod = ATEMethod.HAJEK,
    x_axis: XAxis = XAxis.NUM_SAMPLES,
    y_axis: YAxis = YAxis.CUMULATIVE,
):
    """Compute Uplift curve.

    For computing the area under the Uplift Curve, see :func:`.uplift_auc_score`.

    Args:
        y (1d array-like): Correct (true) target values.
        t (1d array-like): Predicted uplift, as returned by a model.
        a (1d array-like): Treatment labels.
        propensity (1d array-like, optional): Propensity scores. Will be inferred as uniform if not provided.
        descending (bool): Whether to sort the predicted uplift in descending order. Default is True.
        method (ATEMethod): The method to use for ATE estimation. Default is ATEMethod.HAJEK.
        x_axis (XAxis): The metric to use for the x-axis of the uplift curve. Default is XAxis.NUM_SAMPLES.
        y_axis (YAxis): The metric to use for the y-axis of the uplift curve. Default is YAxis.CUMULATIVE.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: x, y points on the uplift curve, and whether the value is identifiable.
    """
    if method == ATEMethod.HAJEK:
        weight_sample = True
    elif method == ATEMethod.HORVITZ_THOMPSON:
        weight_sample = False
    else:
        raise ValueError(f'Unsupported method: {method}')
    (S_0, D_0), (S_1, D_1), n = calculate_mean_po(
        y, t, a, propensity, k=None, remove_ties=True, descending=descending,
        weight_sample=weight_sample, weight_response=True
    )
    ate = S_1 / D_1 - S_0 / D_0
    valid, y_value = ~torch.isnan(ate), nan_to_zero(ate)
    x_value = n.to(y_value.dtype)

    if y_axis == YAxis.CUMULATIVE:
        y_value *= x_value
    elif y_axis != YAxis.AVERAGE:
        raise ValueError(f'Unsupported y_axis: {y_axis}')
    if x_axis == XAxis.PERCENTAGE:
        x_value /= torch.amax(x_value, keepdim=True)
    elif x_axis != XAxis.NUM_SAMPLES:
        raise ValueError(f'Unsupported x_axis: {x_axis}')
    x_value = torch.cat((x_value.new_zeros(1), x_value))
    y_value = torch.cat((y_value.new_zeros(1), y_value))
    valid = torch.cat((valid.new_tensor([True]), valid))

    return x_value, y_value, valid


def uplift_auc_score(
    y: torch.Tensor,
    t: torch.Tensor,
    a: torch.Tensor,
    propensity: torch.Tensor | None = None,
    descending: bool = True,
    method: ATEMethod = ATEMethod.HAJEK,
    x_axis: XAxis = XAxis.NUM_SAMPLES,
    y_axis: YAxis = YAxis.CUMULATIVE,
    reduce_baseline: bool = True
):
    '''
    Compute the Area Under the Uplift Curve from prediction scores.

    Args:
        y (1d array-like): Correct (true) binary target values.
        t (1d array-like): Predicted uplift, as returned by a model.
        a (1d array-like): Treatment labels.
        propensity (1d array-like, optional): Propensity scores. Will be inferred as uniform if not provided.
        descending (bool): Whether to sort the predicted uplift in descending order. Default is True.
        method (ATEMethod): The method to use for ATE estimation. Default is ATEMethod.HAJEK.
        x_axis (XAxis): The metric to use for the x-axis of the uplift curve.
        y_axis (YAxis): The metric to use for the y-axis of the uplift curve.
        reduce_baseline (bool): Whether to subtract the area under the baseline curve.

    Returns:
        float: Area Under the Uplift Curve.
    '''
    x_value, y_value, valid = uplift_curve(
        y, t, a, propensity, method=method,
        x_axis=x_axis, y_axis=y_axis, descending=descending
    )
    x_value, y_value = x_value[valid], y_value[valid]

    if reduce_baseline:
        # Construct baseline curve
        x_baseline = x_value.new_tensor([x_value[0], x_value[-1]])
        ate = y_value[-1]
        if y_axis == YAxis.CUMULATIVE:
            start = ate * x_value[0] / x_value[-1]
        elif y_axis == YAxis.AVERAGE:
            start = ate
        else:
            raise ValueError(f'Unsupported y_axis: {y_axis}')
        y_baseline = y_value.new_tensor([start, ate])

        baseline_score = torch.trapz(y_baseline, x_baseline)
    else:
        baseline_score = 0
    uplift_score = torch.trapz(y_value, x_value)

    return (uplift_score - baseline_score).item()


def qini_curve(
    y: torch.Tensor, t: torch.Tensor, a: torch.Tensor,
    propensity: torch.Tensor | None = None,
    descending: bool = True,
    x_axis: XAxis = XAxis.NUM_SAMPLES
):
    '''
    Compute Qini curve.

    For computing the area under the Qini Curve, see :func:`.qini_auc_score`. Qini curve only supports Hajek estimator for ATE.

    Args:
        y (1d array-like): Correct (true) binary target values.
        t (1d array-like): Predicted uplift, as returned by a model.
        a (1d array-like): Treatment labels.
        propensity (1d array-like, optional): Propensity scores. If provided, inverse propensity score weighting is applied.
        descending (bool): Whether to sort the predicted uplift in descending order. Default is True.
        x_axis (XAxis): The metric to use for the x-axis of the Qini curve. Default is XAxis.NUM_SAMPLES.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: x and y points on the Qini curve, and whether the value is identifiable.
    '''
    use_ipw = propensity is not None
    (S_0, D_0), (S_1, D_1), n = calculate_mean_po(
        y, t, a, propensity, descending=descending,
        k=None, remove_ties=True, weight_sample=use_ipw, weight_response=use_ipw
    )
    ate = S_1 - S_0 * D_1 / D_0
    valid, y_value = ~torch.isnan(ate), nan_to_zero(ate)
    x_value = n.to(y_value.dtype)

    if x_axis == XAxis.PERCENTAGE:
        x_value /= torch.amax(x_value, keepdim=True)
    elif x_axis != XAxis.NUM_SAMPLES:
        raise ValueError(f'Unsupported x_axis: {x_axis}')

    x_value = torch.cat((x_value.new_zeros(1), x_value))
    y_value = torch.cat((y_value.new_zeros(1), y_value))
    valid = torch.cat((valid.new_tensor([True]), valid))

    return x_value, y_value, valid


def qini_auc_score(
    y: torch.Tensor,
    t: torch.Tensor,
    a: torch.Tensor,
    propensity: torch.Tensor | None = None,
    descending: bool = True,
    x_axis: XAxis = XAxis.NUM_SAMPLES,
    reduce_baseline: bool = True
):
    '''
    Compute the area under the Qini Curve.

    For computing the Qini Curve itself, see :func:`.qini_curve`. Qini curve only supports Hajek estimator for ATE.

    Args:
        y (1d array-like): Correct (true) binary target values.
        t (1d array-like): Predicted uplift, as returned by a model.
        a (1d array-like): Treatment labels.
        propensity (1d array-like, optional): Propensity scores. If provided, inverse propensity score weighting is applied.
        descending (bool): Whether to sort the predicted uplift in descending order. Default is True.
        x_axis (XAxis): The metric to use for the x-axis of the Qini curve.
        reduce_baseline (bool): Whether to subtract the area under the baseline curve.

    Returns:
        float: Area Under the Qini Curve.
    '''
    x_value, y_value, valid = qini_curve(
        y, t, a, propensity, x_axis=x_axis, descending=descending
    )
    x_value, y_value = x_value[valid], y_value[valid]

    if reduce_baseline:
        # Construct baseline curve
        x_baseline = x_value.new_tensor([x_value[0], x_value[-1]])
        ate = y_value[-1]
        start = ate * x_value[0] / x_value[-1]
        y_baseline = y_value.new_tensor([start, ate])

        baseline_score = torch.trapz(y_baseline, x_baseline)
    else:
        baseline_score = 0
    uplift_score = torch.trapz(y_value, x_value)

    return (uplift_score - baseline_score).item()

def bucketize(
    value: torch.Tensor,
    num_buckets: int,
    method: BucketMethod = BucketMethod.INDEX,
    descending: bool = True
) -> torch.Tensor:
    '''
    Bucketize the input values into specified number of buckets.

    Args:
        value (torch.Tensor): The input values to bucketize, shape [N].
        num_buckets (int): The number of buckets to create.
        method (BucketMethod): The method to use for bucketing. Default is BucketMethod.INDEX.
        descending (bool): Whether to assign lower bucket ids to higher values. Default is True.

    Returns:
        torch.Tensor: A tensor of bucket ids for each input value, shape [N].
    '''
    value, = _validate_tensor(value)
    validate_shape(value)
    device = validate_device(value)
    if num_buckets <= 0:
        raise ValueError(f'num_buckets should be a positive integer, got {num_buckets}.')

    # Sort
    N = value.size(0)
    index = torch.argsort(value, dim=0, stable=True)
    inverse_index = torch.empty_like(index)
    inverse_index[index] = torch.arange(N, device=device)
    sorted_value = value[index]

    if num_buckets > N:
        raise ValueError(f'num_buckets ({num_buckets}) cannot be greater than the number of samples ({N}).')

    # Bucketize
    quantile = torch.linspace(0, 1, num_buckets + 1, device=device)[1:-1]
    match method:
        case BucketMethod.QUANTILE:
            boundaries = torch.quantile(sorted_value, quantile)
            bucket_input = sorted_value
            bucket_id = torch.bucketize(bucket_input, boundaries, right=False)
        case BucketMethod.UNIFORM:
            l, r = torch.aminmax(sorted_value)
            boundaries = quantile * (r - l) + l
            bucket_input = sorted_value
            bucket_id = torch.bucketize(bucket_input, boundaries, right=False)
        case BucketMethod.INDEX:
            rank = torch.arange(N, device=device)
            bucket_id = torch.div(rank * num_buckets, N, rounding_mode='floor')
        case _:
            raise ValueError(f'Unsupported bucketization method: {method}')
    if descending:
        bucket_id = num_buckets - 1 - bucket_id
    bucket_id = bucket_id[inverse_index]

    return bucket_id


def krcc_score(
    y: torch.Tensor,
    t: torch.Tensor,
    a: torch.Tensor,
    propensity: torch.Tensor | None = None,
    num_buckets: int | None = None,
    true_uplift: torch.Tensor | None = None,
    ate_method: ATEMethod = ATEMethod.HAJEK,
    ignore_unidentifiable: bool = True
):
    '''
    Compute the Kendall Rank Correlation Coefficient (KRCC) between the predicted uplift and the true uplift.

    Args:
        y (1d array-like): Observed outcome values.
        t (1d array-like): Predicted uplift, as returned by a model.
        a (1d array-like): Treatment labels.
        propensity (1d array-like, optional): Propensity scores. Will be inferred as uniform if not provided.
        num_buckets (int, optional): If provided, the predicted uplift and true uplift will be bucketized into the specified number of buckets before computing KRCC. If None, ``true_uplift`` must not be absent.
        true_uplift (1d array-like, optional): The true uplift values for each sample. If provided, KRCC will be computed between the predicted uplift and the true uplift. If None, the true uplift for each bucket will be calculated using the specified ATE method and used for KRCC calculation.
        ate_method (ATEMethod): The method to use for ATE estimation when calculating true uplift for each bucket. Default is ATEMethod.HAJEK. The argument will be ignored if ``true_uplift`` is provided.
        ignore_unidentifiable (bool): Whether to ignore buckets that do not contain samples from both treatment and control groups when calculating KRCC. If False, a ValueError will be raised if any bucket does not contain samples from both groups. The argument will be ignored if ``true_uplift`` is provided.

    Returns:
        float: KRCC score.
    '''
    from scipy.stats import kendalltau

    y, t, a = _validate_tensor(y, t, a)
    if propensity is not None:
        e, = _validate_tensor(propensity)
    else:
        treat_ratio = a.mean()
        e = a.new_ones(a.size()) * treat_ratio
    validate_propensity(e)
    device = validate_device(y, t, a, e)
    validate_shape(y, t, a, e)
    if true_uplift is not None:
        true_uplift, = _validate_tensor(true_uplift)
        validate_shape(true_uplift, y)
        validate_device(true_uplift, y)
    validate_binary(a)

    if num_buckets is None:
        if true_uplift is None:
            raise ValueError('Either num_buckets or true_uplift must be provided to compute KRCC score.')
        else:
            pred_bucket_uplift = t
            true_bucket_uplift = true_uplift
    else:
        bucket_id = bucketize(t, num_buckets=num_buckets, method=BucketMethod.INDEX)

        true_bucket_uplift = []
        if true_uplift is None:
            for i in range(num_buckets):
                mask = bucket_id == i
                y_i, t_i, a_i, e_i = batch_indexing(y, t, a, e, index=mask)
                if not all((a_i == _).any().item() for _ in (0, 1)):
                    if not ignore_unidentifiable:
                        raise ValueError(f'Bucket {i} does not contain samples from both treatment and control groups, which is required for identifiability in KRCC calculation.')
                    lift = float('nan')
                else:
                    lift = lift_at_k(y_i, t_i, a_i, e_i, method=ate_method, k=None)
                true_bucket_uplift.append(lift)
        else:
            for i in range(num_buckets):
                mask = bucket_id == i
                true_bucket_uplift.append(true_uplift[mask].mean().item())

        true_bucket_uplift = y.new_tensor(true_bucket_uplift, device=device)

        pred_bucket_uplift = []
        for i in range(num_buckets):
            mask = bucket_id == i
            t_i = t[mask]
            pred_bucket_uplift.append(t_i.mean().item())
        pred_bucket_uplift = y.new_tensor(pred_bucket_uplift, device=device)

    result = kendalltau(
        pred_bucket_uplift.numpy(force=True),
        true_bucket_uplift.numpy(force=True),
        nan_policy='omit'
    )
    return float(result.statistic)
