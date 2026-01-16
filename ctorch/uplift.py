'''
uplift.py - Module for uplift modeling functionalities in the ctorch library.
'''
from typing import List
import torch

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

def validate_shape(*args: torch.Tensor, batch_dimension: int = 0) -> int:
    '''
    Validate that all input tensors have the same shape on the batch dimension.

    Args:
        *args (torch.Tensor): Tensors to validate.
        batch_dimension (int): The dimension representing the batch size.

    Returns:
        int: The size of the batch dimension.

    Raises:
        ValueError: If the shapes of the tensors do not match in the batch dimension.
    '''
    B = None
    for tensor in args:
        if tensor.dim() <= batch_dimension:
            raise ValueError(f'Tensor with shape {tensor.shape} does not have a batch dimension at index {batch_dimension}.')
        if B is None:
            B = tensor.size(batch_dimension)
        elif tensor.size(batch_dimension) != B:
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


def validate_tensor(*args: torch.Tensor, dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    '''
    Convert input arguments to PyTorch tensors of a specified data type.
    Args:
        *args: Input arguments to convert.
        dtype (torch.dtype): Desired data type for the tensors.
    Raises:
        TypeError: If any argument cannot be converted to a tensor.
    '''
    return [torch.as_tensor(arg, dtype=dtype) for arg in args]


def _calculate_cumulative(
    y_true: torch.Tensor,
    uplift: torch.Tensor,
    treatment: torch.Tensor,
    propensity: torch.Tensor | None = None
):
    '''
    Compute cumulative sample weights and weighted responses for treatment and control groups.

    Args:
        y_true (torch.Tensor): Correct (true) binary target values.
        uplift (torch.Tensor): Predicted uplift, as returned by a model.
        treatment (torch.Tensor): Treatment labels.
        propensity (torch.Tensor, optional): Propensity scores. If provided, inverse propensity score weighting is applied.

    Returns:
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.device]:
            Cumulative weights and responses for treatment and control groups,
            total number of samples at each threshold, and the device used.
    '''
    # Input validation
    y_true, uplift, treatment = validate_tensor(y_true, uplift, treatment)
    if propensity is not None:
        propensity, = validate_tensor(propensity)
    else:
        treat_ratio = treatment.mean()
        propensity = treatment.new_ones(treatment.size()) * treat_ratio
    device = validate_device(y_true, uplift, treatment, propensity)
    validate_shape(y_true, uplift, treatment)
    validate_binary(treatment, y_true)

    # Sort by predicted uplift scores in descending order
    desc_score_indices = torch.argsort(uplift, descending=True, stable=True)
    y_true = y_true[desc_score_indices]
    uplift = uplift[desc_score_indices]
    treatment = treatment[desc_score_indices]
    propensity = propensity[desc_score_indices]

    # Numeric stability
    eps = torch.finfo(propensity.dtype).eps
    propensity = torch.clamp(propensity, eps, 1 - eps)

    # Prepare IPW weights and corresponding weighted outcomes
    w_trmnt = treatment * (1 / propensity)
    w_ctrl = (1 - treatment) * (1 / (1 - propensity))
    y_w_trmnt = y_true * w_trmnt
    y_w_ctrl = y_true * w_ctrl

    distinct_value_indices = torch.where(torch.diff(uplift) != 0)[0]
    last_idx = torch.tensor(
        [uplift.size(0) - 1],
        device=device, dtype=torch.long
    )
    threshold_indices = torch.cat((distinct_value_indices, last_idx))

    # Cumulative sums at threshold indices
    cum_w_trmnt = torch.cumsum(w_trmnt, dim=0)[threshold_indices]
    cum_y_trmnt = torch.cumsum(y_w_trmnt, dim=0)[threshold_indices]
    cum_w_ctrl = torch.cumsum(w_ctrl, dim=0)[threshold_indices]
    cum_y_ctrl = torch.cumsum(y_w_ctrl, dim=0)[threshold_indices]
    num_all = (threshold_indices + 1).to(dtype=torch.long)

    return (cum_w_trmnt, cum_y_trmnt), (cum_w_ctrl, cum_y_ctrl), num_all, device

def uplift_curve(
    y_true: torch.Tensor,
    uplift: torch.Tensor,
    treatment: torch.Tensor,
    propensity: torch.Tensor | None = None
):
    """Compute Uplift curve.

    For computing the area under the Uplift Curve, see :func:`.uplift_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        propensity (1d array-like, optional): Propensity scores. Will be inferred as uniform if not provided.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: x and y points on the Uplift curve.
    """
    (cum_w_trmnt, cum_y_trmnt), (cum_w_ctrl, cum_y_ctrl), num_all, device = \
        _calculate_cumulative(y_true, uplift, treatment, propensity)

    # subgroup uplift (weighted mean difference)
    mu_trmnt = torch.nan_to_num(cum_y_trmnt / cum_w_trmnt, nan=0.0, posinf=0.0, neginf=0.0)
    mu_ctrl = torch.nan_to_num(cum_y_ctrl / cum_w_ctrl, nan=0.0, posinf=0.0, neginf=0.0)
    uplift_at_k = mu_trmnt - mu_ctrl

    curve_values = uplift_at_k * num_all.to(uplift_at_k.dtype)

    zero = torch.tensor([0], device=device, dtype=num_all.dtype)
    zero_f = torch.tensor([0.0], device=device, dtype=curve_values.dtype)

    num_all = torch.cat((zero, num_all))
    curve_values = torch.cat((zero_f, curve_values))

    return num_all, curve_values

def perfect_uplift(
    y_true: torch.Tensor,
    treatment: torch.Tensor
):
    '''
    Compute the perfect (optimum) uplift predictions.

    Args:
        y_true (torch.Tensor): Correct (true) binary target values.
        treatment (torch.Tensor): Treatment labels.

    Returns:
        torch.Tensor: Perfect uplift predictions.
    '''
    # Input validation
    y_true, treatment = validate_tensor(y_true, treatment)
    validate_shape(y_true, treatment)
    validate_binary(treatment, y_true)

    cr_num = torch.sum((y_true == 1) & (treatment == 0)).item()  # Control Responders
    tn_num = torch.sum((y_true == 0) & (treatment == 1)).item()  # Treated Non-Responders

    # express an ideal uplift curve through y_true and treatment
    summand = y_true if cr_num > tn_num else treatment
    perfect_uplift = 2 * (y_true == treatment) + summand
    return perfect_uplift.to(dtype=torch.float32)

def perfect_uplift_curve(
    y_true: torch.Tensor,
    treatment: torch.Tensor,
    propensity: torch.Tensor | None = None
):
    '''
    Compute the perfect (optimum) Uplift curve.

    This is a function, given points on a curve.  For computing the
    area under the Uplift Curve, see :func:`.uplift_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        treatment (1d array-like): Treatment labels.
        propensity (1d array-like, optional): Propensity scores. Will be inferred as uniform if not provided.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Points on the perfect Uplift curve.
    '''

    # Input validation
    tau_perfect = perfect_uplift(y_true, treatment)

    return uplift_curve(y_true, tau_perfect, treatment, propensity)

def uplift_auc_score(y_true, uplift, treatment, propensity=None):
    '''
    Compute normalized Area Under the Uplift Curve from prediction scores.

    By computing the area under the Uplift curve, the curve information is summarized in one number.
    For binary outcomes the ratio of the actual uplift gains curve above the diagonal to that of
    the optimum Uplift Curve.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        propensity (1d array-like, optional): Propensity scores. Will be inferred as uniform if not provided.

    Returns:
        float: Normalized Area Under the Uplift Curve.
    '''
    x_actual, y_actual = uplift_curve(y_true, uplift, treatment, propensity)
    x_perfect, y_perfect = perfect_uplift_curve(y_true, treatment, propensity)
    x_baseline = x_perfect.new_tensor([0, x_perfect[-1]])
    y_baseline = y_perfect.new_tensor([0, y_perfect[-1]])

    auc_score_baseline = torch.trapz(y_baseline, x_baseline)
    auc_score_perfect = torch.trapz(y_perfect, x_perfect) - auc_score_baseline
    auc_score_actual = torch.trapz(y_actual, x_actual) - auc_score_baseline

    return (auc_score_actual / auc_score_perfect).item()


def qini_curve(
    y_true: torch.Tensor,
    uplift: torch.Tensor,
    treatment: torch.Tensor,
    propensity: torch.Tensor | None = None
):
    '''
    Compute Qini curve.

    For computing the area under the Qini Curve, see :func:`.qini_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        propensity (1d array-like, optional): Propensity scores. If provided, inverse propensity score weighting is applied.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: x and y points on the Qini curve.
    '''
    (cum_w_trmnt, cum_y_trmnt), (cum_w_ctrl, cum_y_ctrl), num_all, device = \
        _calculate_cumulative(y_true, uplift, treatment, propensity)

    # Complete here
    ratio = torch.zeros_like(cum_w_trmnt)
    ratio = torch.where(cum_w_ctrl != 0, cum_w_trmnt / cum_w_ctrl, ratio)

    curve_values = cum_y_trmnt - cum_y_ctrl * ratio

    # Prepend zero point
    zero = torch.tensor([0], device=device, dtype=num_all.dtype)
    zero_f = torch.tensor([0.0], device=device, dtype=curve_values.dtype)
    num_all = torch.cat((zero, num_all))
    curve_values = torch.cat((zero_f, curve_values))

    return num_all, curve_values


def perfect_qini_curve(
    y_true: torch.Tensor,
    treatment: torch.Tensor,
    propensity: torch.Tensor | None = None
):
    '''
    Compute the perfect (optimum) Qini curve.

    This is a function, given points on a curve.  For computing the
    area under the Qini Curve, see :func:`.qini_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        treatment (1d array-like): Treatment labels.
        propensity (1d array-like, optional): Propensity scores. Will be inferred as uniform if not provided.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Points on the perfect Qini curve.
    '''

    # Input validation
    tau_perfect = perfect_uplift(y_true, treatment)

    return qini_curve(y_true, tau_perfect, treatment, propensity)


def qini_auc_score(y_true, uplift, treatment, propensity=None):
    '''
    Compute normalized Area Under the Qini Curve from prediction scores.

    By computing the area under the Qini curve, the curve information is summarized in one number.
    For binary outcomes the ratio of the actual uplift gains curve above the diagonal to that of
    the optimum Qini Curve.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        propensity (1d array-like, optional): Propensity scores. Will be inferred as uniform if not provided.

    Returns:
        float: Normalized Area Under the Qini Curve.
    '''
    x_actual, y_actual = qini_curve(y_true, uplift, treatment, propensity)
    x_perfect, y_perfect = perfect_qini_curve(y_true, treatment, propensity)
    x_baseline = x_perfect.new_tensor([0, x_perfect[-1]])
    y_baseline = y_perfect.new_tensor([0, y_perfect[-1]])

    auc_score_baseline = torch.trapz(y_baseline, x_baseline)
    auc_score_perfect = torch.trapz(y_perfect, x_perfect) - auc_score_baseline
    auc_score_actual = torch.trapz(y_actual, x_actual) - auc_score_baseline

    return (auc_score_actual / auc_score_perfect).item()
