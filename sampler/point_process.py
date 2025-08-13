'''
`utils.sampler.point_process` - Point Process Simulation
'''


from typing import Callable, Tuple
import numpy as np

def integrated_intensity(
    lambda_: Callable[[float, np.ndarray], float],
    event_times: np.ndarray,
    t_start: float = 0,
    n_points: int = 10
):
    '''
    Compute the integrated intensity function at given event times from intensity function lambda_

    .. math::
        \\Lambda^\\ast (t) = \\int_{0}^{t} \\lambda^\\ast (u) du

    Args:
        lambda_ (Callable[[float, np.ndarray], float]): The intensity function of the point process.
        event_times (np.ndarray): The event times at which to compute the integrated intensity.
        t_start (float): The start time of the point process.
        n_points (int): Number of points to use for numerical integration.
    '''
    # Sanity checks
    if not np.all(np.diff(event_times) > 0):
        raise ValueError('Event times must be strictly increasing.')

    integrals = np.zeros_like(event_times)
    event_range = zip(
        [t_start, *event_times],
        [*event_times]
    )
    for i, (start, end) in enumerate(event_range):
        # Compute the integrated intensity at each event time
        t_grid = np.linspace(start, end, n_points)
        lambda_between_events = np.vectorize(
            lambda t: lambda_(t, event_times[:i]), otypes=[float]
        )
        integrals[i] = np.trapz(lambda_between_events(t_grid), t_grid)

    return np.cumsum(integrals)

def sample_hpp(
    lambda_: float,
    t_start: float,
    t_end: float
) -> np.ndarray:
    '''
    Sample from a homogeneous Poisson process. The interval time distribution of a homogeneous Poisson process with intensity lambda_ is given by an exponential distribution.

    .. math::
        f(t) = \\lambda e^{-\\lambda t}

    Args:
        lambda\\_ (float): The intensity (rate) of the process.
        t_start (float): The start time of the interval.
        t_end (float): The end time of the interval.

    Returns:
        np.ndarray: An array of sampled event times.
    '''
    # Sanity check
    if lambda_ <= 0:
        raise ValueError('Intensity lambda_ must be positive.')
    if t_start >= t_end:
        raise ValueError('Invalid time interval: t_start must be less than t_end.')

    expected_num = lambda_ * (t_end - t_start)
    u = np.random.rand(int(expected_num * 1.645)) # 95% Confidence
    event_times = -np.log(u) / lambda_ # Exponential distribution

    event_times = np.cumsum(event_times, axis=0) + t_start
    if event_times.shape[0] == 0:
        return np.array([])
    if event_times[-1] < t_end:
        event_times = np.concatenate([
            event_times, sample_hpp(lambda_, event_times[-1], t_end)
        ])

    return event_times[event_times < t_end]


def ogata_thinning(
    lambda_: Callable[[float, np.ndarray], float],
    lambda_ub: float,
    t_start: float,
    t_end: float,
    strict: bool = True
):
    '''
    Perform Ogata's thinning algorithm to obtain samples from any point processes.

    Args:
        lambda_: Callable[[float, np.ndarray], float]: The intensity function of the point process.
        lambda_ub (float): The upper bound for the intensity.
        t_start (float): The start time of the interval.
        t_end (float): The end time of the interval.
        strict (bool): Whether to enforce strict upper bound checking. If ``True``, any proposal with intensity exceeding `lambda_ub` will raise an error.

    Returns:
        np.ndarray: The accepted sample times.
    '''
    # Sanity checks
    if lambda_ub <= 0:
        raise ValueError('Upper bound lambda_ub must be positive.')
    if t_start >= t_end:
        raise ValueError('Invalid time interval: t_start must be less than t_end.')

    # Step 1: Sample from the proposal distribution (homogeneous Poisson process)
    proposal_times = sample_hpp(lambda_ub, t_start, t_end)
    u = np.random.rand(proposal_times.shape[0])
    accepted_samples = np.zeros_like(proposal_times)
    n = 0

    for i in range(proposal_times.shape[0]):
        hist = accepted_samples[:n]
        lambda_value = float(lambda_(proposal_times[i].item(), hist))
        if lambda_value > lambda_ub and strict:
            raise ValueError('lambda_ exceeds upper bound lambda_ub.')
        p = min(1, lambda_value / lambda_ub)
        if u[i] < p:
            accepted_samples[n] = proposal_times[i]
            n += 1

    # Step 3: Accept or reject samples
    accepted_samples = accepted_samples[:n]

    return accepted_samples

def ogata_thinning_adaptive(
    lambda_: Callable[[float, np.ndarray], float],
    lambda_ub: Callable[[float, np.ndarray], Tuple[float, float]],
    t_start: float,
    t_end: float
):
    '''
    Perform adaptive Ogata's thinning algorithm.

    Args:
        lambda_: Callable[[float, np.ndarray], float]: The intensity function of the point process.
        lambda_ub: Callable[[float, np.ndarray], Tuple[float, float]]: The upper bound function that returns ``(upper_bound, t_expiry)``.
        t_start (float): The start time of the interval.
        t_end (float): The end time of the interval.

    Returns:
        np.ndarray: The accepted sample times.
    '''
    if t_start >= t_end:
        raise ValueError('Invalid time interval: t_start must be less than t_end.')

    accepted_events = []
    t = t_start

    while t < t_end:
        accepted = False
        hist = np.array(accepted_events)
        ub, t_expiry = lambda_ub(t, hist)

        if ub <= 0:
            t = t_expiry # No event needed
            continue

        proposal_times = sample_hpp(ub, t, t_expiry)
        u = np.random.rand(proposal_times.shape[0])

        for i in range(proposal_times.shape[0]):
            lambda_value = float(lambda_(proposal_times[i].item(), hist))
            if lambda_value > ub:
                raise ValueError('Proposed intensity exceeds upper bound.')
            p = lambda_value / ub
            if u[i] < p:
                accepted_events.append(proposal_times[i])
                t = proposal_times[i].item()
                accepted = True
                break

        if not accepted:
            t = t_expiry

    ret = np.array(accepted_events)
    return ret[ret < t_end]