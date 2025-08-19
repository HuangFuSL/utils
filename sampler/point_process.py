'''
`utils.sampler.point_process` - Point Process Simulation
'''
from typing import Callable, Tuple
import numpy as np
import abc

def integrated_intensity(
    lambda_: Callable[[float, np.ndarray], float],
    event_times: np.ndarray,
    t_start: float = 0,
    t_end: float | None = None,
    n_points: int = 10
):
    '''
    Compute the integrated intensity function at given event times from intensity function lambda_

    .. math::
        \\Lambda^\\ast (t) = \\int_{0}^{t} \\lambda^\\ast (u) du

    Args:
        lambda_ (Callable[[float, np.ndarray], float]): The intensity function of the point process.
        event_times (np.ndarray): Shape (N,), The event times at which to compute the integrated intensity.
        t_start (float): The start time of the point process.
        t_end (float, optional): The end time of the point process. If not provided, it will be set to the last event time.
        n_points (int): Number of points to use for numerical integration.

    Returns:
        np.ndarray: The integrated intensity function evaluated at the event times. If `t_end` is None, the length should be (N,), otherwise (N + 1,)
    '''
    # Sanity checks
    if not np.all(np.diff(event_times) > 0):
        raise ValueError('Event times must be strictly increasing.')
    if t_end is None:
        t_end = event_times[-1]
        integrals = np.zeros_like(event_times)
    else:
        if t_end < event_times[-1]:
            raise ValueError('t_end must be greater than the last event time.')
        integrals = np.zeros(event_times.shape[0] + 1)
    assert isinstance(t_end, float)

    event_range = zip(
        [t_start, *event_times],
        [*event_times, t_end]
    )
    for i, (start, end) in enumerate(event_range):
        # Compute the integrated intensity at each event time
        if end <= start:
            continue
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
    history: np.ndarray | None = None,
    strict: bool = True
):
    '''
    Perform Ogata's thinning algorithm to obtain samples from any point processes.

    Args:
        lambda\\_: Callable[[float, np.ndarray], float]: The intensity function of the point process.
        lambda\\_ub (float): The upper bound for the intensity.
        t_start (float): The start time of the interval.
        t_end (float): The end time of the interval.
        history (np.ndarray, optional): The history of events before t_start. Defaults to None.
        strict (bool): Whether to enforce strict upper bound checking. If ``True``, any proposal with intensity exceeding `lambda_ub` will raise an error.

    Returns:
        np.ndarray: The accepted sample times.
    '''
    # Sanity checks
    if lambda_ub <= 0:
        raise ValueError('Upper bound lambda_ub must be positive.')
    if t_start >= t_end:
        raise ValueError('Invalid time interval: t_start must be less than t_end.')
    if history is None:
        history = np.array([])

    # Step 1: Sample from the proposal distribution (homogeneous Poisson process)
    proposal_times = sample_hpp(lambda_ub, t_start, t_end)
    u = np.random.rand(proposal_times.shape[0])
    accepted_samples = np.concatenate([
        history,
        np.full_like(proposal_times, fill_value=t_end)
    ])
    n = history.shape[0]

    for i in range(proposal_times.shape[0]):
        hist = accepted_samples[accepted_samples < proposal_times[i]]
        lambda_value = float(lambda_(proposal_times[i].item(), hist))
        if lambda_value > lambda_ub and strict:
            print(lambda_value, lambda_ub)
            raise ValueError('lambda_ exceeds upper bound lambda_ub.')
        p = min(1, lambda_value / lambda_ub)
        if u[i] < p:
            accepted_samples[n] = proposal_times[i]
            n += 1

    # Step 3: Accept or reject samples
    accepted_samples = accepted_samples[accepted_samples < t_end]

    return accepted_samples

def ogata_thinning_adaptive(
    lambda_: Callable[[float, np.ndarray], float],
    lambda_ub: Callable[[float, np.ndarray], Tuple[float, float]],
    t_start: float,
    t_end: float,
    history: np.ndarray | None = None
):
    '''
    Perform adaptive Ogata's thinning algorithm.

    Args:
        lambda_: Callable[[float, np.ndarray], float]: The intensity function of the point process.
        lambda_ub: Callable[[float, np.ndarray], Tuple[float, float]]: The upper bound function that returns ``(upper_bound, t_expiry)``.
        t_start (float): The start time of the interval.
        t_end (float): The end time of the interval.
        history (np.ndarray, optional): The history of events before t_start. Defaults to None.

    Returns:
        np.ndarray: The accepted sample times.
    '''
    if t_start >= t_end:
        raise ValueError('Invalid time interval: t_start must be less than t_end.')
    if history is None:
        history = np.array([])

    accepted_events = []
    t = t_start

    while t < t_end:
        accepted = False
        hist = np.concatenate([history, np.array(accepted_events)])
        hist = hist[hist < t]
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

class IntensityFunction(abc.ABC):
    def __init__(*args, **kwargs):
        pass

    @abc.abstractmethod
    def lambda_(self, t: float, history: np.ndarray) -> float:
        '''
        The intensity function at time t given the history of events.

        Args:
            t (float): The time at which to evaluate the intensity function.
            history (np.ndarray): The history of events up to time t.

        Returns:
            float: The intensity at time t.
        '''
        ...

    @abc.abstractmethod
    def lambda_upperbound(self, t: float, history: np.ndarray) -> Tuple[float, float]:
        '''
        The upper bound of the intensity function at time t given the history of events.

        Args:
            t (float): The time at which to evaluate the upper bound of the intensity function.
            history (np.ndarray): The history of events up to time t.

        Returns:
            Tuple[float, float]: The upper bound and expiry time of the upper bound.
        '''
        ...

    def sample(self, t_start: float, t_end: float, history: np.ndarray | None = None) -> np.ndarray:
        '''
        Sample event times from the point process defined by the intensity function.

        Args:
            t_start (float): The start time of the interval.
            t_end (float): The end time of the interval.
            history (np.ndarray | None): The history of events before t_start.

        Returns:
            np.ndarray: The sampled event times.
        '''
        return ogata_thinning_adaptive(
            self.lambda_, self.lambda_upperbound,
            t_start, t_end, history
        )

class HawkesIntensity(IntensityFunction):
    '''
    Hawkes process intensity function.

    .. math::
        \\lambda(t) = \\mu + \\sum_{t_i < t} \alpha \\cdot e^{-\\beta (t - t_i)}

    Args:
        mu (float): The baseline intensity.
        alpha (float): The excitation coefficient.
        beta (float): The decay rate.
    '''
    def __init__(self, mu: float, alpha: float, beta: float):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def lambda_(self, t: float, history: np.ndarray) -> float:
        # Compute the intensity function using the Hawkes process formula
        return self.mu + self.alpha * np.sum(np.exp(-self.beta * (t - history[history < t])))

    def lambda_upperbound(self, t: float, history: np.ndarray) -> Tuple[float, float]:
        # Compute the upper bound using the Hawkes process formula
        return self.lambda_(t, history), t + 10
