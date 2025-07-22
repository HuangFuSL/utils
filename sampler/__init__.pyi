import numpy as np

def setup_threads(n: int) -> None:
    '''
    Configure the global thread-pool. First call wins; later calls are ignored.

    * `n` – number of threads to use globally

    This function must be called **after** forking, to avoid deadlocks.
    '''
    ...


def batched_choices_with_negative(
    range_: np.ndarray,  # ndarray of 1D integers
    exclude: np.ndarray,  # ndarray of 2D integers, nan as padding
    k: int  # int, number of samples to draw
) -> np.ndarray:
    '''
    Batched negative sampling, without replacement.

    * `range_`   – 1-D array containing the full candidate set (population)
    * `exclude`  – 2-D array; each row lists ids that **must not** be drawn for that sample (use a negative number / `np.nan` as padding)
    * `k`        – number of negatives to sample **per row**

    Returns an `(N, k)` ndarray where `N = exclude.shape[0]`.
    '''
    ...


def weighted_batched_choices_with_negative(
    range_: np.ndarray,  # ndarray of 1D integers
    weights: np.ndarray,  # ndarray of 1D floats, same length as range_
    exclude: np.ndarray,  # ndarray of 2D integers, nan as padding
    k: int  # int, number of samples to draw
) -> np.ndarray:
    '''
    Weighted batched negative sampling, with replacement.

    * `range_`   – 1-D array containing the full candidate set (population)
    * `weights`  – 1-D array of weights corresponding to the population; must be non-negative and same length as `range_`
    * `exclude`  – 2-D array; each row lists ids that **must not** be drawn for that sample (use a negative number / `np.nan` as padding)
    * `k`        – number of negatives to sample **per row**

    Returns an `(N, k)` ndarray where `N = exclude.shape[0]`.
    '''
    ...


def batched_sample_with_negative(
    range_: np.ndarray, # ndarray of 1D integers
    exclude: np.ndarray, # ndarray of 2D integers, nan as padding
    k: int # int, number of samples to draw
) -> np.ndarray:
    '''
    Batched negative sampling, with replacement.

    * `range_`   – 1-D array containing the full candidate set (population)
    * `exclude`  – 2-D array; each row lists ids that **must not** be drawn for that sample (use a negative number / `np.nan` as padding)
    * `k`        – number of negatives to sample **per row**

    Returns an `(N, k)` ndarray where `N = exclude.shape[0]`.
    '''
    ...


def weighted_batched_sample_with_negative(
    range_: np.ndarray, # ndarray of 1D integers
    weights: np.ndarray, # ndarray of 1D floats, same length as range_
    exclude: np.ndarray, # ndarray of 2D integers, nan as padding
    k: int # int, number of samples to draw
) -> np.ndarray:
    '''
    Weighted batched negative sampling, without replacement.

    * `range_`   – 1-D array containing the full candidate set (population)
    * `weights`  – 1-D array of weights corresponding to the population; must be non-negative and same length as `range_`
    * `exclude`  – 2-D array; each row lists ids that **must not** be drawn for that sample (use a negative number / `np.nan` as padding)
    * `k`        – number of negatives to sample **per row**

    Returns an `(N, k)` ndarray where `N = exclude.shape[0]`.
    '''
    ...
