use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use once_cell::sync::OnceCell;
use ordered_float::OrderedFloat;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::Array2 as NdArray2;

use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::Rng;
use rayon::prelude::*;


static RAYON_INIT: OnceCell<()> = OnceCell::new();

/// Configure the global thread-pool. First call wins; later calls are ignored.
///
/// * `n` – number of threads to use globally
///
/// This function must be called **after** forking, to avoid deadlocks.
#[pyfunction]
fn setup_threads(n: usize) -> PyResult<()> {
    RAYON_INIT.get_or_try_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build thread-pool: {e}")))?;
        Ok::<(), PyErr>(())
    })?;
    Ok(())
}

/// Batched negative sampling, with replacement.
///
/// * `range_`   – 1-D array containing the full candidate set (population)
/// * `exclude`  – 2-D array; each row lists ids that **must not** be drawn for that sample (use a negative number / `np.nan` as padding)
/// * `k`        – number of negatives to sample **per row**
///
/// Returns an `(N, k)` ndarray where `N = exclude.shape[0]`.
#[pyfunction]
fn batched_sample_with_negative<'py>(
    py: Python<'py>,
    range_: PyReadonlyArray1<'py, i64>,
    exclude: PyReadonlyArray2<'py, i64>,
    k: usize,
) -> PyResult<Py<PyArray2<i64>>> {
    // Validate inputs
    let population = range_.as_slice()?.to_vec();
    let excl_view = exclude.as_array();
    let n_rows = excl_view.nrows();
    let pop_len = population.len();

    // Sanity checks
    if k == 0 {
        return Err(PyValueError::new_err("k must be >= 1"));
    }
    // Assert all the population values are non-negative
    if population.iter().any(|&v| v < 0) {
        return Err(PyValueError::new_err("population must contain only non-negative values"));
    }

    // Rejection sampling
    // Optimal when k and exclude are small relative to the population size.
    let all_rows: Vec<Vec<i64>> = py.allow_threads(|| {
        (0..n_rows)
            .into_par_iter()
            .map(|row_idx| {
                // Build per-row exclusion set
                let mut skip = HashSet::with_capacity(excl_view.ncols());
                for &v in excl_view.row(row_idx) {
                    if v >= 0 {
                        skip.insert(v);
                    }
                }

                // Sample k unique negatives with replacement
                let mut rng = rand::rng();
                let mut chosen = Vec::with_capacity(k);
                while chosen.len() < k {
                    let candidate = population[rng.random_range(0..pop_len)];
                    if !skip.contains(&candidate) && !chosen.contains(&candidate) {
                        chosen.push(candidate);
                    }
                }
                chosen
            })
            .collect()
    });

    // Assemble result ndarray
    let mut out = NdArray2::<i64>::zeros((n_rows, k));
    for (i, row) in all_rows.into_iter().enumerate() {
        for (j, v) in row.into_iter().enumerate() {
            out[(i, j)] = v;
        }
    }

    // Convert to Python object
    Ok(PyArray2::from_owned_array(py, out).to_owned().into())
}

/// Weighted batched negative sampling, with replacement.
///
/// * `range_`   – 1-D array containing the full candidate set (population)
/// * `weights`  – 1-D array of weights corresponding to the population; must be non-negative and same length as `range_`
/// * `exclude`  – 2-D array; each row lists ids that **must not** be drawn for that sample (use a negative number / `np.nan` as padding)
/// * `k`        – number of negatives to sample **per row**
///
/// Returns an `(N, k)` ndarray where `N = exclude.shape[0]`.
#[pyfunction]
fn weighted_batched_choices_with_negative<'py>(
    py: Python<'py>,
    range_: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
    exclude: PyReadonlyArray2<'py, i64>,
    k: usize,
) -> PyResult<Py<PyArray2<i64>>> {
    // Type conversions
    let population: Vec<i64> = range_.as_slice()?.to_owned();
    let weights: Vec<f64> = weights.as_slice()?.to_owned();
    let excl_view = exclude.as_array();
    let pop_len = population.len();
    let n_rows = excl_view.nrows();

    // Sanity checks
    if k == 0 {
        return Err(PyValueError::new_err("k must be >= 1"));
    }
    if population.iter().any(|&v| v < 0) {
        return Err(PyValueError::new_err("population must contain only non-negative values"));
    }
    if weights.len() != pop_len {
        return Err(PyValueError::new_err("weights must match population size"));
    }
    if weights.iter().any(|&w| w < 0.0) {
        return Err(PyValueError::new_err("weights must be non-negative"));
    }

    // Build index map for fast lookups
    let idx_of: HashMap<i64, usize> = population
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    let result: Vec<Vec<i64>> = py.allow_threads(|| {
        (0..n_rows).into_par_iter().map(|row| {
            // We can directly use WeightedIndex for weighted sampling
            let mut row_weight = weights.clone();
            // Unset weights for excluded items
            for &val in excl_view.row(row) {
                if let Some(&ix) = idx_of.get(&val) {
                    row_weight[ix] = 0.0; // Set weight to 0 for excluded
                }
            }
            // Sanity check: ensure we have at least 1 valid item to sample from
            let valid_count = row_weight.iter().filter(|&&w| w > 0.0).count();
            if valid_count == 0 {
                return Err(PyValueError::new_err(format!(
                    "row {row} has no selectable elements after exclusion"
                )));
            }
            let dist = WeightedIndex::new(&row_weight).expect("weight error");
            let mut rng = rand::rng();
            let result = (0..k).map(|_| population[dist.sample(&mut rng)]).collect();
            Ok(result)
        }).collect::<PyResult<Vec<Vec<i64>>>>()
    })?;

    // Assemble result ndarray
    let mut out = NdArray2::<i64>::zeros((n_rows, k));
    for (i, row) in result.into_iter().enumerate() {
        for (j, v) in row.into_iter().enumerate() {
            out[(i, j)] = v;
        }
    }

    // Convert to Python object
    Ok(PyArray2::from_owned_array(py, out).to_owned().into())
}

/// Batched negative sampling, without replacement.
///
/// * `range_`   – 1-D array containing the full candidate set (population)
/// * `exclude`  – 2-D array; each row lists ids that **must not** be drawn for that sample (use a negative number / `np.nan` as padding)
/// * `k`        – number of negatives to sample **per row**
///
/// Returns an `(N, k)` ndarray where `N = exclude.shape[0]`.
#[pyfunction]
fn batched_choices_with_negative<'py>(
    py: Python<'py>,
    range_: PyReadonlyArray1<'py, i64>,
    exclude: PyReadonlyArray2<'py, i64>,
    k: usize,
) -> PyResult<Py<PyArray2<i64>>> {
    // Validate inputs
    let population = range_.as_slice()?.to_vec();
    let excl_view = exclude.as_array();
    let n_rows = excl_view.nrows();
    let pop_len = population.len();

    if k == 0 {
        return Err(PyValueError::new_err("k must be >= 1"));
    }
    if k > pop_len {
        return Err(PyValueError::new_err("k cannot exceed population size"));
    }
    // Assert all the population values are non-negative
    if population.iter().any(|&v| v < 0) {
        return Err(PyValueError::new_err("population must contain only non-negative values"));
    }

    // Perform the work without the GIL & in parallel
    let all_rows: Vec<Vec<i64>> = py.allow_threads(|| {
        (0..n_rows)
            .into_par_iter()
            .map(|row_idx| {
                // Build per-row exclusion set
                let mut skip = HashSet::with_capacity(excl_view.ncols());
                for &v in excl_view.row(row_idx) {
                    if v >= 0 {
                        skip.insert(v);
                    }
                }

                // Rejection sampling until we have k unique negatives
                let mut rng = rand::rng();
                let mut chosen = Vec::with_capacity(k);
                while chosen.len() < k {
                    let candidate = population[rng.random_range(0..pop_len)];
                    if !skip.contains(&candidate) && !chosen.contains(&candidate) {
                        chosen.push(candidate);
                    }
                }
                chosen
            })
            .collect()
    });

    // Assemble result ndarray
    let mut out = NdArray2::<i64>::zeros((n_rows, k));
    for (i, row) in all_rows.into_iter().enumerate() {
        for (j, v) in row.into_iter().enumerate() {
            out[(i, j)] = v;
        }
    }

    // Convert to Python object
    Ok(PyArray2::from_owned_array(py, out).to_owned().into())
}

/// Weighted batched negative sampling, without replacement.
///
/// * `range_`   – 1-D array containing the full candidate set (population)
/// * `weights`  – 1-D array of weights corresponding to the population; must be non-negative and same length as `range_`
/// * `exclude`  – 2-D array; each row lists ids that **must not** be drawn for that sample (use a negative number / `np.nan` as padding)
/// * `k`        – number of negatives to sample **per row**
///
/// Returns an `(N, k)` ndarray where `N = exclude.shape[0]`.
#[pyfunction]
fn weighted_batched_sample_with_negative<'py>(
    py: Python<'py>,
    range_: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
    exclude: PyReadonlyArray2<'py, i64>,
    k: usize,
) -> PyResult<Py<PyArray2<i64>>> {
    // Type conversions
    let population: Vec<i64> = range_.as_slice()?.to_owned();
    let weights: Vec<f64>    = weights.as_slice()?.to_owned();
    let excl_view = exclude.as_array();
    let pop_len = population.len();
    let n_rows    = excl_view.nrows();

    // Sanity checks
    // 1. Assert k is valid
    if k == 0 {
        return Err(PyValueError::new_err("k must be >= 1"));
    }
    if k > pop_len {
        return Err(PyValueError::new_err("k cannot exceed population size"));
    }

    // 2. Assert all the population values are non-negative
    if population.iter().any(|&v| v < 0) {
        return Err(PyValueError::new_err("population must contain only non-negative values"));
    }

    // 3. Assert weights are valid
    if weights.len() != pop_len {
        return Err(PyValueError::new_err("weights must match population size"));
    }
    if weights.iter().any(|&w| w < 0.0) {
        return Err(PyValueError::new_err("weights must be non-negative"));
    }

    let idx_of: HashMap<i64, usize> = population
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();


    let result: Vec<Vec<i64>> = py.allow_threads(|| {
        (0..n_rows).into_par_iter().map(|row| {
            // Build exclude set
            let mut skip: HashSet<usize> = HashSet::new();
            for &val in excl_view.row(row) {
                if let Some(&ix) = idx_of.get(&val) {
                    skip.insert(ix);
                }
            }
            let avail = pop_len - skip.len();
            if avail < k {
                return Err(PyValueError::new_err(format!(
                    "row {row} has not enough selectable elements after exclusion"
                )));
            }

            let mut heap: BinaryHeap<Reverse<(OrderedFloat<f64>, usize)>> =
                BinaryHeap::with_capacity(k);
            let mut rng = rand::rng();

            for i in 0..pop_len {
                if skip.contains(&i) || weights[i] == 0.0 {
                    continue;
                }
                // K_i = U^{1/w_i}
                let u: f64  = rng.random::<f64>().max(f64::MIN_POSITIVE); // 避免 0
                let key: f64 = u.powf(1.0 / weights[i]);

                if heap.len() < k {
                    heap.push(Reverse((OrderedFloat(key), i)));
                } else if key > heap.peek().unwrap().0 .0.into_inner() {
                    heap.pop();
                    heap.push(Reverse((OrderedFloat(key), i)));
                }
            }

            // Return the indices of the sampled items
            let result = heap.into_iter()
                .map(|rev| population[rev.0 .1])
                .collect();
            Ok(result)
        }).collect::<PyResult<Vec<Vec<i64>>>>()
    })?;

    // Assemble result ndarray
    let mut out = NdArray2::<i64>::zeros((n_rows, k));
    for (i, row) in result.into_iter().enumerate() {
        for (j, v) in row.into_iter().enumerate() {
            out[(i, j)] = v;
        }
    }
    // Convert to Python object
    Ok(PyArray2::from_owned_array(py, out).to_owned().into())
}

/// Python module definition
#[pymodule]
fn _rs_sampler(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(setup_threads, m)?)?;
    m.add_function(wrap_pyfunction!(batched_choices_with_negative, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_batched_choices_with_negative, m)?)?;
    m.add_function(wrap_pyfunction!(batched_sample_with_negative, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_batched_sample_with_negative, m)?)?;
    Ok(())
}
