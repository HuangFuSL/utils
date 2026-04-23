from typing import Any, Dict, List, Literal

import numpy as np
import torch

from .. import BaseHook, LoopControl

class TqdmHook(BaseHook):
    '''
    Hook to display a progress bar using tqdm during training.

    Args:
        unit: The unit of progress to track. Can be 'data_step', 'update_step', or 'epoch'.
    '''

    def __init__(
        self, unit: Literal['data_step', 'update_step', 'epoch'] = 'data_step',
        metrics_keys: List[str] | None = None, floatfmt: str = '.4g'
    ):
        self.unit = unit
        self.metrics_keys = metrics_keys
        self.floatfmt = floatfmt
        self._last_metrics_idx = -1

    def _refresh_metrics_postfix(self) -> None:
        if self.metrics_keys is None or not self.parent.global_context.metrics:
            return
        metrics_log = self.parent.global_context.metrics
        if len(metrics_log) - 1 == self._last_metrics_idx:
            return
        self._last_metrics_idx = len(metrics_log) - 1

        _, _, metrics = metrics_log[self._last_metrics_idx]
        if not isinstance(metrics, dict) or not metrics:
            return
        if self.metrics_keys is not None:
            metrics = {k: metrics[k]
                       for k in self.metrics_keys if k in metrics}

        postfix = {}
        for k, v in metrics.items():
            if isinstance(v, float):
                postfix[k] = format(v, self.floatfmt)
            else:
                postfix[k] = v

        self.tqdm.set_postfix(postfix, refresh=False)

    def before_stage(self) -> LoopControl | None:
        import tqdm
        self.tqdm = tqdm.tqdm(unit=' ' + self.unit)

    def before_step(self) -> LoopControl | None:
        if self.unit == 'data_step':
            self.tqdm.update(1)
            self._refresh_metrics_postfix()

    def after_optimizer_step(self) -> LoopControl | None:
        if self.unit == 'update_step':
            self.tqdm.update(1)
            self._refresh_metrics_postfix()

    def after_epoch(self) -> LoopControl | None:
        if self.unit == 'epoch':
            self.tqdm.update(1)
            self._refresh_metrics_postfix()

    def finalize_stage(self) -> LoopControl | None:
        if hasattr(self, "tqdm"):
            self.tqdm.close()

class WandBHook(BaseHook):
    '''
    Hook to log training metrics to Weights and Biases (wandb).
    The hook must be registered with the least priority (largest priority number)

    Items will be logged:

    - Losses from ``step_context['loss']`` and ``step_context['losses']``, both are PyTorch tensors.
    - Metrics gathered from ``global_context.metrics[-1]``.
    - Learning rates from ``optimizer.param_groups``.
    - Additional entries from ``step_context`` defined in ``additional_entries``

    PyTorch tensors are handled according to its shape:

    - Scalar tensors are logged as numbers.
    - 1D tensors are logged as histograms.
    - 2D tensors with shape (2, N) are logged as line plots, otherwise as black-white image.
    - 3D tensors with shape (C, H, W) where C is 1, 3 or 4 are logged as images. Other shapes are not supported.

    Args:
        project: The wandb project name.
        entity: The wandb entity (user or team) name.
        run_name: The name of the wandb run. If not provided, a random name will be generated.
        config: A dictionary of hyperparameters to be logged in wandb.
        flush_interval: The interval (in steps) at which to flush the logged data to wandb, to reduce device synchronization overhead.
        loss_keys: The names of the losses.
        optimizer_keys: The names of the optimizers.
        additional_entries: Additional entries in step_context to be logged, either a list of entry names (logged with the same name) or a dict of entry name and logged name pairs.
    '''
    def __init__(
        self, project: str, entity: str | None = None,
        run_name: str | None = None, config: dict | None = None,
        flush_interval: int = 1000,
        loss_keys: List[str] | None = None,
        optimizer_keys: List[str] | None = None,
        additional_entries: Dict[str, str] | List[str] | None = None
    ):
        import wandb
        self.wandb = wandb
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = config or {}
        self.flush_interval = flush_interval
        self.loss_keys = loss_keys or []
        self.optimizer_keys = optimizer_keys or []
        self.num_evaluations = 0

        self.pending_data: Dict[int, Dict[str, Any]] = {}
        if isinstance(additional_entries, list):
            self.additional_entries = {entry: entry for entry in additional_entries}
        else:
            self.additional_entries = additional_entries or {}

    def before_stage(self) -> LoopControl | None:
        self.wandb_run = self.wandb.init(
            project=self.project, entity=self.entity,
            name=self.run_name,
            config=self.config | self.parent.global_context.hyperparams
        )

    def _write_data(self, step: int, key: str, value: Any):
        if torch.is_tensor(value):
            value = value.detach() # Detach but not sync
        self.pending_data.setdefault(step, {})[key] = value

    def _sync_tensor(self, data: Dict[str, Any]):
        for k, v in data.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    data[k] = v.item()
                elif v.dim() == 1:
                    # Histogram
                    hist = np.histogram(v.cpu().numpy(), bins='auto')
                    data[k] = self.wandb.Histogram(np_histogram=hist)
                elif v.dim() == 2 and v.shape[0] == 2:
                    # Line plot
                    data[k] = self.wandb.plot.line_series(
                        xs=v[0].cpu().numpy(), ys=v[1].cpu().numpy()
                    )
                elif v.dim() == 2 or (v.dim() == 3 and v.shape[0] in [1, 3, 4]):
                    data[k] = self.wandb.Image(v.numpy(force=True), normalize=True) # CHW
                else:
                    raise ValueError(f'Unsupported tensor shape {v.shape} for logging.')
            else:
                data[k] = v
        return data

    def _flush(self):
        if hasattr(self, 'wandb_run'):
            for step in sorted(self.pending_data):
                data = self.pending_data[step]
                self.wandb_run.log(
                    self._sync_tensor(data), step=step, commit=True
                )
            self.pending_data.clear()

    def before_step(self) -> LoopControl | None:
        # commit data
        # Some metrics are logged in finalize_epoch, which happens after finalize_step
        # So, here must flush in before_step
        if self.pending_data and self.parent.global_context.step % self.flush_interval == 0:
            self._flush()

    def _add_metrics(self):
        if len(self.parent.global_context.metrics) == self.num_evaluations:
            # No new evaluation come in
            return
        *_, metrics = self.parent.global_context.metrics[-1]
        step = self.parent.global_context.step - 1 # Align with the step at which losses are computed
        for key, value in metrics.items():
            self._write_data(step, f'metrics/{key}', value)
        self.num_evaluations = len(self.parent.global_context.metrics)

    def finalize_optimizer_step(self) -> LoopControl | None:
        # Log learning rate
        step = self.parent.global_context.step
        if self.optimizer_keys and len(self.parent.optimizer) != len(self.optimizer_keys):
            raise ValueError('Unmatched number of optimizers.')
        for i, optim in enumerate(self.parent.optimizer):
            name = self.optimizer_keys[i] if self.optimizer_keys else i
            for j, param_group in enumerate(optim.param_groups):
                self._write_data(step, f'lr/{name}/{j}', param_group['lr'])

    def finalize_backward(self) -> LoopControl | None:
        step = self.parent.global_context.step
        if 'loss' in self.parent.step_context:
            self._write_data(step, 'loss', self.parent.step_context['loss'])
        if 'losses' in self.parent.step_context:
            losses = self.parent.step_context['losses'].reshape(-1).detach()
            if self.loss_keys and len(self.loss_keys) != losses.numel():
                raise ValueError('Unmatched number of losses.')
            for i in range(losses.numel()):
                name = self.loss_keys[i] if self.loss_keys else i
                self._write_data(step, f'losses/{name}', losses[i])

    def finalize_step(self) -> LoopControl | None:
        self._add_metrics()
        for entry, name in self.additional_entries.items():
            if entry in self.parent.step_context:
                self._write_data(
                    self.parent.global_context.step - 1,
                    name, self.parent.step_context[entry]
                )

    def finalize_epoch(self) -> LoopControl | None:
        # Log epoch-level metrics if any
        self._add_metrics()

    def finalize_stage(self) -> LoopControl | None:
        if hasattr(self, 'wandb_run'):
            self._flush()
            self.wandb_run.finish()
