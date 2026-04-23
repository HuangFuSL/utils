from typing import Any, Sequence

import torch

from .. import BaseHook, LoopControl


class FillHyperParamHook(BaseHook):
    def __init__(self, **hyperparams: Any):
        self.hyperparams = hyperparams

    def before_stage(self) -> LoopControl | None:
        self.parent.global_context.hyperparams.update(self.hyperparams)

class OptunaHook(BaseHook):
    def __init__(
        self, storage: str, study_name: str,
        target_metric: str, maximize: bool = True,
        **hyperparams: Sequence[Any]
    ):
        import optuna
        self.optuna = optuna
        self.study = optuna.create_study(
            storage=storage, study_name=study_name,
            direction='maximize' if maximize else 'minimize',
            load_if_exists=True
        )
        self.trial = None
        self.target_metric = target_metric
        self.hyperparams = hyperparams
        self.generated_hyperparams = {}

    def check_stage(self):
        if self.trial is not None:
            return
        self.generated_hyperparams.clear()
        self.trial = self.study.ask()
        for name, values in self.hyperparams.items():
            sampled_value = self.trial.suggest_categorical(name, list(values))
            self.generated_hyperparams[name] = sampled_value

    def before_stage(self) -> LoopControl | None:
        self.parent.global_context.hyperparams.update(
            self.generated_hyperparams
        )

    def finalize_stage(self):
        try:
            if self.trial is None:
                return
            _, _, metrics = self.parent.global_context.metrics[-1]

            target_value = metrics[self.target_metric]
            if torch.is_tensor(target_value):
                target_value = target_value.item()
            else:
                target_value = float(target_value)
            self.study.tell(self.trial, target_value)
        except Exception:
            if self.trial is not None:
                self.study.tell(self.trial, float('nan'))
        finally:
            self.trial = None
            self.generated_hyperparams.clear()
