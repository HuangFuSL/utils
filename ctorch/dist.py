import torch
import torch.distributions.constraints

class Deterministic(torch.distributions.Distribution):
    has_rsample = True

    def __init__(self, loc, eps, validate_args=None):
        self.loc = loc
        self.eps = torch.as_tensor(eps, dtype=loc.dtype, device=loc.device)
        if loc.ndim > 0:
            batch_shape = loc.shape[:-1]
            event_shape = loc.shape[-1:]
        else:
            batch_shape = loc.shape
            event_shape = torch.Size()
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape, validate_args=validate_args
        )

    @property
    def arg_constraints(self):
        return {
            'loc': torch.distributions.constraints.real,
            'eps': torch.distributions.constraints.positive
        }

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return torch.zeros_like(self.loc)

    @property
    def entropy(self):
        # Zero entropy for deterministic distribution
        return self.loc.new_zeros(self.batch_shape)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Deterministic, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape + self.event_shape)
        new.eps = self.eps.expand(batch_shape + self.event_shape)
        super(Deterministic, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return self.loc.expand(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return self.loc.expand(shape)

    def log_prob(self, value):
        lp = torch.where(
            (value - self.loc).abs() < self.eps,
            torch.full_like(value, float('inf')),
            torch.full_like(value, float('-inf'))
        )
        # log_prob must sum over event dims, returning a scalar per sample
        if self._event_shape:
            for _ in range(len(self._event_shape)):
                lp = lp.sum(dim=-1)
        return lp
