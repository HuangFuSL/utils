import logging
from pathlib import Path
import sys

from .hashing import HashPolicy
from .pipeline import PipelineConfig, ScriptSpec
from .storage import RunDB

logger = logging.getLogger(__name__)

def compare_input(spec: ScriptSpec, db: RunDB, hash_policy: HashPolicy) -> bool:
    try:
        input_real = spec.input_hash(hash_policy)
        met = set(input_real.keys())
        for _, input_version in db.get_input(spec.id).items():
            met -= set(input_version.keys())
            for k, v in input_version.items():
                assert input_real[k] == v
        return not bool(met)
    except:
        return False

def compare_output(spec: ScriptSpec, db: RunDB, hash_policy: HashPolicy) -> bool:
    try:
        output_real = spec.output_hash(hash_policy)
        output_recorded = db.get_output(spec.id)
        met = set(output_real.keys()) - set(output_recorded.keys())
        for k, v in output_recorded.items():
            assert output_real[k] == v
        return not bool(met)
    except:
        return False

class HashTracker:
    def __init__(
        self,
        config: PipelineConfig,
        script_id: str,
        hash_policy: HashPolicy = HashPolicy.SAMPLE,
    ):
        self.config = config
        self.script_id = script_id
        self.hash_policy = hash_policy

        if script_id not in config.scripts:
            raise ValueError(
                f"Script '{script_id}' not found in pipeline '{config.name}'"
            )
        self.spec = config.scripts[script_id]
        self._upstream_ids = config._graph.get(script_id, set())
        self.db = RunDB(config.db_path, self.script_id)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        script_id: str,
        hash_policy: HashPolicy = HashPolicy.SAMPLE,
    ) -> 'HashTracker':
        config = PipelineConfig.from_yaml(yaml_path)
        return cls(config, script_id, hash_policy)

    def __enter__(self) -> 'HashTracker':
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.finish()

    def start(self) -> None:
        with self.db:
            if not self.upstream_ready:
                raise RuntimeError('Upstream is not ready.')
            self.db.set_input(self.spec.input_hash(self.hash_policy))
            self.db.commit()

    def finish(self) -> None:
        with self.db:
            self.db.set_output(self.spec.output_hash(self.hash_policy))
            self.db.commit()

    def skip(self):
        if not self.upstream_ready:
            raise RuntimeError('Upstream is not ready.')
        if self.can_skip:
            sys.exit(0)

    def skip_or_start(self):
        self.skip()
        self.start()

    @property
    def can_skip(self) -> bool:
        pool = [self.spec]
        while pool:
            spec = pool.pop(0)
            if not compare_input(spec, self.db, self.hash_policy):
                return False
            for uid in self.config._graph.get(spec.id, set()):
                pool.append(self.config.scripts[uid])

        return True

    @property
    def upstream_ready(self) -> bool:
        for uid in self._upstream_ids:
            if not compare_output(self.config.scripts[uid], self.db, self.hash_policy):
                return False
        return True
