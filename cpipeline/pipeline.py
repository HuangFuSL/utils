from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

from .hashing import HashPolicy, hash_entry


@dataclass
class ScriptSpec:
    '''A single script node in the pipeline DAG.'''
    id: str
    path: str
    script_root: Path
    artifact_path: Path
    description: str = ''
    args: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)

    @property
    def script_path(self) -> str:
        return ' '.join([
            str(self.script_root / self.path),
            *self.args
        ])

    @property
    def hash(self) -> Dict[str, str]:
        return hash_entry(self.path, self.script_root, HashPolicy.FULL)

    def input_hash(self, hash_policy: HashPolicy) -> Dict[str, str]:
        return hash_entry(self.inputs, self.artifact_path, hash_policy) | self.hash

    def output_hash(self, hash_policy: HashPolicy) -> Dict[str, str]:
        return hash_entry(self.outputs, self.artifact_path, hash_policy)

@dataclass
class PipelineConfig:
    '''Parsed pipeline configuration with dependency graph resolved.'''
    name: str
    script_root: Path
    artifact_path: Path
    db_path: Path
    scripts: Dict[str, ScriptSpec] = field(default_factory=dict)
    _graph: Dict[str, Set[str]] = field(default_factory=dict)

    @property
    def topological_order(self) -> List[str]:
        '''Return a topological ordering of the scripts in the DAG.'''
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {sid: WHITE for sid in self._graph}
        order: List[str] = []

        def dfs(node: str) -> None:
            color[node] = GRAY
            for neighbor in self._graph.get(node, set()):
                if color[neighbor] == GRAY:
                    raise ValueError(f'Cycle detected in pipeline DAG.')
                if color[neighbor] == WHITE:
                    dfs(neighbor)
            color[node] = BLACK
            order.append(node)

        for sid in self._graph:
            if color[sid] == WHITE:
                dfs(sid)

        return order

    @property
    def execution_order(self) -> List[str]:
        '''Return a topological ordering of the scripts in the DAG.'''
        return [self.scripts[sid].path for sid in self.topological_order]

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'PipelineConfig':
        '''
        Load a pipeline YAML, resolve upstream dependencies, validate DAG.

        The dependency graph is built by:
        1. Auto-infer: script B's input that matches script A's output → A is upstream of B.
        2. Explicit: honour each script's ``depends_on`` list.

        Raises:
            ValueError: if a cycle is detected or ``depends_on`` references an unknown id.
        '''
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        pipe = data['pipeline']
        config = PipelineConfig(
            name=pipe['name'],
            script_root=Path(pipe['script_root']).resolve(),
            artifact_path=Path(pipe['artifact_path']).resolve(),
            db_path=Path(pipe['db_path']).resolve(),
        )

        for s in data.get('scripts', []):
            spec = ScriptSpec(
                id=s['id'],
                script_root=config.script_root,
                path=s['path'],
                artifact_path=config.artifact_path,
                description=s.get('description', ''),
                args=s.get('args', []),
                inputs=s.get('inputs', []),
                outputs=s.get('outputs', []),
                depends_on=s.get('depends_on', []),
            )
            config.scripts[spec.id] = spec

        config.build_graph()
        config.topological_order # trigger cycle detection

        return config

    def build_graph(self) -> None:
        output_map: Dict[str, List[str]] = {}
        for sid, spec in self.scripts.items():
            for out in spec.outputs:
                norm = out.rstrip('/')
                output_map.setdefault(norm, []).append(sid)

        for sid, spec in self.scripts.items():
            upstream: Set[str] = set()
            for inp in spec.inputs:
                norm = inp.rstrip('/')
                if any(c in norm for c in '*?['):
                    continue
                for producer in output_map.get(norm, []):
                    if producer != sid:
                        upstream.add(producer)
            upstream.update(spec.depends_on)
            self._graph[sid] = upstream

        for sid in self.scripts:
            if set(self._graph[sid]) - set(self.scripts.keys()):
                raise ValueError(f"Script '{sid}' depends_on unknown script.")
            if sid in self._graph[sid]:
                raise ValueError(f"Script '{sid}' depends_on itself.")
