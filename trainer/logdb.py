import abc
import dataclasses
import itertools
import json
import sqlite3
import time
import uuid
from types import UnionType
from typing import Any, Callable, ClassVar, Dict, List, Tuple, Union, get_args, get_origin

NoneType = type(None)


@dataclasses.dataclass(frozen=True, slots=True)
class ForeignKey:
    fields: Tuple[str, ...]
    ref_table: str
    ref_fields: Tuple[str, ...]
    on_delete: str | None = None
    on_update: str | None = None


@dataclasses.dataclass(slots=True)
class BaseRecord(abc.ABC):
    __table_name__: ClassVar[str] = ' '
    __primary_key__: ClassVar[Tuple[str, ...]] = ()
    __indexes__: ClassVar[Tuple[Tuple[str, ...], ...]] = ()
    __update_fields__: ClassVar[Tuple[str, ...]] = ()
    __foreign_keys__: ClassVar[Tuple[ForeignKey, ...]] = ()

    @staticmethod
    def resolve_field_type(field_type: Any) -> Tuple[type, bool]:
        def normalize(tp: Any) -> type:
            origin = get_origin(tp)
            if origin is not None:
                tp = origin
            if tp not in (int, float, str, bool, list, dict):
                raise ValueError(f'Unsupported field type: {field_type}')
            return tp

        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin in (Union, UnionType):
            non_none_args = [arg for arg in args if arg is not NoneType]
            has_none = len(non_none_args) != len(args)
            if has_none and len(non_none_args) == 1:
                return normalize(non_none_args[0]), True
            raise ValueError(f'Unsupported union field type: {field_type}')

        return normalize(field_type), False

    @staticmethod
    def py_type_to_sql_type(field_type: Any) -> str:
        base_type, _ = BaseRecord.resolve_field_type(field_type)
        if base_type in (int, bool):
            return 'INTEGER'
        elif base_type == float:
            return 'REAL'
        elif base_type in (str, dict, list):
            return 'TEXT'
        else:
            raise ValueError(f'Unsupported field type: {field_type}')

    @staticmethod
    def sql_to_py(
        field_name: str, field_type: type | str | Any, value: Any
    ) -> Any:
        base_type, _ = BaseRecord.resolve_field_type(field_type)
        if value is None:
            return None
        if base_type == bool:
            return bool(int(value))
        elif base_type == int:
            return int(value)
        elif base_type == float:
            return float(value)
        elif base_type == str:
            return str(value)
        elif base_type in (dict, list):
            return json.loads(value)
        else:
            raise ValueError(
                f'Unsupported field type for {field_name}: {field_type}'
            )

    @staticmethod
    def py_to_sql(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        if isinstance(value, bool):
            return int(value)
        if not isinstance(value, (int, float, str, type(None))):
            raise ValueError(f'Unsupported value type: {type(value)}')
        return value

    @classmethod
    def test_table_exists(cls, conn: sqlite3.Connection) -> bool:
        cursor = conn.execute('''
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
        ''', (cls.__table_name__,))
        return cursor.fetchone() is not None

    @classmethod
    def create_table(cls, conn: sqlite3.Connection):
        fields = {
            field.name: cls.resolve_field_type(field.type)
            for field in dataclasses.fields(cls)
            if not field.name.startswith('_')
        }
        # Verify FK
        for fk in cls.__foreign_keys__:
            if len(fk.fields) != len(fk.ref_fields):
                raise ValueError(
                    f'Foreign key fields mismatch: {fk.fields} -> {fk.ref_fields}'
                )
            for field_name in fk.fields:
                if field_name not in fields:
                    raise ValueError(
                        f'Foreign key field {field_name} not found in {cls.__table_name__}'
                    )
        # Create table
        columns = []
        for name, (field_type, nullable) in fields.items():
            if nullable and name in cls.__primary_key__:
                raise ValueError(f'Primary key field {name} cannot be nullable')
            sql_type = cls.py_type_to_sql_type(field_type)
            suffix = ' NOT NULL' if not nullable else ''
            columns.append(f'{name} {sql_type} {suffix}'.strip())
        columns_sql = ', '.join(columns)
        pk_cols = ', '.join(cls.__primary_key__)

        fk_clauses = []
        for fk in cls.__foreign_keys__:
            fk_cols = ', '.join(fk.fields)
            ref_cols = ', '.join(fk.ref_fields)
            fk_clause = f'FOREIGN KEY ({fk_cols}) REFERENCES {fk.ref_table} ({ref_cols})'
            if fk.on_delete:
                fk_clause += f' ON DELETE {fk.on_delete}'
            if fk.on_update:
                fk_clause += f' ON UPDATE {fk.on_update}'
            fk_clauses.append(fk_clause)
        if fk_clauses:
            fk_sql = ', ' + ', '.join(fk_clauses)
        else:
            fk_sql = ''

        conn.execute(f'''
        CREATE TABLE IF NOT EXISTS {cls.__table_name__} (
            {columns_sql},
            PRIMARY KEY ({pk_cols})
            {fk_sql}
        )
        ''')

        # Create indexes
        for index_fields in cls.__indexes__:
            index_name = f'{cls.__table_name__}_{"_".join(index_fields)}_idx'
            index_cols = ', '.join(index_fields)
            conn.execute(f'''
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {cls.__table_name__} ({index_cols})
            ''')

    @classmethod
    def from_row(cls, row: sqlite3.Row):
        kwargs = {}
        for field in dataclasses.fields(cls):
            if field.name.startswith('_'):
                continue
            kwargs[field.name] = cls.sql_to_py(
                field.name, field.type, row[field.name]
            )
        return cls(**kwargs)

    def upsert(self, conn: sqlite3.Connection):
        record_dict = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if not field.name.startswith('_')
        }
        record_keys = list(record_dict.keys())
        placeholders = ', '.join('?' for _ in record_keys)
        columns = ', '.join(record_keys)
        pk_cols = ', '.join(self.__primary_key__)
        values = [
            self.py_to_sql(record_dict[key]) for key in record_keys
        ]
        pk_conflict = ', '.join(
            f'{col}=excluded.{col}'
            for col in self.__update_fields__
            if col not in self.__primary_key__
        )
        if pk_conflict:
            conn.execute(f'''
            INSERT INTO {self.__table_name__} ({columns})
            VALUES ({placeholders})
            ON CONFLICT ({pk_cols}) DO UPDATE SET {pk_conflict}
            ''', values)
        else:
            conn.execute(f'''
            INSERT INTO {self.__table_name__} ({columns})
            VALUES ({placeholders})
            ON CONFLICT ({pk_cols}) DO NOTHING
            ''', values)

@dataclasses.dataclass(slots=True)
class RunRecord(BaseRecord):
    __table_name__: ClassVar[str] = 'runs'
    __primary_key__: ClassVar[Tuple[str, ...]] = ('id',)
    __indexes__: ClassVar[Tuple[Tuple[str, ...], ...]] = (('project',),)
    __update_fields__: ClassVar[Tuple[str, ...]] = (
        'end_time', 'updated_at'
    )

    id: str
    project: str
    name: str
    start_time: float
    config: dict
    created_at: float
    end_time: float | None = None
    updated_at: float | None = None

    @classmethod
    def new_name(cls, conn: sqlite3.Connection, project: str) -> str:
        # Query the number of existing runs in the same project to generate a name
        cursor = conn.execute('''
        SELECT COUNT(id) FROM runs WHERE project=?
        ''', (project,))
        count = cursor.fetchone()[0]
        return f'run_{count + 1}'

@dataclasses.dataclass(slots=True)
class HistoryRecord(BaseRecord):
    __table_name__: ClassVar[str] = 'history'
    __primary_key__: ClassVar[Tuple[str, ...]] = ('run_id', 'step', 'key')
    __indexes__: ClassVar[Tuple[Tuple[str, ...], ...]] = (
        ('run_id',),
        ('run_id', 'key', 'step'),
    )
    __foreign_keys__: ClassVar[Tuple[ForeignKey, ...]] = (
        ForeignKey(
            fields=('run_id',),
            ref_table='runs',
            ref_fields=('id',),
            on_delete='CASCADE'
        ),
    )
    __update_fields__: ClassVar[Tuple[str, ...]] = ('value_json', 'value_type', 'timestamp')

    run_id: str
    step: int
    key: str
    timestamp: float
    value_json: str
    value_type: str

    @property
    def value(self) -> Any:
        return json.loads(self.value_json)

    @classmethod
    def from_value(
        cls, run_id: str, step: int, key: str, value: Any
    ) -> 'HistoryRecord':
        return cls(
            run_id=run_id,
            step=step,
            key=key,
            timestamp=time.time(),
            value_json=json.dumps(value),
            value_type=type(value).__name__
        )

class Api():
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, timeout=30.0)
        self.conn.execute('PRAGMA foreign_keys = ON')
        self.conn.execute('PRAGMA journal_mode = WAL')
        self.conn.execute('PRAGMA synchronous = NORMAL')
        self.conn.execute('PRAGMA busy_timeout = 30000')
        self.conn.row_factory = sqlite3.Row

        if not RunRecord.test_table_exists(self.conn) or \
            not HistoryRecord.test_table_exists(self.conn):
            raise ValueError('Database tables not found. Please initialize a Run first.')
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _ensure_open(self):
        if self._closed:
            raise ValueError('API is closed')

    def runs(
        self, record_filter: 'Callable[[RunRecord], bool]' = lambda x: True
    ):
        self._ensure_open()
        cursor = self.conn.execute('SELECT * FROM runs')
        for row in cursor:
            run_record = RunRecord.from_row(row)
            if record_filter(run_record):
                yield run_record

    def run(self, run_id: str):
        self._ensure_open()
        cursor = self.conn.execute('SELECT * FROM runs WHERE id=?', (run_id,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f'Run with id {run_id} not found')
        return RunRecord.from_row(row)

    def run_keys(self, run_id: str):
        self._ensure_open()
        cursor = self.conn.execute(
            '''
        SELECT DISTINCT key
        FROM history
        WHERE run_id=?
        ORDER BY key ASC
        ''',
            (run_id,)
        )
        yield from (row['key'] for row in cursor)

    def history(
        self, run_id: str, keys: List[str] | str | None = None,
        min_step: int | None = None,
        max_step: int | None = None
    ):
        self._ensure_open()
        sql = '''
        SELECT * FROM history
        WHERE {}
        ORDER BY step ASC, key ASC
        '''
        conditions = ['run_id=?']
        args: List[Any] = [run_id]
        match keys:
            case None:
                pass
            case str():
                conditions.append('key=?')
                args.append(keys)
            case []:
                yield from ()
                return
            case list():
                conditions.append(
                    'key IN ({})'.format(','.join('?' for _ in keys))
                )
                args.extend(keys)
        if min_step is not None:
            conditions.append('step>=?')
            args.append(min_step)
        if max_step is not None:
            conditions.append('step<=?')
            args.append(max_step)

        cursor = self.conn.execute(sql.format(' AND '.join(conditions)), args)
        yield from (HistoryRecord.from_row(row) for row in cursor)

    def scan_history(
        self, run_id: str, keys: List[str] | str | None = None,
        min_step: int | None = None,
        max_step: int | None = None
    ):
        yield from map(
            lambda g: {obj.key: obj.value for obj in g[1]} | {'_step': g[0]},
            itertools.groupby(
                self.history(run_id, keys, min_step, max_step),
                key=lambda r: r.step
            )
        )

    def close(self):
        if self._closed:
            return

        self.conn.close()
        self._closed = True


class Run():
    def __init__(
        self, project: str, name: str | None = None,
        config: Dict[str, Any] | None = None, db_path: str | None = None
    ):
        if db_path is None:
            db_path = ':memory:'
        if config is None:
            config = {}

        self.conn = sqlite3.connect(db_path, timeout=30.0)
        self.conn.execute('PRAGMA foreign_keys = ON')
        self.conn.execute('PRAGMA journal_mode = WAL')
        self.conn.execute('PRAGMA synchronous = NORMAL')
        self.conn.execute('PRAGMA busy_timeout = 30000')
        RunRecord.create_table(self.conn)
        HistoryRecord.create_table(self.conn)

        if name is None:
            name = RunRecord.new_name(self.conn, project)

        self.run_id = str(uuid.uuid4())
        self.record = RunRecord(
            id=self.run_id,
            project=project,
            name=name,
            start_time=time.time(),
            config=config,
            created_at=time.time()
        )
        self.record.upsert(self.conn)
        self.conn.commit()
        self._finished = False
        self.pending_steps = []
        self.step = 0

    def log(self, data: Dict[str, Any], step: int | None = None, commit: bool = True):
        if step is None:
            self.step += 1
            step = self.step
        for key, value in data.items():
            record = HistoryRecord.from_value(
                run_id=self.run_id,
                step=step,
                key=key,
                value=value
            )
            self.pending_steps.append(record)
        if commit:
            self.write_pending()

    def write_pending(self):
        if not self.pending_steps:
            return
        with self.conn:
            for record in self.pending_steps:
                record.upsert(self.conn)
            self.record.updated_at = time.time()
            self.record.upsert(self.conn)
        self.pending_steps.clear()

    def finish(self):
        if self._finished:
            return
        self.write_pending()

        self.record.end_time = time.time()
        self.record.updated_at = time.time()
        self.record.upsert(self.conn)
        self.conn.commit()
        self.conn.close()
        self._finished = True


def init(
    project: str, name: str | None = None,
    config: Dict[str, Any] | None = None, db_path: str | None = None
):
    return Run(project, name, config, db_path)
