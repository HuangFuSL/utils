import itertools
import sqlite3
from pathlib import Path
from typing import Dict

_SCHEMA = '''
CREATE TABLE IF NOT EXISTS scripts (
    script_id   TEXT    PRIMARY KEY NOT NULL
);

CREATE TABLE IF NOT EXISTS run_inputs (
    script_id   TEXT    REFERENCES scripts(script_id) ON DELETE CASCADE,
    upstream_id TEXT    REFERENCES scripts(script_id) ON DELETE CASCADE,
    path        TEXT    NOT NULL,
    hash        TEXT    NOT NULL,
    UNIQUE(script_id, upstream_id, path)
);

CREATE TABLE IF NOT EXISTS run_outputs (
    script_id   TEXT    REFERENCES scripts(script_id) ON DELETE CASCADE,
    path        TEXT    NOT NULL,
    hash        TEXT    NOT NULL,
    UNIQUE(script_id, path)
);

CREATE INDEX IF NOT EXISTS idx_scripts    ON scripts (script_id);
CREATE INDEX IF NOT EXISTS idx_inputs     ON run_inputs (script_id);
CREATE INDEX IF NOT EXISTS idx_outputs    ON run_outputs (script_id);
'''

class RunDB():
    def __init__(self, db_path: str | Path, script_id: str):
        self._conn = None
        self.db_path = db_path
        self.script_id = script_id
        with self:
            self.conn.executescript(_SCHEMA)
            self.conn.execute(
                'INSERT OR IGNORE INTO scripts (script_id) VALUES (?)',
                (self.script_id,),
            )
            self.conn.commit()

    def connect(self):
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute('PRAGMA busy_timeout=5000')
        self._conn.execute('PRAGMA foreign_keys=ON')

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        return self._conn # type: ignore

    def commit(self):
        self.conn.commit()

    def set_input(self, inputs: Dict[str, str]):
        self.conn.executemany(
            '''
            INSERT INTO run_inputs (script_id, upstream_id, path, hash)
            SELECT ?, ro.script_id, ?, ?
            FROM run_outputs ro
            WHERE ro.path = ? AND ro.hash = ?

            UNION ALL

            SELECT ?, NULL, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM run_outputs WHERE path = ? AND hash = ?
            )
            ON CONFLICT(script_id, upstream_id, path) DO UPDATE SET hash=excluded.hash
            ''',
            [
                (self.script_id, p, h, p, h, self.script_id, p, h, p, h)
                for p, h in inputs.items()
            ],
        )

    def set_output(self, outputs: Dict[str, str]):
        self.conn.executemany(
            '''
            INSERT INTO run_outputs (script_id, path, hash) VALUES (?, ?, ?)
            ON CONFLICT(script_id, path) DO UPDATE SET hash=excluded.hash
            ''',
            [(self.script_id, p, h) for p, h in outputs.items()],
        )

    def get_input(self, script_id: str | None = None) -> Dict[str, Dict[str, str]]:
        if script_id is None:
            script_id = self.script_id
        ret = sorted(
            [dict(r) for r in self.conn.execute(
                'SELECT * FROM run_inputs WHERE script_id = ?', (script_id,)
            ).fetchall()],
            key=lambda r: r['upstream_id'] if r['upstream_id'] is not None else '',
        )
        ret_grouped = itertools.groupby(ret, key=lambda r: r['upstream_id'])
        return {
            k: {r['path']: r['hash'] for r in g}
            for k, g in ret_grouped
        }

    def get_output(self, script_id: str | None = None) -> Dict[str, str]:
        if script_id is None:
            script_id = self.script_id
        ret = [
            dict(r) for r in self.conn.execute(
                'SELECT * FROM run_outputs WHERE script_id = ?', (script_id,)
            ).fetchall()
        ]
        return {r['path']: r['hash'] for r in ret}
