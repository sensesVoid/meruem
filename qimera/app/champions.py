"""QiMera Champions Registry (QiMera.md §6.9 / M9).

A local SQLite database (stdlib ``sqlite3`` — no ORM dependency) storing every
promoted / historical champion candidate so the GP, interpretation and
synthesis outputs have a durable, queryable home:

Schema (``champions`` table):

    id            INTEGER PRIMARY KEY AUTOINCREMENT
    asset         TEXT    (symbol, e.g. EURUSD)
    regime        TEXT    (regime label the champion was validated under)
    source        TEXT    (gp | interp | synthesis | classifier)
    strategy      TEXT    (indicator family / strategy name)
    params_json   TEXT    (typed parameter vector)
    run_id        TEXT    (validation run id from M4 ValidationLog)
    promotion     TEXT    (promoted | superseded | rejected)
    manifest_version INTEGER
    metrics_json  TEXT    (net_pnl, win_rate, profit_factor, max_dd_pct, trades)
    created_at    TEXT    (ISO UTC)

Utilities:
- ``promote(...)``          register a champion (auto-supersedes the previous
                            promoted champion for the same (asset, regime)).
- ``supersede``/``reject``  lifecycle transitions.
- ``rollback(asset, regime, target_id=None)``  restore a previously superseded
                            champion (M9 rollback requirement).
- ``list_champions``/``current``  query helpers.

The registry is consumed by M10 (manifest builder reads the current promoted
champion per asset+regime) and by M11 (seed populations from prior champions).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "champions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS champions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    regime TEXT NOT NULL DEFAULT 'ANY',
    source TEXT NOT NULL DEFAULT 'gp',
    strategy TEXT NOT NULL DEFAULT '',
    params_json TEXT NOT NULL DEFAULT '[]',
    run_id TEXT NOT NULL DEFAULT '',
    promotion TEXT NOT NULL DEFAULT 'promoted',
    manifest_version INTEGER NOT NULL DEFAULT 1,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_champions_asset_regime
    ON champions (asset, regime);
"""


class ChampionsRegistry:
    """SQLite-backed champions registry (thread-safe per-connection)."""

    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ write

    def promote(self, asset: str, regime: str = "ANY", source: str = "gp",
                strategy: str = "", params: Any = None, run_id: str = "",
                metrics: dict | None = None,
                manifest_version: int = 1) -> dict:
        """Register a champion, superseding the previous one for (asset, regime).

        Returns the new champion row (and ``superseded_id`` of what it replaced).
        """
        asset = asset.upper()
        regime = regime or "ANY"
        superseded_id = self._promoted_id(asset, regime)
        row = {
            "asset": asset, "regime": regime, "source": source,
            "strategy": strategy,
            "params_json": json.dumps(params if params is not None else []),
            "run_id": run_id, "promotion": "promoted",
            "manifest_version": int(manifest_version),
            "metrics_json": json.dumps(metrics or {}, default=str),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        cur = self._conn.execute(
            """INSERT INTO champions (asset, regime, source, strategy,
               params_json, run_id, promotion, manifest_version, metrics_json,
               created_at) VALUES (:asset, :regime, :source, :strategy,
               :params_json, :run_id, :promotion, :manifest_version,
               :metrics_json, :created_at)""", row)
        new_id = int(cur.lastrowid)
        if superseded_id is not None:
            self._set_promotion(superseded_id, "superseded")
        self._conn.commit()
        out = self.get(new_id)
        out["superseded_id"] = superseded_id
        return out

    def supersede(self, champion_id: int) -> dict | None:
        self._set_promotion(champion_id, "superseded")
        return self.get(champion_id)

    def reject(self, champion_id: int, note: str = "") -> dict | None:
        self._set_promotion(champion_id, "rejected", note=note)
        return self.get(champion_id)

    def _set_promotion(self, champion_id: int, promotion: str,
                       note: str = "") -> None:
        self._conn.execute(
            "UPDATE champions SET promotion = ? WHERE id = ?",
            (promotion, champion_id))
        if note:
            self._conn.execute(
                "UPDATE champions SET metrics_json = json_set(metrics_json, "
                "'$.reject_note', ?) WHERE id = ?", (note, champion_id))
        self._conn.commit()

    # ------------------------------------------------------------------ query

    def _promoted_id(self, asset: str, regime: str) -> int | None:
        row = self._conn.execute(
            """SELECT id FROM champions WHERE asset = ? AND regime = ?
               AND promotion = 'promoted' ORDER BY id DESC LIMIT 1""",
            (asset, regime)).fetchone()
        return int(row["id"]) if row else None

    def current(self, asset: str, regime: str = "ANY") -> dict | None:
        """The currently promoted champion for (asset, regime)."""
        rid = self._promoted_id(asset.upper(), regime or "ANY")
        return self.get(rid) if rid is not None else None

    def get(self, champion_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM champions WHERE id = ?", (champion_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_champions(self, asset: str | None = None,
                       regime: str | None = None,
                       promotion: str | None = None) -> list[dict]:
        sql = "SELECT * FROM champions WHERE 1=1"
        args: list[Any] = []
        if asset:
            sql += " AND asset = ?"
            args.append(asset.upper())
        if regime:
            sql += " AND regime = ?"
            args.append(regime)
        if promotion:
            sql += " AND promotion = ?"
            args.append(promotion)
        sql += " ORDER BY id DESC"
        return [self._row_to_dict(r) for r in self._conn.execute(sql, args)]

    # --------------------------------------------------------------- rollback

    def rollback(self, asset: str, regime: str = "ANY",
                 target_id: int | None = None) -> dict | None:
        """Restore a superseded champion for (asset, regime).

        ``target_id`` picks a specific historical champion; otherwise the most
        recently superseded one is restored. The current promoted champion (if
        any) is demoted to superseded.
        """
        asset = asset.upper()
        regime = regime or "ANY"
        current_id = self._promoted_id(asset, regime)
        if target_id is not None:
            target = self.get(target_id)
            if target is None or target["asset"] != asset or target["regime"] != regime:
                raise ValueError(
                    f"champion {target_id} does not belong to {asset}/{regime}")
        else:
            row = self._conn.execute(
                """SELECT id FROM champions WHERE asset = ? AND regime = ?
                   AND promotion = 'superseded' ORDER BY id DESC LIMIT 1""",
                (asset, regime)).fetchone()
            target_id = int(row["id"]) if row else None
        if target_id is None:
            return None
        if current_id is not None:
            self._set_promotion(current_id, "superseded")
        self._set_promotion(target_id, "promoted")
        return self.get(target_id)

    # ------------------------------------------------------------------ misc

    def history(self, asset: str, regime: str = "ANY") -> list[dict]:
        """Full promotion history for (asset, regime), newest first."""
        return self.list_champions(asset=asset, regime=regime)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = {k: row[k] for k in row.keys()}
        for key in ("params_json", "metrics_json"):
            try:
                d[key] = json.loads(d[key])
            except (ValueError, TypeError):
                d[key] = {}
        return d
