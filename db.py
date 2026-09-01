"""
SQLite persistence helper for contract master, orders and positions.
File: trader.db in the working directory.
"""
import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)
DB_FILE = "trader.db"

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS contract_master (
  symbol TEXT NOT NULL,
  token TEXT,
  exch_seg TEXT,
  lotsize INTEGER,
  raw_json TEXT,
  fetched_at TIMESTAMP,
  PRIMARY KEY(symbol, exch_seg)
);

CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  broker_order_id TEXT,
  entry_order_id TEXT,
  symbol TEXT,
  exch_seg TEXT,
  qty INTEGER,
  price REAL,
  order_type TEXT,
  status TEXT,
  raw_json TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  entry_order_id TEXT UNIQUE,
  symbol TEXT,
  exch_seg TEXT,
  qty INTEGER,
  avg_entry_price REAL,
  sl_price REAL,
  target_price REAL,
  broker_entry_id TEXT,
  token TEXT,
  status TEXT,
  raw_json TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fills (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  order_id INTEGER,
  broker_order_id TEXT,
  symbol TEXT,
  qty INTEGER,
  price REAL,
  side TEXT,
  filled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _get_conn():
    conn = sqlite3.connect(DB_FILE, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def save_contract_master(data: List[Dict[str, Any]]):
    if not data:
        return

    conn = _get_conn()
    try:
        cur = conn.cursor()
        now = datetime.utcnow().isoformat()
        for row in data:
            symbol = str(row.get('symbol') or row.get('tradingsymbol') or '').upper()
            token = str(row.get('token') or '')
            exch = str(row.get('exch_seg') or row.get('exchSeg') or '')
            try:
                lotsize = int(float(row.get('lotsize', 1) or 1))
            except Exception:
                lotsize = 1
            cur.execute(
                """INSERT OR REPLACE INTO contract_master
                   (symbol, token, exch_seg, lotsize, raw_json, fetched_at)
                   VALUES(?,?,?,?,?,?)""",
                (symbol, token, exch, lotsize, json.dumps(row), now),
            )
        conn.commit()
    except Exception:
        logger.exception("save_contract_master failed")
    finally:
        conn.close()


def load_contract_master_rows(limit: int = 1000) -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT symbol, token, exch_seg, lotsize, raw_json FROM contract_master LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def save_order(
    broker_order_id: str,
    entry_order_id: str,
    symbol: str,
    exch: str,
    qty: int,
    price: float,
    order_type: str,
    status: str,
    raw: Dict[str, Any] = None,
):
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO orders
               (broker_order_id, entry_order_id, symbol, exch_seg, qty, price,
                order_type, status, raw_json, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                broker_order_id,
                entry_order_id,
                symbol,
                exch,
                qty,
                price,
                order_type,
                status,
                json.dumps(raw or {}),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    except Exception:
        logger.exception("save_order failed")
    finally:
        conn.close()


def update_order_status(broker_order_id: str, status: str, updates: Dict[str, Any] = None):
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE orders SET status=?, updated_at=?, raw_json=? WHERE broker_order_id=?",
            (
                status,
                datetime.utcnow().isoformat(),
                json.dumps(updates or {}),
                broker_order_id,
            ),
        )
        conn.commit()
    except Exception:
        logger.exception("update_order_status failed")
    finally:
        conn.close()


def save_position(
    entry_order_id: str,
    symbol: str,
    exch: str,
    qty: int,
    avg_entry_price: float,
    sl_price: float,
    target_price: float,
    broker_entry_id: str,
    token: str,
    status: str = 'OPEN',
    raw: Dict[str, Any] = None,
):
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO positions
               (entry_order_id, symbol, exch_seg, qty, avg_entry_price, sl_price,
                target_price, broker_entry_id, token, status, raw_json, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                entry_order_id,
                symbol,
                exch,
                qty,
                avg_entry_price,
                sl_price,
                target_price,
                broker_entry_id,
                token,
                status,
                json.dumps(raw or {}),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    except Exception:
        logger.exception("save_position failed")
    finally:
        conn.close()


def update_position_status(entry_order_id: str, status: str, updates: Dict[str, Any] = None):
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE positions SET status=?, updated_at=?, raw_json=? WHERE entry_order_id=?",
            (
                status,
                datetime.utcnow().isoformat(),
                json.dumps(updates or {}),
                entry_order_id,
            ),
        )
        conn.commit()
    except Exception:
        logger.exception("update_position_status failed")
    finally:
        conn.close()


def load_open_positions() -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status = 'OPEN'"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def load_orders(limit: int = 1000) -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM orders ORDER BY COALESCE(updated_at, created_at) DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def upsert_broker_order(broker_order_id: str, symbol: str, exch: str, qty: int, price: float,
                        order_type: str, status: str, raw: Dict[str, Any] = None,
                        entry_order_id: str = None):
    if not broker_order_id:
        return
    conn = _get_conn()
    try:
        existing = conn.execute(
            "SELECT id FROM orders WHERE broker_order_id=? ORDER BY id DESC LIMIT 1",
            (str(broker_order_id),),
        ).fetchone()
        payload = (
            str(entry_order_id or broker_order_id), str(symbol or ''), str(exch or ''),
            int(qty or 0), float(price or 0), str(order_type or ''), str(status or ''),
            json.dumps(raw or {}), datetime.utcnow().isoformat(),
        )
        if existing:
            conn.execute(
                """UPDATE orders SET entry_order_id=?, symbol=?, exch_seg=?, qty=?, price=?,
                   order_type=?, status=?, raw_json=?, updated_at=? WHERE id=?""",
                payload + (existing['id'],),
            )
        else:
            conn.execute(
                """INSERT INTO orders (broker_order_id, entry_order_id, symbol, exch_seg, qty, price,
                   order_type, status, raw_json, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (str(broker_order_id),) + payload,
            )
        conn.commit()
    except Exception:
        logger.exception("upsert_broker_order failed")
    finally:
        conn.close()


def get_daily_pnl(date_str: Optional[str] = None) -> Dict[str, float]:
    # Realized P/L is supplied by the broker position/trade reconciliation layer.
    # Keep the DB API deterministic rather than returning a fabricated value.
    return {'realized': 0.0, 'unrealized': 0.0}
