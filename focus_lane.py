"""
注意力（focus）通道 —— 独立库、独立表，当前只有骨架，暂不接入 hook。

状态：TODO（默认关闭，未接入）
定位：只存"当前目标 + 进度"，不存事实/决策/偏好。
      与持久记忆（mem.py 的 mem_items）、工作记忆（mem_segments）严格分开：
        - 持久记忆：跨会话、门槛高、可推翻
        - 工作记忆：本会话、无门槛、随段老化
        - 注意力（本文件）：当前在追什么目标、进展到哪、是否已完成

为什么独立库：用户明确要求"四库共用入口，不强求同表"。
放在 MEM_HOME/focus.db，不参与 mem.db 的事务与迁移。

启用方式（未实现）：
    MEM_FOCUS=1 后由 hook.py 串行调用 get_active(scope)，结果进 tail。
"""

from __future__ import annotations

import os
import sqlite3
import time

HOME = os.path.expanduser("~")
BASE = os.environ.get("MEM_HOME", os.path.join(HOME, ".mem"))
DB_PATH = os.environ.get("MEM_FOCUS_DB", os.path.join(BASE, "focus.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS goals(
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  scope      TEXT,                    -- 域，与记忆 scope 同名概念
  goal       TEXT NOT NULL,           -- 一句话目标（自足、无代词）
  status     TEXT DEFAULT 'active',   -- active | done | dropped
  updated_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_goals_scope ON goals(scope, status, updated_at DESC);
"""


def db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    con.commit()
    return con


def get_active(con, scope=None, limit=3):
    """返回当前活跃目标（串行调用，无并发）。"""
    q = "SELECT id,scope,goal,status,updated_at FROM goals WHERE status='active'"
    params = []
    if scope:
        q += " AND scope=?"
        params.append(scope)
    q += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in con.execute(q, params)]


def set_goal(con, scope, goal, status="active"):
    """新增或刷新一条目标（同一 scope 下 goal 文本相同则更新 updated_at）。"""
    now = int(time.time())
    row = con.execute("SELECT id FROM goals WHERE scope=? AND goal=?",
                      (scope, goal)).fetchone()
    if row:
        con.execute("UPDATE goals SET status=?, updated_at=? WHERE id=?",
                    (status, now, row["id"]))
        gid = row["id"]
    else:
        gid = con.execute("INSERT INTO goals(scope,goal,status,updated_at) VALUES(?,?,?,?)",
                          (scope, goal, status, now)).lastrowid
    con.commit()
    return gid


def close_goal(con, gid, status="done"):
    con.execute("UPDATE goals SET status=?, updated_at=? WHERE id=?",
                (status, int(time.time()), gid))
    con.commit()
