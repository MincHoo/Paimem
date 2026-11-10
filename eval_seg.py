#!/usr/bin/env python3
"""段索引质量测试（尺子 · 工作记忆）

为什么要有它：记忆条目有 eval_mem.py，段（工作记忆）一直没有尺子。
没有尺子，切段阈值、摘要长度、向量窗口、域激活权重全都只能靠感觉调。

做法：
  从 mem_segments 找到有原文（transcript 指向会话 JSONL）的段，
  按段的 turn_start..turn_end 从原文里取**用户当时的原话**作为 query，
  答案为该段 id —— 查询与段是独立产生的，不是拿摘要反推自己。

用法:
    python eval_seg.py                 # 现算（段会随库变化）
    python eval_seg.py --snapshot      # 冻结成 segbench.json 后再跑
    python eval_seg.py --frozen        # 只用冻结集跑（推荐，可跨版本比较）
    python eval_seg.py --json          # 机器可读
"""
import argparse
import importlib.util
import json
import os
import sqlite3
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "segbench.json")

# 纯确认/寒暄不构成检索需求，拿它们出题只会惩罚系统
TRIVIAL = ("可以", "好的", "好", "行", "继续", "赞同", "对的", "是的", "嗯", "ok",
           "go", "去吧", "开始吧", "做吧", "嗯嗯", "okay", "1", "2", "3")


def load_mem():
    spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "mem.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m"] = m
    spec.loader.exec_module(m)
    return m


def user_msgs(path):
    """按 pi 的约定取用户消息序列（第 n 轮 = 第 n 条用户消息）"""
    out = []
    if not path or not os.path.exists(path):
        return out
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                msg = e.get("message") or {}
                if msg.get("role") != "user":
                    continue
                c = msg.get("content")
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                c = (c or "").strip()
                if not c:
                    continue
                # 注入块不是用户的提问
                if c.startswith("[mem:") or "⚠️ 以下为记忆数据" in c:
                    continue
                out.append(c)
    except Exception:
        return out
    return out


def build_cases(con, limit_per_seg=2):
    """(query, seg_id, scope, session_id, turn) —— query 必须是用户当时的原话"""
    cases = []
    rows = con.execute(
        "SELECT id,scope,session_id,transcript,turn_start,turn_end FROM mem_segments"
        " WHERE transcript IS NOT NULL AND transcript<>'' ORDER BY created_at DESC").fetchall()
    seen = set()
    for r in rows:
        msgs = user_msgs(r["transcript"])
        if not msgs:
            continue
        ts, te = (r["turn_start"] or 0), (r["turn_end"] or 0)
        idxs = []
        if 1 <= ts <= len(msgs):
            idxs.append(ts - 1)
        if te > ts and 1 <= te <= len(msgs):
            idxs.append(te - 1)
        picked = 0
        for i in idxs:
            q = msgs[i].strip()
            if len(q) < 10 or q.strip().lower() in TRIVIAL or q in seen:
                continue
            if q.startswith("<") or "tool_result" in q:
                continue
            seen.add(q)
            cases.append({"q": q[:400], "seg": r["id"], "scope": r["scope"],
                          "session": r["session_id"], "turn": i + 1})
            picked += 1
            if picked >= limit_per_seg:
                break
    return cases


def rank_segments(m, con, query, scope, k=10):
    """复用域激活的打分（语义 + 关键词，权重读配置，min-max 归一）"""
    cfg = m.load_config()
    return m._activate_domains(con, query, scope, k=k, min_score=0.0,
                               w_sem=m.fcfg(cfg, "segment.w_sem"),
                               w_kw=m.fcfg(cfg, "segment.w_kw"))


def run(cases, con, m, topk=10):
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    rrs, lat, misses, near = [], [], [], []
    # 同会话相邻段（相邻 seq）算"近邻命中"：语义上通常也该看
    by_id = {r["id"]: r for r in con.execute(
        "SELECT id,session_id,seq FROM mem_segments")}
    for c in cases:
        t0 = time.time()
        ranked = rank_segments(m, con, c["q"], c["scope"], k=topk)
        lat.append((time.time() - t0) * 1000)
        ids = [x["id"] for x in ranked]
        if c["seg"] in ids:
            pos = ids.index(c["seg"]) + 1
            rrs.append(1.0 / pos)
            for k in hits:
                if pos <= k:
                    hits[k] += 1
        else:
            rrs.append(0.0)
            misses.append({"q": c["q"][:60], "want": c["seg"],
                           "got": [[x["id"], round(x["score"], 3)] for x in ranked[:3]]})
            tgt = by_id.get(c["seg"])
            if tgt:
                for x in ranked[:5]:
                    rr = by_id.get(x["id"])
                    if rr and rr["session_id"] == tgt["session_id"] and \
                            abs((rr["seq"] or 0) - (tgt["seq"] or 0)) <= 1:
                        near.append({"q": c["q"][:50], "want": c["seg"], "near": x["id"]})
                        break
    n = max(1, len(cases))
    return {
        "cases": len(cases),
        "Hit@1": round(100.0 * hits[1] / n, 1),
        "Hit@3": round(100.0 * hits[3] / n, 1),
        "Hit@5": round(100.0 * hits[5] / n, 1),
        "Hit@10": round(100.0 * hits[10] / n, 1),
        "MRR": round(sum(rrs) / n, 3),
        "median_ms": int(statistics.median(lat)) if lat else 0,
        "near_miss_adjacent": len(near),
        "misses": misses[:8],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true", help="重建冻结集 segbench.json")
    ap.add_argument("--frozen", action="store_true", help="只用冻结集")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    m = load_mem()
    con = m.db()

    if a.frozen or (os.path.exists(SNAP) and not a.snapshot and not a.json):
        if not os.path.exists(SNAP):
            print("冻结集不存在，先跑 --snapshot", file=sys.stderr)
            return 1
        with open(SNAP, encoding="utf-8") as f:
            cases = json.load(f)["cases"]
    else:
        cases = build_cases(con)

    if a.snapshot:
        with open(SNAP, "w", encoding="utf-8") as f:
            json.dump({"v": 1, "built_at": int(time.time()), "cases": cases},
                      f, ensure_ascii=False, indent=1)
        print("冻结 %d 条用例 → %s" % (len(cases), SNAP))

    res = run(cases, con, m)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    print("=== 段索引检索（%d 条真实用例，query 为用户当时的原话）===" % res["cases"])
    print("  Hit@1  %5.1f%%     Hit@3  %5.1f%%" % (res["Hit@1"], res["Hit@3"]))
    print("  Hit@5  %5.1f%%     Hit@10 %5.1f%%" % (res["Hit@5"], res["Hit@10"]))
    print("  MRR    %.3f        延迟 中位 %d ms" % (res["MRR"], res["median_ms"]))
    if res["near_miss_adjacent"]:
        print("  近邻命中（同会话相邻段）: %d 条" % res["near_miss_adjacent"])
    if res["misses"]:
        print("\n  未命中样例：")
        for x in res["misses"][:5]:
            print("    Q %s" % x["q"].replace("\n", " "))
            print("      want %s | got %s" % (x["want"], x["got"][:2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
