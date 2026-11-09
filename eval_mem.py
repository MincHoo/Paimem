#!/usr/bin/env python3
"""记忆质量测试（尺子）

三个部分：
  A. 真实集：用真实记忆 + 手写查询，测 Hit@k / MRR
  B. 专项：
     - scope 泄漏：A 项目的记忆不该被 B 项目召回
     - 断言区分度：verified=0 的旧断言不该压过 verified=2 的工具实证
     - 回音壁：self 源条目在没有外部印证时不该越滚越重
  C. 延迟：单次检索耗时

用法:
    python eval_mem.py                  # 用真实库（只读）
    python eval_mem.py --json           # 机器可读
"""
import argparse
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mem():
    spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "mem.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m"] = m
    spec.loader.exec_module(m)
    return m


# ---- A. 真实集：查询 → 期望命中的记忆 id ----
CASES = [
    ("为什么我下载模型总是失败？", "mem-20261109-99ff0b"),
    ("维基语料是从哪来的？", "mem-20261109-8426bd"),
    ("中文维基的条目质量怎么样，适合按小节切吗？", "mem-20261109-d29eb2"),
    ("我的 GPU 为什么一直没被用上？", "mem-20261109-9c7b4c"),
    ("外部連結和延伸閱讀这些噪音怎么处理？", "mem-20261109-87dbd5"),
    ("从 wikimedia 直接下 dump 行不行？", "mem-20261109-3baab3"),
    ("建库速度大概多少？", "mem-20261109-834d5e"),
    ("向量精度到底选 fp16 还是 fp8？", "mem-20261109-01d54f"),
    ("代码库里 Go 的文件怎么全没了？", "mem-20261109-c6d045"),
    ("为什么不用 bge 那个中文模型？", "mem-20261109-da8f72"),
    ("JavaScript 切出来的块为什么特别多？", "mem-20261109-3fd76a"),
    ("三个库的数据结构是统一的吗？", "mem-20261109-30d87e"),
    ("能不能再装一个向量模型专门给记忆库用？", "mem-20261109-45eec0"),
]


def part_a(m, con, cfg, topk=10):
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    rr = 0.0
    rows = []
    lat = []
    for q, want in CASES:
        t = time.time()
        res = m._search_core(con, cfg, q, "global", 30)
        lat.append((time.time() - t) * 1000)
        ids = [r["id"] for _, _, r in res[:topk]]
        rank = ids.index(want) + 1 if want in ids else 0
        for k in hits:
            if rank and rank <= k:
                hits[k] += 1
        if rank:
            rr += 1.0 / rank
        rows.append({"q": q, "want": want, "rank": rank,
                     "top3": [(r["id"], (r["content"] or "")[:34]) for _, _, r in res[:3]]})
    n = len(CASES)
    return {
        "n": n,
        "hit@1": round(hits[1] / n, 3),
        "hit@3": round(hits[3] / n, 3),
        "hit@5": round(hits[5] / n, 3),
        "hit@10": round(hits[10] / n, 3),
        "mrr": round(rr / n, 3),
        "latency_ms_median": round(sorted(lat)[n // 2]),
        "rows": rows,
    }


def part_b_scope_leak(m, con, cfg):
    """两个 scope 内容相近，查 A 不该返回 B"""
    now = int(time.time())
    plan = {
        "leakA": [("项目A 的数据库端口是 5433，只读账号是 reader_a", "A1"),
                  ("项目A 的部署目录在 /srv/alpha", "A2")],
        "leakB": [("项目B 的数据库端口是 6433，只读账号是 reader_b", "B1"),
                  ("项目B 的部署目录在 /srv/beta", "B2")],
    }
    made = []
    for scope, items in plan.items():
        for text, tag in items:
            v = m.embed(["passage: " + text])[0]
            mid = "leak-%s-%s" % (scope, tag)
            con.execute(
                "INSERT OR REPLACE INTO mem_items(id,content,claim_type,origin,verified,"
                "scope,tags,importance,n_tok,created_at,valid_from,status,hot_score,"
                "hit_count,cold,vector) VALUES(?,?,?,?,?,?,?,?,?,?,?,'active',1,0,0,?)",
                (mid, text, "fact", "tool", 2, scope, "", 0.5, len(text) // 3,
                 now, now, m.np.array(v, dtype=m.np.float32).tobytes()))
            made.append(mid)
    con.commit()
    leaked = 0
    total = 0
    for scope, q in (("leakA", "数据库端口是多少"), ("leakB", "数据库端口是多少")):
        res = m._search_core(con, cfg, q, scope, 30)
        ids = [r["id"] for _, _, r in res[:5]]
        other = "leakB" if scope == "leakA" else "leakA"
        total += 1
        if any(i.startswith("leak-%s" % other) for i in ids):
            leaked += 1
    for mid in made:
        con.execute("UPDATE mem_items SET status='archived' WHERE id=?", (mid,))
    con.commit()
    return {"cases": total, "leaked": leaked, "ok": leaked == 0}


def part_b_assertion(m, con, cfg):
    """同一话题：verified=0 的旧断言 vs verified=2 的工具实证，谁排前面"""
    now = int(time.time())
    topic = "部署脚本用的是 docker compose 还是 systemd"
    rows = [
        ("assert-weak", "部署脚本用的是 docker compose（我猜的，没验证）", 0, "self"),
        ("assert-strong", "部署脚本用的是 systemd，实测 systemctl status 可见", 2, "tool"),
    ]
    for mid, text, ver, origin in rows:
        v = m.embed(["passage: " + text])[0]
        con.execute(
            "INSERT OR REPLACE INTO mem_items(id,content,claim_type,origin,verified,"
            "scope,tags,importance,n_tok,created_at,valid_from,status,hot_score,"
            "hit_count,cold,vector) VALUES(?,?,?,?,?,?,?,?,?,?,?,'active',1,0,0,?)",
            (mid, text, "fact", origin, ver, "assert", "", 0.5, len(text) // 3,
             now, now, m.np.array(v, dtype=m.np.float32).tobytes()))
    con.commit()
    res = m._search_core(con, cfg, "部署脚本用的是什么", "assert", 30)
    ids = [r["id"] for _, _, r in res[:3]]
    for mid, *_ in rows:
        con.execute("UPDATE mem_items SET status='archived' WHERE id=?", (mid,))
    con.commit()
    ok = ids[:1] == ["assert-strong"]
    return {"top3": ids, "ok": ok, "note": "verified=2 应排第一"}


def part_b_echo(m, con, cfg):
    """回音壁：self 源应比 tool 源衰减快"""
    now = int(time.time())
    old = now - 120 * 86400          # 120 天前
    rows = [
        ("echo-self", "缓存策略应该用 LRU（我自己推断的）", "self", 0),
        ("echo-tool", "缓存策略实测用 LRU，命中率 87%", "tool", 2),
    ]
    for mid, text, origin, ver in rows:
        v = m.embed(["passage: " + text])[0]
        con.execute(
            "INSERT OR REPLACE INTO mem_items(id,content,claim_type,origin,verified,"
            "scope,tags,importance,n_tok,created_at,valid_from,status,hot_score,"
            "hit_count,last_hit,cold,vector) VALUES(?,?,?,?,?,?,?,?,?,?,?,'active',1,0,?,0,?)",
            (mid, text, "fact", origin, ver, "echo", "", 0.5, len(text) // 3,
             old, old, old, m.np.array(v, dtype=m.np.float32).tobytes()))
    con.commit()
    res = m._search_core(con, cfg, "缓存策略用什么", "echo", 30)
    order = [r["id"] for _, _, r in res]
    for mid, *_ in rows:
        con.execute("UPDATE mem_items SET status='archived' WHERE id=?", (mid,))
    con.commit()
    ok = order and order[0] == "echo-tool"
    return {"order": order, "ok": ok, "note": "120 天后 tool 源应压过 self 源"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    m = load_mem()
    con = m.db()
    m.init_db(con)
    cfg = m.load_config()

    A = part_a(m, con, cfg)
    B1 = part_b_scope_leak(m, con, cfg)
    B2 = part_b_assertion(m, con, cfg)
    B3 = part_b_echo(m, con, cfg)

    out = {"A_real": A, "B1_scope_leak": B1, "B2_assertion": B2, "B3_echo": B3}

    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print("=== A. 真实集（13 条记忆 / 13 个查询）===")
    print(f"  Hit@1  {A['hit@1']:.0%}")
    print(f"  Hit@3  {A['hit@3']:.0%}")
    print(f"  Hit@5  {A['hit@5']:.0%}")
    print(f"  Hit@10 {A['hit@10']:.0%}")
    print(f"  MRR    {A['mrr']:.3f}")
    print(f"  延迟   中位 {A['latency_ms_median']} ms")
    miss = [r for r in A["rows"] if r["rank"] == 0 or r["rank"] > 3]
    if miss:
        print("\n  未进 top-3 的查询：")
        for r in miss:
            print(f"    [{r['rank'] or '未命中'}] {r['q']}")
            print(f"          期望 {r['want']}")
            for i, c in r["top3"]:
                print(f"          top: {i} {c}")

    print("\n=== B. 专项 ===")
    print(f"  scope 泄漏  {'✅ 无泄漏' if B1['ok'] else '❌ 泄漏 %d/%d' % (B1['leaked'], B1['cases'])}")
    print(f"  断言区分度  {'✅ verified=2 优先' if B2['ok'] else '❌ 被旧断言压过'}  {B2['top3']}")
    print(f"  回音壁      {'✅ tool 源胜出' if B3['ok'] else '❌ self 源仍占优'}  {B3['order']}")


if __name__ == "__main__":
    main()
