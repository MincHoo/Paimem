#!/usr/bin/env python3
"""从真实 pi 会话里蒸馏测试语料

思路：用户问过的问题 = 查询（真实、无我的措辞偏差）
      助手当时答出的关键事实 = 记忆（出处可追溯）
      真值 = 这条记忆来自哪个问答对（provenance，客观）

用法:
    python build_eval_from_sessions.py --out /tmp/sess.db --max-sessions 3
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SESS_DIR = os.path.join(os.path.expanduser("~"), ".pi", "agent", "sessions")
SELF_SESSION = "01a1b879"          # 本对话，排除

FACT_RE = re.compile(r"(\d|ms|MB|GB|版本|端口|默认|只能|必须|不能|建议)")
CJK = re.compile(r"[一-鿿]")
BAD_START = ("|", "↑", "↓", "$", "#", "```", "---", ">", "*", "-")
SKIP_RE = re.compile(r"^(好|行|可以|嗯|ok)[，。!！]?$", re.I)


def texts(msg):
    out = []
    c = msg.get("content")
    if isinstance(c, str):
        out.append(c)
    elif isinstance(c, list):
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                out.append(b["text"])
    return out


def parse_session(path):
    """返回 [(user_text, assistant_text)]"""
    pairs, pending = [], None
    try:
        lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    except Exception:
        return pairs
    for ln in lines:
        try:
            o = json.loads(ln)
        except Exception:
            continue
        if o.get("type") != "message":
            continue
        m = o.get("message") or {}
        role = m.get("role")
        if role == "user":
            t = " ".join(texts(m)).strip()
            if t:
                pending = t
        elif role == "assistant" and pending:
            t = " ".join(texts(m)).strip()
            if t:
                pairs.append((pending, t))
                pending = None
    return pairs


def split_sentences(text):
    parts = re.split(r"(?<=[。！？!?\n])", text)
    return [p.strip() for p in parts if p.strip()]


def topic_of(question):
    q = re.sub(r"\s+", " ", question.strip())
    q = re.split(r"[？?。！!\n]", q)[0]
    q = re.sub(r"^(请问|帮我|麻烦|你|能不能|可不可以|我想知道|问一下)", "", q).strip()
    return q[:22]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sessions", type=int, default=3)
    ap.add_argument("--max-queries", type=int, default=40)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SESS_DIR, "**", "*.jsonl"), recursive=True),
                   key=os.path.getsize, reverse=True)
    files = [f for f in files if SELF_SESSION not in os.path.basename(f)]

    memories, queries = [], []
    for path in files[:a.max_sessions]:
        pairs = parse_session(path)
        sid = os.path.basename(path).split("_")[-1].replace(".jsonl", "")[:8]
        for qi, (q, ans) in enumerate(pairs):
            if len(q) < 6 or SKIP_RE.match(q):
                continue
            topic = topic_of(q)
            if len(topic) < 6 or any(ch in topic for ch in "|↑↓$#`"):
                continue
            # 答案开头通常是结论；比逐句抽取稳
            head = re.sub(r"\s+", " ", ans).strip()
            head = head.lstrip("#>-* ")
            if len(head) < 30:
                continue
            if any(ch in head[:20] for ch in "|↑↓$"):
                continue
            if len(CJK.findall(head[:60])) < 12:
                continue
            facts = [head[:170]]
            if not facts:
                continue
            # 每个问答对最多取 2 条，带话题前缀保证自包含
            for f in facts[:1]:
                mid = "sess-%s-%03d-%d" % (sid, qi, facts.index(f))
                memories.append({"id": mid, "content": f,
                                 "evidence": "turn:%s:%d" % (sid, qi), "query": q})
                queries.append({"q": q, "want": mid})

    # 去重（同 id）
    seen, mem2 = set(), []
    for m in memories:
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        mem2.append(m)

    # 均匀抽样查询
    step = max(1, len(queries) // a.max_queries)
    queries = queries[::step][:a.max_queries]

    with open(a.out.replace(".db", "_mem.jsonl"), "w", encoding="utf-8") as f:
        for m in mem2:
            f.write(json.dumps({"id": m["id"], "content": m["content"], "scope": "sesseval",
                                "claim_type": "fact", "origin": "tool",
                                "evidence": m["evidence"], "tags": "会话蒸馏"},
                               ensure_ascii=False) + "\n")
    with open(a.out.replace(".db", "_q.json"), "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=1)

    print(f"会话 {a.max_sessions} 个 → 记忆 {len(mem2)} 条，查询 {len(queries)} 个")
    print(f"写入 {a.out.replace('.db','_mem.jsonl')} 和 _q.json")


if __name__ == "__main__":
    main()
