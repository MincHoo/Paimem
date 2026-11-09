"""合成 2000 条记忆，用于规模验证（延迟 / 图扩展 / 召回）

用法:
    python _synth_2000.py --db <path> --n 2000
"""
import argparse
import json
import os
import random
import sqlite3
import sys

random.seed(20261109)

DOMAINS = {
    "python": ["GIL", "asyncio", "dataclass", "typing", "decorator", "generator",
               "contextvar", "walrus", "f-string", "match-case"],
    "javascript": ["event-loop", "promise", "closure", "prototype", "async-await",
                   "Proxy", "WeakMap", "microtask", "module", "hoisting"],
    "database": ["WAL", "B-tree", "MVCC", "vacuum", "index-only-scan", "isolation",
                 "deadlock", "sharding", "replica", "checkpoint"],
    "network": ["TCP", "QUIC", "TLS", "DNS", "NAT", "MTU", "congestion", "keepalive",
                "HTTP2", "websocket"],
    "ml": ["attention", "embedding", "quantization", "distillation", "LoRA",
           "tokenizer", "batch-norm", "dropout", "beam-search", "KV-cache"],
    "os": ["page-cache", "cgroup", "namespace", "epoll", "mmap", "fork",
           "signal", "syscall", "inode", "swap"],
    "storage": ["parquet", "columnar", "compression", "bloom-filter", "LSM",
                "segment", "compaction", "fsync", "sparse-index", "dictionary"],
    "tooling": ["git-rebase", "make", "docker", "CI", "lockfile", "semver",
                "monorepo", "linter", "formatter", "pre-commit"],
}

ATTRS = ["实测耗时", "默认值", "常见坑", "内存占用", "并发行为", "失败模式",
         "调试方法", "版本差异", "最佳实践", "限制条件"]

VALS = [
    "在 4 核机器上约 {a}-{b} ms",
    "默认关闭，需要显式开启",
    "超过 {a} 条后会线性退化",
    "约占 {a}-{b} MB 常驻内存",
    "并发超过 {a} 时开始争锁",
    "失败时返回空而不报错",
    "用 --verbose 可以看到内部状态",
    "v{a} 之后行为有变化",
    "建议配合显式超时使用",
    "只在小数据量下可靠",
]

TAGS = ["性能", "踩坑", "配置", "并发", "存储", "调试", "版本", "最佳实践"]


def gen(n):
    items = []
    domains = list(DOMAINS)
    per = n // len(domains)
    for d in domains:
        base = DOMAINS[d]
        techs = list(base) + [f"{t}-v2" for t in base] + [f"{t}-v3" for t in base[:5]]
        for i in range(per):
            tech = techs[i % len(techs)]
            attr = ATTRS[i % len(ATTRS)]
            val = VALS[i % len(VALS)].format(a=random.randint(1, 200), b=random.randint(200, 2000))
            content = f"{d}/{tech} 的{attr}：{val}。"
            tags = ",".join(random.sample(TAGS, k=2))
            ct = "fact" if i % 10 else random.choice(["decision", "preference", "procedure"])
            items.append((content, d, tags, ct))
    return items[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--scope", default="synth")
    a = ap.parse_args()

    if os.path.exists(a.db):
        os.remove(a.db)
    cfg_path = a.db + ".toml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(
            "[db]" + chr(10) +
            f'path = "{a.db}"' + chr(10) +
            "[model]" + chr(10) +
            'path = "C:/Users/Cros/.mem/models/e5-small-onnx"' + chr(10)
        )
    os.environ["MEM_CONFIG"] = cfg_path
    sys.argv = ["mem.py", "init"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mem.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["m"] = m
    spec.loader.exec_module(m)

    os.environ["MEM_DB"] = a.db
    con = m.db()
    m.init_db(con)

    items = gen(a.n)
    print(f"生成 {len(items)} 条，开始嵌入（约 {len(items) * 0.002:.0f}s）...", flush=True)

    batch = 64
    t0 = __import__("time").time()
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        vecs = m.embed(["passage: " + c for c, _, _, _ in chunk])
        now = int(__import__("time").time())
        for (content, dom, tags, ct), v in zip(chunk, vecs):
            mid = "syn-%06d" % (i + chunk.index((content, dom, tags, ct)))
            con.execute(
                "INSERT OR REPLACE INTO mem_items(id,content,claim_type,origin,verified,"
                "scope,tags,importance,n_tok,created_at,valid_from,status,hot_score,"
                "hit_count,cold,vector) VALUES(?,?,?,?,?,?,?,?,?,?,?,'active',1.0,0,0,?)",
                (mid, content, ct, "tool", 2, a.scope, tags, 0.5,
                 max(1, len(content) // 3), now, now, m.np.array(v, dtype=m.np.float32).tobytes()))
        con.commit()
        if (i // batch) % 5 == 0:
            print(f"  {i + len(chunk)}/{len(items)}  {__import__('time').time() - t0:.1f}s", flush=True)

    n = con.execute("SELECT COUNT(*) FROM mem_items").fetchone()[0]
    print(f"完成：{n} 条，用时 {__import__('time').time() - t0:.1f}s，库 {os.path.getsize(a.db) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
