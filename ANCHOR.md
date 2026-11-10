# 锚点接口（宿主无关）

> 任何 agent 只需实现下面两个调用，就能接入这套记忆系统。
> 不依赖 pi，不依赖 HTTP 服务，不要求数据库表结构与其它模块一致。

---

## 1. 每轮：`hook`（读）

```bash
python mem.py hook --session <会话id> --query "<本轮用户输入>" --scope <工作区> \
                   --turn <第几轮> --entry-id <当前条目id> [--transcript <会话JSONL>] [--pretty]
```

返回稳定的 v1 信封：

```json
{
  "v": 1,
  "hook": "anchor",
  "system": "永不变化的记忆块（可整段塞进系统提示；字节不变才能吃到缓存）",
  "tail":   "每轮变化的块（追加到本轮消息末尾）",
  "stable_hash": "76dfeaec",
  "ids": { "stable": ["mem-…"], "tail": [] },
  "domains": [{ "id": "seg-…", "title": "…", "score": 1.0, "turns": [5, 7] }],
  "segment": { "action": "open|continue|close", "closed": { "id": "seg-…", "…": "…" } },
  "insert": { "system": "append", "afterUserText": "\n\n" },
  "lanes": { "persistent": "ok|empty", "working": "ok|empty", "kb": "off", "focus": "off" },
  "debug": { "candidates": 36, "best": 0.62, "elapsedMs": 30 }
}
```

**宿主必须遵守的两条不变量**

1. `system` 逐字节不变时，**不要重写系统提示**（否则前缀缓存全废）
2. `tail` 只追加到本轮消息，不要拼进系统提示

**`domains` 只在第一轮有值**（`--turn 1`）：上一段会话的相关段，可拼成一行提示。

---

## 2. 每轮：`segment commit`（写）

```bash
python mem.py segment commit --session <会话id> --scope <工作区> \
    --turn <第几轮> --last-entry-id <最后一条条目id> [--transcript <会话JSONL>] \
    --payload '<JSON>'
```

payload：

```json
{
  "title": "≤20 字",
  "digest": "≤120 字",
  "statements": [{ "text": "自足陈述", "kind": "decision|fact|rejected|…", "keywords": ["…"] }],
  "supersedes": [{ "old_id": "mem-…", "text": "完整新陈述" }]
}
```

自检不通过时**不写库**，返回：

```json
{ "ok": false, "errors": ["#1 含指代词（这个），必须替换成具体名词"] }
```

成功后返回 `segment` / `promoted` / `conflicts` / `superseded`。
`conflicts` 是"可能推翻的旧记忆"候选，宿主应把它交给模型判断，再用 `supersede` 处理：

```bash
python mem.py supersede <旧id> --new-text "<完整新陈述>"
```

---

## 3. 其它可选用法

| 用途 | 命令 |
|---|---|
| 主动检索 | `python mem.py search "<查询>" --scope <工作区> -k 5` |
| 取全文 | `python mem.py get --ids id1,id2` |
| 写入单条 | `python mem.py add "<陈述>" --scope x --claim-type decision --origin user --evidence <ref>` |
| 查看/调参 | `python mem.py config keys`、`config set inject.dynamic_k=4`、`config describe` |

HTTP 方式（`python mem.py serve --port 7788`，仅 127.0.0.1）：
`POST /retrieve`（同 hook 参数）、`POST /search`、`POST /add`、`POST /get`、`POST /snapshot`。

---

## 4. 可移植性

- 代码里**没有绝对路径**：数据目录默认 `~/.mem`，可用环境变量覆盖
  `MEM_HOME` / `MEM_DB` / `MEM_CONFIG` / `MEM_MODEL`
- 模型与数据库默认都在 `MEM_HOME` 下，整个目录可直接拷走
- 宿主适配层（如 `memskill.ts`）通过 `join(<扩展目录>, "mem.py")` 定位脚本，不写死盘符

---

## 5. lanes 与未来扩展（串行）

`hook` 内部按顺序跑三个任务，**不并行**（避免并发 debug）：

| lane | 状态 | 数据源 |
|---|---|---|
| `persistent` | 已实现 | `mem.db` / `mem_items` |
| `working` | 已实现 | `mem.db` / `mem_segments` + `mem_segment_state` |
| `kb` | 默认 off（`MEM_KB=1` 接） | 知识库独立仓（LanceDB） |
| `focus` | 默认 off（`MEM_FOCUS=1` 接） | `focus.db` / `goals`（注意力，独立库） |

`lanes` 字段逐路报状态；任一路失败只降级自己，不影响其它两路。

## 接入方式（两种，宿主自己选）

| 方式 | 怎么用 | 适合 |
|---|---|---|
| **每轮起进程**（默认，永远可用） | `python mem.py hook --session <id> --query "<文本>" --scope <域> --turn <n>`，写完等下一个人读 stdout | 命令行 pi、任何宿主、冷启动无所谓、要零后台 |
| **本地 daemon**（可选加速） | `python mem.py serve` 常驻 `127.0.0.1:7788`，嵌入模型常驻内存，每轮走 HTTP | 高频交互、机器常开 |

两条路返回同一个 v1 信封，**实现可以随时切换而不改调用方**。
关机/端口被占/服务挂了都自动退回起进程 —— 设 `MEM_DAEMON=0` 可强制只走进程。
实测：起进程一次约 1.4s（含模型加载），daemon 一次约 50ms；但这只影响每轮开头的等待，
不影响模型自身的思考时间。写路径（`segment commit` / `supersede`）一律起进程，
保证在任何环境都能用，不依赖 daemon 是否在跑。
