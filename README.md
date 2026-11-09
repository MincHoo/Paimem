# ΠMem

**一个嵌入模型，干三件事：检索、建图、判冲突。**

> 给 [pi](https://pi.dev) 的本地长期记忆。每轮注入、到点总结、随时检索。
> 纯本地 · 无 GPU · 无 torch · 单文件 SQLite · 约 220 MB

```
你：e5-small 在这台机器上建库速度多少？
（无记忆）没有这个数字，上下文里也没有任何实测数据——随便报一个就是编的。
（有记忆）450–500 chunk/s（RTX 3050 Laptop 4GB，fp16，含写 LanceDB）。
```

---

## 设计初衷

主流 agent 记忆方案是这么长出来的：LLM 抽实体 → 建知识图谱 → 向量检索 → 每轮塞进上下文。
每一步都要一次模型调用。

我们反过来问了一个问题：**如果一台机器只跑一套嵌入模型，能做到什么？**

ΠMem 的答案是——三件事都用同一个 384 维向量完成：

| 用同一个向量做的事 | 怎么做的 |
|---|---|
| **检索** | 点积找相关记忆（和知识库共用 e5-small，不引入第二个模型） |
| **建图** | 相似度 ≥0.88 且共享标签 → `related`；≥0.97 → `near_dup` |
| **判冲突** | ≥0.94 且词面出现否定/数字矛盾 → `contradicts` |

没有实体抽取、没有本体、没有第二个模型、没有 LLM 标注。

**代价**：图比较"糊"，没有实体级精度。
**收益**：写入零 LLM 成本、全离线、220 MB、单文件可拷。

---

## 三个不一样的地方

**1. 注入几乎不花钱**

pi 的 system prompt 支持增量补丁。ΠMem 把 8 条长期记忆放进 `sections`，首轮生成后**逐字节不变**——
pi 的 diff 发现没变化，就再也不发送。实测 8 轮，补丁发送 0 次。

3 条当前相关的记忆走尾部消息，只付新增 token。合计约 **165 token/轮**。

**2. 写入不调模型**

主流方案每捕获一条观察就起一次 LLM。ΠMem 的写入是：嵌入 → 查重 → 落库，全程 2 ms。
需要模型参与时（总结），只在到点提醒一句，piggyback 在已有请求里，不额外开轮。

**3. 记忆会腐烂，也会被唤醒**

刚被检索过的记忆在接下来半小时内轻微提权（+15%，30 分钟半衰），
让专注同一个话题时越用越顺手；离开话题后自然回落。

同时有一套卫生机制防止越存越乱：

| 机制 | 作用 |
|---|---|
| 衰减（按类型半衰期） | fact 35 天、preference 180 天、observation 7 天 |
| 写入去重（0.96） | 相似记忆合并，不重复存 |
| 冲突标记 | 新旧记忆矛盾时建 `contradicts` 边 |
| 证据审计 | 每条带 `cmd:` / `file:` / `tool:` 前缀，可复现 |
| 端点信任分级 | 未登记端点写入一律 `draft`，不进检索 |

---

## 取舍

同类项目多走"LLM 抽实体建图"的路线（Zep/Graphiti、Mem0、Cognee 等），
ΠMem 选了另一条：**图从向量几何里长出来**，不抽实体、不带本体。

代价是图比较糊，没有实体级精度；换来的是零 LLM 写入、全离线、220 MB。

---

## 它解决什么

LLM 每次会话都从零开始。你上周踩过的坑、定下的方案、测出的数字，它全忘了。

ΠMem 把这些存进本地 SQLite，每轮把最相关的几条塞回上下文：

| 通道 | 时机 | 作用 |
|---|---|---|
| **注入** | 每轮 | 8 条长期记忆进 system（缓存零损失）+ 3 条当前相关进尾部 |
| **总结** | 压缩前 / 60K token / 10 轮 | 提醒模型把可复用的经验写进来 |
| **检索** | 模型自行判断 | `mem_search` 工具，查既往决策和教训 |

---

## 安装

```bash
# 1. 装包
pi install "https://github.com/<you>/PiMem"

# 2. 准备 Python 环境（装依赖 + 下模型，约 3–5 分钟）
python "<包目录>/setup.py"

# 3. 重启 pi
```

`setup.py` 会做四件事：检查 Python → 装 `numpy/onnxruntime/tokenizers` → 从镜像下载 e5-small ONNX（130 MB）→ 初始化数据库 + 自检。

### 依赖

| 项 | 体积 |
|---|---|
| numpy + onnxruntime + tokenizers | 90 MB |
| multilingual-e5-small (int8 ONNX) | 130 MB |
| SQLite | 0（标准库） |
| **合计** | **≈ 220 MB** |

**不需要** GPU、CUDA、torch、向量数据库。

---

## 用法

装好后什么都不用做，正常用 pi 即可。

```bash
/mem            # 看状态：轮次 / 缓存命中 / 注入量 / 延迟
/mem off        # 临时关闭注入
/mem on         # 恢复
```

### 手动操作

```bash
python mem.py add "HF 直连不通，要设 HF_ENDPOINT" --scope work --origin tool \
  --evidence cmd:curl --reproduce "curl https://huggingface.co"
python mem.py search "huggingface 下载" --scope work
python mem.py stats
python mem.py selftest
```

---

## 缓存经济学：为什么这样注入

pi 的 `before_agent_start` 提供两条通道，缓存代价完全不同：

| 通道 | 机制 | 代价 |
|---|---|---|
| `systemPromptOptions.sections` | pi **diff sections**，只追加变化的补丁 | 内容不变 → **一次都不再发送** |
| `return { message }` | 持久化消息，追加在尾部 | 前缀照常命中，只付新增 token |

**实测（8 轮）**：stable hash 逐字节不变 → 补丁一次都没发；累计注入 5.9 KB ≈ 165 token/轮。

**别做的事**：
- ❌ `return { systemPrompt }` —— 整段替换，每轮都是 cache miss（成本差 4–8 倍）
- ❌ 从历史里剥掉上轮注入 —— 断点之后全部 miss，比留着更贵

---

## 记忆怎么排序

```
候选 = 语义 top-30 ∪ 图 2 跳扩展
打分 = 0.50 内容 + 0.20 标签 + 0.15 关键词 + 0.15 图
乘数 = 衰减 × 信任 × 短期激活
```

三个乘数各管一件事：

| 乘数 | 作用 |
|---|---|
| 衰减 | 按类型半衰期（fact 35 天 / observation 7 天）；AI 自己写的再乘 0.9 |
| 信任 | verified=2 满分，verified=0 打 0.85 |
| 短期激活 | 刚命中过 +15%，30 分钟半衰，离开话题自然回落 |

**相似度窄带问题**：e5 中文余弦被压在 0.84–0.89，方差只有 0.03，
而标签/关键词能贡献 0.1+ —— 排序会被词面主导。所以候选内做 min-max 归一化，
把方差拉回来（实测 Hit@1 从 85% 提到 92%，13 条记忆时）。

## 常驻服务

不做开机自启、不做后台常驻。服务是 pi 的子进程，跟着 pi 生、跟着 pi 死。

```
pi 启动 → 拉起 mem.py serve（预热 ~900ms）
每轮    → HTTP，8–20 ms
退出    → 关掉服务
```

三重防孤儿：`session_shutdown` 显式 kill、stdin 管道 EOF 自杀、空闲 1 小时自动退出。
服务拉不起来时自动回退到每次起子进程（慢但可用）。

| | 子进程 | 常驻服务 |
|---|---|---|
| 单轮延迟 | ~900 ms | **8–20 ms** |

---

## 配置

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `MEM_SCOPE` | 按 cwd 推导（`~` → `home`） | 记忆分区 |
| `MEM_SCOPE_FALLBACK` | `global` | 本项目无记忆时回退 |
| `MEM_HARVEST` | 开 | `0` 关闭自动总结 |
| `MEM_DAEMON` | 开 | `0` 关闭常驻服务 |
| `MEM_PORT` | 7788 | 服务端口 |
| `MEM_ENDPOINT` | 空 | 写入时声明的端点；留空 = 未登记 = `draft` |
| `MEM_DRY_RUN` | — | `1` 只记录不注入 |
| `MEM_DISABLE` | — | `1` 完全关闭 |
| `MEM_PY` / `MEM_PY_PATH` | 自动探测 | Python 解释器 / mem.py 路径 |

项目级配置 `<项目>/.pi/memskill.json`：

```json
{ "enabled": true, "scope": "my-project", "harvest": true }
```

---

## 总结触发（harvest）

三种来源，一个动作：

| 钩子 | 条件 |
|---|---|
| 压缩前 | `session_before_compact` 后下一轮 |
| 长任务 | 距上次 ≥ 60K token，之后每 +40K |
| 多轮对话 | 距上次 ≥ 10 轮且新增 ≥ 2K token |

触发时在 tail 末尾追加一行（**不额外开一轮**）：

```
[mem:harvest] 第 10 轮 / 约 35K token（多轮对话）。
如有值得记忆的内容则记录 1~5 条，无则不记。
```

**实测**：12 轮任务触发 1 次，写入 3 条（没有过量），第 11–12 轮不再写。

---

## 安全边界

- **端点白名单**：按 `scheme://host:port` 登记，不按模型自报。未登记端点写入一律 `draft`，不进默认检索。
- **`revoked` 永不复活**：SQL 触发器强制。
- **硬删除禁止**：只能软删除。
- **服务只监听 127.0.0.1**：无鉴权，同机进程可访问。

⚠️ 配置即信任：白名单防的是"无意降级"，不防恶意提权。远程端点请显式登记。

---

## 移植到其他 harness

核心逻辑全在 `mem.py`，HTTP 契约是稳定的接缝：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | `{ok, active, pid}` |
| POST | `/retrieve` | `{session, query, scope, exclude_ids}` → `{stable, tail, debug}` |
| POST | `/search` | `{query, scope, k}` → `{results}` |
| POST | `/add` | `{content, scope, origin, claim_type, ...}` → `{id, status}` |
| POST | `/snapshot` | `{session}` → `{text, ids}` |

宿主只需要实现三件事：`stable.text` 进 system 固定位置、`tail.text` 追加到用户消息后、**不要从历史剥离注入**。

---

## 开发

```bash
python mem.py selftest        # 11 项不变量
python eval_mem.py            # 记忆质量：Hit@k / MRR + 三个专项
python mem.py serve --port 7788   # 手动起服务
python mem.py stats
```

架构与设计决策见 `ARCHITECTURE.md`。

## 卸载

```bash
python uninstall.py --purge    # 停服务 + 删 ~/.mem
pi remove "<安装路径>"          # 从 pi 配置里移除
```

## 与 npm 上 pi-mem 的区别

同名生态里已有一个 [`pi-mem`](https://github.com/georgebashi/pi-mem)（作者 georgebashi），思路不同：

| | ΠMem | pi-mem |
|---|---|---|
| 抽取方式 | 嵌入 + 规则，**零 LLM 调用** | 每个 tool_result 起一个 pi 子进程做 LLM 抽取 |
| 嵌入 | **本地 ONNX e5-small**，离线 | 需要 OpenAI 兼容的 `/v1/embeddings` API |
| 存储 | SQLite 单文件 | LanceDB |
| 注入位置 | stable 进 **system sections**（缓存零损失） | 每轮一条 message |
| 记忆卫生 | 衰减 / 去重 / 冲突 / 图扩展 / 证据审计 | 会话摘要 + 观察 |
| 检索粒度 | 单层，返回全文 | 三层：index → timeline → details |
| 项目隔离 | cwd / scope | git remote |

**取舍**：ΠMem 用零成本、离线换取了"不会自动观察每个工具调用"——写入靠 harvest 提示模型主动写。
pi-mem 全自动，但每个工具调用都要一次 LLM，成本随工具使用量线性增长。

## License

MIT
