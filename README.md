# ΠMem

> 给 [pi](https://pi.dev) 的**本地长期记忆**。每轮自动注入相关记忆，到点提醒模型总结，模型也能主动检索。
>
> 纯本地 · 无 GPU · 无 torch · 单文件 SQLite · 约 220 MB

```
你：e5-small 在这台机器上建库速度多少？
（无记忆）没有这个数字，上下文里也没有任何实测数据——随便报一个就是编的。
（有记忆）450–500 chunk/s（RTX 3050 Laptop 4GB，fp16，含写 LanceDB）。
```

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
python mem.py serve --port 7788   # 手动起服务
python mem.py stats
```

架构与设计决策见 `ARCHITECTURE.md`。

## License

MIT
