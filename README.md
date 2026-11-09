# ΠMem

**给 pi 的本地长期记忆。每轮注入、到点总结、随时检索。**

> 目前只在 **Windows + pi** 上验证过。纯本地运行，不需要 GPU。

装好之后，pi 会记住你之前踩过的坑、定下的方案、测出的数字，并在合适的时机自动带到上下文里。

---

## 实现方式

全部逻辑在一个 Python 文件里，宿主侧只有一层很薄的适配。

**存储**：SQLite 单文件，向量存 BLOB。没有向量数据库、没有服务依赖。

**嵌入**：本地 ONNX 跑的 e5-small（384 维），和知识库共用同一个模型，不额外引入。

**图**：不抽实体、不做本体。边直接由向量几何产生——
相似度 ≥0.88 且共享标签建 `related`，≥0.97 建 `near_dup`，
≥0.94 且词面出现否定或数字矛盾建 `contradicts`。
检索时从种子节点做 2 跳扩展，路径强度按 0.85 逐跳衰减。

**排序**：内容 0.50 + 标签 0.20 + 关键词 0.15 + 图 0.15，
再乘上衰减（按类型半衰期）、信任（按证据等级）、短期激活（刚命中过的 30 分钟内轻微提权）。

e5 的中文余弦分布很窄（0.84–0.89），直接加权会被标签和关键词的方差主导，
所以候选内先做一次 min-max 归一化再打分。

---

## 注入成本

pi 的 system prompt 支持增量补丁。ΠMem 把 8 条长期记忆放进 `sections`，
首轮生成后内容逐字节不变，pi 的 diff 检测到没有变化就不再发送补丁。

3 条当前相关的记忆走尾部消息，只付新增 token。

实测：约 165 token/轮，stable 块 8 轮逐字节不变。

**两条反例**（踩过才知道）：
- 用 `return { systemPrompt }` 整段替换 → 每轮都是 cache miss，成本差 4–8 倍
- 从历史里剥掉上一轮的注入 → 前缀在断点处失配，之后全部 miss，比留着更贵

写入侧同样不调模型：嵌入 → 查重 → 落库，单条约 2 ms。
需要模型参与时（总结），只在到点提醒一句，附加在已有请求里，不额外开轮。

---

## 记忆的维护

| 机制 | 作用 |
|---|---|
| 衰减 | 按类型半衰期（fact 35 天 / preference 180 天 / observation 7 天） |
| 写入去重 | 相似度 ≥0.96 时合并，不重复存 |
| 冲突标记 | 新旧记忆矛盾时建边并标记 disputed |
| 证据审计 | 每条带 `cmd:` / `file:` / `tool:` 前缀，可复现 |
| 端点信任分级 | 未登记端点写入为 draft，不进检索 |
| 短期激活 | 刚命中过的记忆 30 分钟内 +15%，之后自然回落 |

---

## 它解决什么

LLM 每次会话都从零开始。上周踩过的坑、定下的方案、测出的数字，它全忘了。

ΠMem 把这些存进本地 SQLite，每轮把最相关的几条带回上下文：

| 通道 | 时机 | 作用 |
|---|---|---|
| 注入 | 每轮 | 8 条长期记忆进 system（缓存友好）+ 3 条当前相关进尾部 |
| 总结 | 压缩前 / 上下文增长 / 多轮之后 | 提醒模型把可复用的经验写进来 |
| 检索 | 模型自行判断 | `mem_search` 工具，查既往决策和教训 |

---

## 安装

```bash
# 1. 装包
pi install "https://github.com/MincHoo/Paimem"

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

## 端点白名单

记忆的写入权限按**实际连接的端点**判定，不按模型自报的名字。

```toml
# ~/.mem/mem.toml
[[trust.endpoints]]
endpoint = "http://127.0.0.1:11434"   # 只比前缀
model_class = "large"                  # unknown / small / medium / large
```

| 情况 | `origin=self` 写入的结果 |
|---|---|
| 未登记 | `draft`（不进默认检索） |
| 已登记 `unknown` / `small` | `draft` |
| 已登记 `medium` | `active`，`verified=0` |
| 已登记 `large` | `active`，`verified≤1` |
| 任何端点 + `origin=tool` + 证据 | `active`，`verified=2` |

`origin=self` 指模型自己写的；`origin=tool` 指工具实测；`origin=user` 指用户明说。

**为什么按端点而不是按模型名**：模型自报不可信，同一端口后面换了模型系统感知不到。
登记端点等于让换模型必须显式改配置。

也可以直接用命令维护（写在 `~/.mem/trust.toml`，不动你手写的 `mem.toml`）：

```bash
python mem.py trust list
python mem.py trust add "http://127.0.0.1:8080" --class large --note "本地 llama.cpp"
python mem.py trust remove "http://127.0.0.1:8080"
```

⚠️ 配置即信任：白名单防的是「无意降级」，不防恶意提权。远程端点请显式登记。

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

全部可调参数（相似度阈值、四分量权重、衰减、注入规模、配额）都在
`~/.mem/mem.toml`，完整说明用：

```bash
python mem.py config describe
```

⚠️ **相似度阈值与嵌入模型绑定**，换模型必须重测，不能照抄。

项目级配置 `<项目>/.pi/memskill.json`：

```json
{ "enabled": true, "scope": "my-project", "harvest": true }
```

---

## 总结触发（harvest）

总结分两条路：**线程内只留一个钩子，其余全部挪到会话结束后异步跑。**

| 触发 | 位置 | 为什么 |
|---|---|---|
| 压缩前 | 线程内，一行提示 | 压缩是不可逆的信息丢失，值得打断一次 |
| 会话结束 | 后台进程，不占线程 | 常规总结不该干扰正在做的事 |

### 线程内那一次

```
[mem:harvest] 第 12 轮 / 约 63K token（压缩前）。如有值得记忆的内容则记录 1~5 条，无则不记。
```

只是一行尾巴，不额外开轮。触发后记一个自定义条目，避免重复提醒。

### 会话结束后那次

```
session_shutdown
  → 导出转录到 ~/.mem/transcripts/<sid>-<ts>.json
  → 写入任务队列（毫秒级，不阻塞退出）
  → detached 拉起 `mem.py harvest`，pi 退出后它继续跑
```

**提取是纯 CPU 的**，不调任何模型：

```
按句切分 → 规则打分（实测数字 / 强制语气 / 报错+解法 / 命令行）
        → 嵌入去重（与现有记忆 sim≥0.90 视为已覆盖）
        → 走完整写入路径（落 draft，不进检索）
```

写进去的是 `draft`：能看到、能 `promote`，但不会自动注入上下文。

### 关机了怎么办

任务写在 SQLite 里，不是内存里：

```
session_start
  → detached 拉起 `mem.py harvest --drain`
  → 捡起所有 pending，以及心跳超过 5 分钟的 running（上次进程被杀的）
```

所以关机、断电、任务管理器杀进程，都不会丢任务。下次启动 pi 自动补跑。

```bash
python mem.py harvest --list     # 看队列
python mem.py harvest --drain    # 手动补跑
```

---

## 会话锚点

每个会话开始写一条锚点，落两处：

- **DB**：`mem_anchors` 表（`mem.py anchor --show` 可查）
- **pi 条目**：`mem_anchor` 自定义条目，跟着转录走，压缩后仍在

锚点记录：会话 id、scope、cwd、上一个会话 id、上次遗留的自动候选数。

用途是给**短期工作记忆**留挂载点：将来把"这个会话正在做什么"挂在锚点上，
会话结束时随之失效，不会沉淀成长期记忆。

---

## 安全边界

- **`revoked` 永不复活**：SQL 触发器强制。
- **硬删除禁止**：只能软删除。
- **服务只监听 127.0.0.1**：无鉴权，同机进程可访问。

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

## License

MIT
