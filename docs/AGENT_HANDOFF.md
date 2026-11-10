# AGENT HANDOFF · ΠMem

> 写给下一个我。压缩上下文或开新会话后**先读这个**，不要重新讨论已定的事。
> 最后更新：2026-11-10 · v0.9.0 已发布 https://github.com/MincHoo/Paimem

---

## 0. 三十秒速览

**ΠMem** 是给 pi 的本地长期记忆插件，已开源。

```
E:\本地数据库\
├── 源码-记忆\v0.1\pi-package\   ← **主仓库**（git，已推 GitHub）
├── 文档\                        ← 进度 / 方法论 / 功能清单
├── 语料-对话\                   ← 聊天记录语料 + 提取规范
├── 源码-知识库\v0.1\            ← 暂停中
└── 数据\                        ← raw 46G / out 21G / lancedb 8.5G / models 2.2G
```

**记忆数据在 `C:\Users\Cros\.mem\`**（mem.db + mem.toml + trust.toml + models + 日志 + transcripts）

---

## 1. 当前状态

| 项 | 值 |
|---|---|
| 版本 | **v0.9.0**（仅 Windows + pi 验证） |
| 仓库 | `E:/本地数据库/源码-记忆/v0.1/pi-package/` → github.com/MincHoo/Paimem |
| 记忆量 | global 100 条 active / sesseval 104 条（测试数据） |
| 质量 | 13 题 Hit@1 85% / Hit@3 92% / MRR 0.900；20 题盲测 Hit@3 80% |
| 延迟 | 检索 9–15 ms，注入 8–20 ms |
| 注入量 | stable 398 token（一次性）/ tail 123 token（每轮） |
| 成本 | **+3.5%** token（100K 上下文 / 每轮新增 2000） |
| 自检 | `python mem.py selftest` → 11/11 |

---

## 2. 架构

```
mem.py（~2600 行，单文件，零宿主依赖）
  ├── CLI 23 个子命令
  └── HTTP serve（8 端点）
        ↓
memskill.ts（~700 行，pi 专属胶水）
```

**所有逻辑在 mem.py。移植到别的 harness 只需重写 memskill.ts。**

### 数据流

```
写入：embed → 去重(0.96) → 冲突检测 → confirmed_by → 落库
检索：语义 top-30 ∪ 图 2 跳 → 四分量打分 → 衰减×信任×激活
注入：stable(8 条) → system sections（缓存零损失）
      tail(3 条)  → 尾部消息（每轮付费 ~123 token）

段索引：每轮 retrieve 顺带跟踪 → 硬上限闭合 → 落 mem_segments
        会话结束 → finalize --no-enrich（只收口，不调 LLM）

★ 2026-11-10 起改为【前台模式】：
  每轮结束时由主模型调 mem_turn_close → 本地收口 + 摘要 + 提升 + 冲突比对
  推翻旧记忆用 mem_supersede（或 payload 里的 supersedes）
  后台的 detached L2 / harvest 子进程默认关闭（MEM_FOREGROUND_ONLY=0 可回滚）
  详见 docs/记忆系统-段索引方案对比.md §15
```

---

## 3. ★ 宿主接口契约（移植时看这里）

这一节是**移植的唯一依据**。mem.py 不假设宿主形态，只要求宿主提供下面这些能力。

### 3.1 宿主必须提供的能力

| # | 能力 | 用途 | pi 里是什么 | DSH 里对应什么 |
|---|---|---|---|---|
| 1 | **注入系统提示段** | stable 块 | `before_agent_start` 改 `systemPromptOptions.sections.mem` | `ctx.systemPrompt.section({name, order, text})` |
| 2 | **追加对话消息** | tail 块 | `return { message }` | 消息流 append |
| 3 | **会话开始事件** | 写锚点、补跑队列 | `session_start` | `session/created` |
| 4 | **会话结束事件** | finalize、harvest | `session_shutdown` | `session/disposed` |
| 5 | **每轮开始事件** | 检索注入 | `before_agent_start` | `agent/pre-step` |
| 6 | **拿到当前用户输入** | 检索 query | `event.prompt` | 事件参数 |
| 7 | **拿到会话条目列表** | 段跟踪、token 估算 | `ctx.sessionManager.getEntries()` | 会话日志读取 |
| 8 | **拿到当前条目 id** | 段指针 | `entry.id` | 会话条目 id |
| 9 | **拿到会话文件路径** | 段原文指针 | `ctx.sessionManager.getSessionFile()` | 会话日志路径 |
| 10 | **压缩前事件**（可选） | snapshot | `session_before_compact` | `dsh-compaction` 后端 |
| 11 | **写自定义条目**（可选） | 锚点/段记录持久化 | `appendCustomEntry` | 插件状态存储 |

### 3.2 mem.py 对外暴露的接口

**CLI**（`python mem.py <cmd> --json`）：

| 命令 | 用途 | 关键参数 |
|---|---|---|
| `retrieve` | **每轮必调**：返回 stable/tail + 段跟踪结果 | `--session --query --scope --turn --entry-id --turn-tokens` |
| `snapshot` | 压缩前取 top-5 | `--session --query --scope` |
| `search` | 语义检索 | `query --scope -k` |
| `add` | 写入记忆 | `content --scope --origin --claim-type --evidence` |
| `anchor` | 写会话锚点 | `--session --scope --cwd --kind` |
| `harvest` | 会话结束异步总结 | `--transcript --drain --list` |
| `segment` | 段索引 | `add / list / get / activate / enrich / finalize` |
| `ingest` | 导入 pi-observational-memory 的观察 | `--session-file` |
| `trust` | 端点白名单 | `list / add / remove` |
| `config describe` | 参数说明（29 条） | — |
| `selftest` | 11 项不变量 | — |

**HTTP**（`mem.py serve --port 7788`，只监听 127.0.0.1）：

```
GET  /health
POST /retrieve {session,query,scope,exclude_ids,candidates,turn,entry_id,turn_tokens}
POST /search   {query,scope,k,include_cold,include_drafts}
POST /add      {content,scope,origin,claim_type,endpoint,evidence,tags}
POST /get      {ids}
POST /snapshot {session,query,scope}
```

**`retrieve` 的返回结构**（这是注入层的唯一契约）：

```json
{
  "v": 1,
  "stable": { "text": "...", "hash": "d20bc4bd", "ids": [...], "cache": "hit" },
  "tail":   { "text": "...", "ids": [...], "anchor": false, "dynamic": 3 },
  "insert": { "system": "append", "afterUserText": "\n\n" },
  "domains": [
    { "id": "seg-...", "scope": "home", "title": "...", "score": 1.0, "turns": [1,9] }
  ],
  "segment": {
    "action": "open|continue|close",
    "closed": { "id": "seg-...", "title": "...", "digest": "...", "turns": [1,9], "prev": "seg-..." }
  },
  "debug": { "candidates": 36, "best": 0.62, "drift": 0, "elapsedMs": 30 }
}
```

**`domains` 只在会话第一轮（`turn<=1`）有值**，宿主把它拼成 `[mem:domains] ...` 一行追加到 tail；第二轮起是空数组。

**`--transcript <会话 JSONL 路径>`** 是段索引的原文指针：retrieve 和 finalize 都要传，否则段无法做 L2 摘要（见 §6 坑 16）。

**`insert` 字段是给宿主看的提示**，不是命令：告诉宿主这段文本应该放哪。

### 3.3 适配层必须实现的三件事

```typescript
// 1. 每轮：拿 query → 调 retrieve → 把 stable 塞进系统提示段、tail 塞进消息流
on("before_agent_start", (event, ctx) => {
  const out = await retrieve({ session, query: event.prompt, scope, turn, entryId, turnTokens });
  if (out.stable.text) opts.sections.mem = out.stable.text;   // 稳定 → 系统提示
  if (out.tail.text) return { message: out.tail.text };        // 变化 → 消息流
  if (out.segment?.action === "close") appendTailLine(out.segment.closed);  // 段指针
});

// 2. 会话开始：写锚点 + 补跑上次没跑完的队列
on("session_start", () => { runPy(["anchor", ...]); spawnDetached(["harvest", "--drain"]); });

// 3. 会话结束：收口段 + L2 升级（detached，不阻塞退出）
on("session_shutdown", () => { spawnDetached(["segment", "finalize", "--session", sid]); });
```

### 3.4 宿主必须保证的两条不变量

1. **stable 块的文本逐字节不变时，不要改动系统提示**
   → 否则前缀缓存失效（实测：改 section 会让 cacheRead 卡死，每轮 input 从 340 涨到 3250）

2. **会话结束事件里不要跑同步的 LLM 调用**
   → 会卡住退出。用 detached 进程 + 任务队列（`harvest_jobs` 表）

---

## 4. 移植到 DSH（已调研）

### 4.1 DSH 是什么

`@deepseek-ai/dsh` v0.2.1-rc.5，Cordis 架构，**305 个插件包**。

```
profile 目录
  ├ package.json（dsh.profile.bundles 列出组合包）
  ├ cordis.patch.yml（用户 patch 层）
  └ node_modules（树外插件）
```

配置树从空根开始，依次叠加：bundles 的 patch → profile 的 cordis.patch.yml → home 级 patch → `--patch`。

### 4.2 事件系统（已确认存在）

```
session/created     session/disposed
agent/created       agent/disposed
agent/pre-step      ← 等价于 pi 的 before_agent_start
agent/assistant-stream
tools/pre-execute   tools/post-execute   ← 比 pi 更细
```

### 4.3 系统提示组装

```javascript
ctx.systemPrompt.section({ name: 'mem', order: 100, text: '...' })
ctx.systemPrompt.variable('name', () => 'value')     // {{name}} 插值
```

**关键差别**：DSH 每次 `assemble()` 重新拼整个提示词，**没有 pi 的 section 补丁机制**。

但这不影响我们的设计——pi 的补丁实测也是全量 miss。**两边行为一致，缓存结论直接适用**：

```
内容永不变化 → 系统提示（逐字节不变 → 前缀缓存命中）
内容会变化   → 消息流（每次只付该段）
```

### 4.4 压缩

`dsh-compaction` 提供**后端接口**（`dsh-compaction-basic` 是自带实现）。
实现一个后端就能接管压缩——这是移植的最大机会点，可能还没人做。

### 4.5 移植成本估计

| 部分 | 工作量 |
|---|---|
| mem.py（2600 行） | **0** —— 全是宿主无关的 |
| 适配层（约 700 行） | 全部重写 |
| 打包（Cordis 插件 + profile bundle） | 需学习，比 pi 的"一个 ts 文件"重 |

### 4.6 建议顺序

```
① 先验证段索引能不能跨会话接上（见 §8 待办 1）
② 写 DSH 最小适配（只做注入，~150 行），验证 agent/pre-step + systemPrompt.section
③ 验证通过再移植全部
```

---

## 5. 关键决策（别再翻案）

| 决策 | 理由 |
|---|---|
| **SQLite 而非 LanceDB** | 启动 248ms vs 1839ms；单文件可拷 |
| **ONNX int8 而非 torch** | 启动 355ms vs 9396ms；省 4GB |
| **图从向量几何长出来** | 不抽实体、无本体、零 LLM 写入；代价是图比较糊 |
| **self 衰减 0.9 不是 0.33** | 0.33 让 AI 自写记忆几乎检索不到（盲测 Hit@3 60% vs 90%） |
| **扩展项用真实余弦** | 归一化 BM25 能到 1.0，量纲压死语义 |
| **段索引进 tail 不进 system** | 会话内变 3-5 次，放 system 每次全量重算 |
| **硬上限切段，不做漂移检测** | 漂移信号不可用（见 §6） |
| **只提升 decision / rejected** | 这两类最怕忘，数量可控（≤4/段） |
| **只做前台，不再维持双路** | 前后台并用要协调两套逻辑（谁能推翻谁、谁先写、冲突谁负责），维护成本超过收益。例：后台切片永远拿不到跨段语境，“赞同/可以”无法还原 |
| **L2 提示词核心是“去指代”** | 用户说"可以"时必须还原出他赞同什么 |

---

## 6. 踩过的坑（按代价排序）

### 致命级

1. **归一化 BM25 冒充 similarity** —— 能到 1.0，而 e5 余弦只有 0.80–0.89。一条纯词面匹配的无关记忆压掉所有语义候选。
2. **`gc_sessions` 不 commit** —— DELETE 开了写事务不提交会锁死整库。
3. **Windows 管道默认 GBK** —— `↳` 触发 `UnicodeEncodeError`，子进程静默失败。
4. **改 system section 会让缓存从该位置起全部失效** —— 实测 cacheRead 卡死在 2048。
5. **话题漂移检测不可用** —— 12 轮真实对话，真实边界余弦 0.862，同话题内部最低 0.827；2-gram 12 对里 8 对是 0.000。
6. **纯余弦做域激活不可用** —— 50 段时信号/噪声只差 0.01–0.03。必须加关键词分量（语义 0.65 + 关键词 0.35 后 4/4 正确）。

### 操作级

7. **`-path` 排除对自动发现的扩展无效**
8. **junction 不跟随** —— `pi.skills` 要用绝对路径
9. **llama-server 的思考模型** —— 不关 thinking 会把 token 全烧在 reasoning 上，`content` 返回空
10. **Cloudflare 拦 Python urllib 默认 UA** —— 403 / error 1010，要加 `User-Agent`
11. **deepseek-v4.1-flash 默认带思考** —— 要 `"thinking": {"type": "disabled"}`
12. **热重载后 `lastSegTokens=0`** —— 首轮 delta = 整个会话 token 数，瞬间触发段上限
13. **改表结构后 SQLite 不自动加列** —— 必须 `DROP TABLE` 让 `init_db` 重建，否则静默失败
14. **段原文路径从未落库** —— `_seg_close` 曾把 `transcript` 硬编码成 `''`，导致 `segment enrich` 全部 `source_too_short` 跳过、`enriched` 永远是 0、`segment get` 取不回原文。现已改为 retrieve/finalize 传 `--transcript`，`init_db` 用 `ALTER TABLE` 补列（不再 DROP）。
15. **热重载只换宿主代码，不换常驻 daemon** —— 改了 mem.py 必须杀掉 7788 的 `serve` 进程，否则新一轮仍走旧内存代码。

### 评测级

14. **turn 粒度的 provenance 不能当真值** —— 助手一轮回答覆盖多个话题
15. **自己出题有乐观偏差** —— 手写 92% vs 盲测 70%，真实水平在中间

---

## 7. 参数速查（全部可在 mem.toml 改）

```
嵌入    e5-small 384 维 int8 ONNX，passage:/query: 前缀由调用方加
去重    0.96
图边    related ≥0.88+共享标签 或 ≥0.95；near_dup ≥0.97；contradicts ≥0.94+词面
图剪枝  每节点 5 条（contradicts 也是 5）
打分    0.50 content + 0.20 tags + 0.15 keyword + 0.15 graph
归一化  候选内 min-max，α=0.5，span<0.02 不归一
衰减    fact 35d / preference 180d / decision 90d / procedure 120d / observation 7d；self ×0.9
激活    命中后 +15%，30 分钟半衰
信任    verified=2 → 1.0，verified=0 → 0.85
注入    STABLE_K 8 / DYNAMIC_K 3 / 单条 180 字
段      硬上限 15 轮 / 12000 token；漂移阈值 0（关闭）
L2      只提升 decision/rejected，≤4/段
配额    5 写/分、20 查/分、300 条/天
```

**换嵌入模型必须重测全部相似度阈值。** 完整说明：`python mem.py config describe`（29 条）

---

## 8. 待办

| # | 任务 | 说明 |
|---|---|---|
| **1** | **验证段索引** | 见 §8.1 测试流程 |
| 2 | 段索引注入完整列表 | 现在只在闭合时追加一行，压缩后没重新注入。方案对比见 `docs/记忆系统-段索引方案对比.md` |
| 3 | DSH 最小适配 | ~150 行，验证 `agent/pre-step` + `systemPrompt.section` |
| 4 | 自动化测试 | 现在只有 selftest |
| 5 | 陈述提升的闸门 | 重复出现的决策权重更高 |
| 6 | 补"元认知"类记忆 | 库里缺"为什么这么做" |
| ⏸ | Linux/macOS 验证 | 用户明确说短期不做 |

---

### 8.1 段索引测试流程（明天第一步）

**前提**：已关闭旧窗口，`segment finalize` 已跑完（收口段 + L2 摘要 + 提升陈述）

**测试步骤**：

```bash
# 1. 先看库里有几段
cd "E:/本地数据库/源码-记忆/v0.1/pi-package"
python mem.py segment list

# 2. 开新窗口（同一工作区 C:/Users/Cros），随便说一句
#    正常情况：第一轮 tail 里应出现
#    [mem:domains] 相关历史会话：seg-xxx「标题」、seg-yyy「标题」

# 3. 如果没出现，手动验证检索是否工作
python mem.py segment activate --query "继续上次那个段索引的设计" --scope home
```

**通过标准**：

| 检查 | 期望 |
|---|---|
| `segment list` | 有昨天产生的段，`enriched=1`、`has_src=1` |
| 新会话第一轮 | tail 出现 `[mem:domains]` |
| `segment activate` | 能命中昨天关于“段索引/记忆插件”的段 |
| `segment get <id> --part` | 能取回原文（`text` 非空） |

**失败排查**：

- 段表空 → `finalize` 没跑（看 `~/.mem/pi-adapter.log` 的 `segment_finalize_spawn`）
- `enriched=0` / `has_src=0` → 段的 `transcript` 为空，**不是** L2 调用失败；查 retrieve 有没有带 `--transcript`
- `[mem:domains]` 没出现 → 先看 `turn` 日志里的 `domains`/`firstTurn` 字段：`firstTurn:true, domains:0` 是 python 侧没算出来；`domains>0` 但 tail 没有，是宿主侧拼装问题
- `activate` 无结果 → 段的 scope 与查询 scope 不匹配（用 `--all` 试）

---

## 9. 用户偏好（重要）

- **要直答**，不要客套和铺垫
- **要数字和实测**，不要"应该没问题"
- **不要浮夸**：README 里"三个不一样的地方"这类措辞被要求改掉
- **不要 diss 别人**：不要写"别人怎么差"
- **不要暴露项目细节**：公开仓库里不能出现本地路径、私有项目名
- **每个大步骤**：① 写进 md ② 明确主线 ③ 屏幕输出进度
- **会主动质疑设计**（如"衰减惩罚是不是太高"），要认真验证而不是辩护
- **不主动跑实测、不起服务、不下载模型**，只给命令和判断依据，需要动手前先问

---

## 10. 文件地图

| 文件 | 作用 |
|---|---|
| `pi-package/mem.py` | 全部逻辑（~2600 行单文件） |
| `pi-package/memskill.ts` | pi 适配层（~700 行） |
| `pi-package/README.md` | 公开文档 |
| `pi-package/ARCHITECTURE.md` | 设计决策与校准数据 |
| `pi-package/eval_mem.py` | 质量评测（13 题 + 3 专项） |
| `docs/ΠMem-待办与进度.md` | **主线文档，每次任务后更新** |
| `docs/方法论-实验与测量.md` | 缓存测量 / A/B 实验 / 成本折算 / 评测法 |
| `docs/ΠMem-功能清单.md` | 九节功能清单 |
| `docs/ΠMem-GitHub简介.txt` | 仓库信息 + 批改记录 |
| `语料-对话/提取规范.md` | 记忆提取格式规范 |

---

## 11. 验证一切正常

```bash
cd "E:/本地数据库/源码-记忆/v0.1/pi-package"
python mem.py selftest              # 11 项不变量
python mem.py stats                 # 库况
python eval_mem.py                  # 质量：Hit@k / MRR + 3 专项
python mem.py segment list          # 段索引
python mem.py trust list            # 端点白名单
```

真实环境冒烟：

```bash
cd C:/Users/Cros && timeout 240 pi -p --session-id smoke "好"
# 看 ~/.mem/pi-adapter.log，应有 turn 事件、elapsedMs 个位数
```

---

## 12. 两个尚未验证的猜想

1. **pi 的 `return { message }` 每轮累积** —— 长会话下 tail 会堆到几十 KB。实测 8 轮没问题，但没测 50 轮。
2. **段索引跨会话能否真的接上** —— 这是待办 #1，还没跑过真实测试。

---

## 13. 一句话总结

> 用一套嵌入模型同时做检索、建图、判冲突，靠 SQLite 单文件跑起来，
> 在 pi 上实现"缓存友好的记忆注入"，成本约 1 毛钱 / 100 轮。
> 段索引给被压缩的历史留一个指针——不是替代压缩，是给压缩加撤销键。
