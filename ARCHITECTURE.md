# 记忆系统 · 架构与重构计划

> 目标：把 `mem.py` 从「扁平检索脚本」重构成「可扩展的记忆管线」，让图、双通道、重锚这些能力**长在架构里**，而不是外挂。
> 独立性：本 skill 与知识库完全解耦，自带依赖（numpy / onnxruntime / tokenizers / sqlite3）。

---

## 一、目标

| 要解决的问题 | 现状 | 目标 |
|---|---|---|
| 加图扩展要改检索函数 | 检索=加载+打分+渲染混在一起 | 拆成四层，加能力只改一层 |
| 稳定块/动态块无处存放 | 无会话概念 | `mem_sessions` 表 |
| 边表空着没人用 | `mem_edges` 存在但未使用 | `graph` 命令建边 |
| 打分不可单测 | 混在 IO 里 | 纯函数 |
| 逻辑边缺失 | 只有 replaces/merges | related / near_dup / contradicts |

## 二、架构：一条管线，三个出口

```
存储层 (SQLite)
  mem_items · mem_edges · mem_sessions · mem_graph_meta · mem_events
        ↓
① candidates(query, scope)   语义 top-N  ∪  图扩展 2 跳
        ↓
② rank(cands)                纯函数：(0.5·sim + 0.2·tags + 0.15·kw + 0.15·graph) / 固定分母 × decay
        ↓
③ select(ranked, policy)     stable / dynamic / anchor 三档分配
        ↓
④ render(items)              截断 65/35 + 格式化
        ↓
   ┌────────┬──────────┬──────────┐
 search    retrieve    snapshot
（拉取）   （注入）    （压缩）
```

**三个出口共用 ①②④，只在 ③ 分叉。**

## 三、数据模型

```sql
-- 记忆条目（已存在，不动）
mem_items(id, content, assertion, claim_type, origin, origin_model, origin_endpoint,
          verified, evidence_ref, verified_at, scope, tags, entities, importance,
          n_tok, created_at, valid_from, valid_to, revoked_at, superseded_by,
          merged_from, status, hot_score, hit_count, last_hit, cold, vector, reproduce)

-- 边（已存在，启用）
mem_edges(src_id, dst_id, etype, weight, meta, created_at)
--   etype: related (0.90–0.96) | near_dup (≥0.96) | contradicts | replaces | merges

-- 会话状态（新增）★ 注入式的基础
mem_sessions(session_id PK, scope, stable_text, stable_hash, stable_ids,
             anchor_text, anchor_ids, drift, reanchors, last_query,
             created_at, updated_at)

-- 图元信息（新增）★ 计数触发重建
mem_graph_meta(scope PK, built_at, item_count, edge_count)

-- 审计（已存在）
mem_events(event_id, ts, action, item_id, actor, detail)
```

## 四、接口

```bash
mem.py add / get / reinforce / supersede / forget / decay / stats / recheck
mem.py search   "q" --scope s [--k 5] [--json]        # 拉取式
mem.py retrieve --session X --query "q" [--json]      # 注入式 → {stable, tail}
mem.py graph    --scope s [--auto|--force]            # 建边
mem.py snapshot --session X                            # 压缩快照
```

## 五、重构步骤

| 步骤 | 内容 | 工时 | 验证 |
|---|---|---|---|
| **S1** | 加 3 张表 + `graph` 命令（建边 + 计数触发） | 2 h | 边数 > 0；重复运行可跳过 |
| **S2** | 拆管线：candidates / rank / select / render | 2 h | ✅ **已完成**：5/5 查询顺序与 token 数完全一致 |
| **S3** | `retrieve` + 会话状态 + 重锚 | 3 h | 连续 3 轮 stable.hash 不变 |

每步独立可验证；S2 前后用现有 13 条记忆做回归。

## 六、待办功能（重构后）

| # | 功能 | 工时 | 依赖 |
|---|---|---|---|
| 1 | `graph` 建边（related / near_dup / contradicts）+ 计数触发重建 | 2 h | — |
| 2 | 检索加 2 跳扩展 + BM25 补分 | 2 h | 1 |
| 3 | 打分改四分量固定分母 | 1 h | 2 |
| 4 | `retrieve` 注入接口（stable / tail） | 3 h | 需确认宿主 |

## 七、关键设计决策

| 决策 | 理由 |
|---|---|
| **边放 SQLite 表，不放 JSON 文件** | 避开 handoff 里的真实事故：邻接表文件名不带 hash 被覆盖，55 节点图变 2 节点 |
| **会话状态也放 SQLite** | 原子写、并发、GC、备份全部免费 |
| **打分是纯函数** | 可单测、可组件视图调参（handoff 强调"没有组件视图，调参就是盲调"） |
| **候选源可插拔** | 图扩展只是候选源之一，以后加 BM25/实体召回不动主流程 |
| **不做同义边** | 几何上「同义」与「高度相关」不可区分，用 `near_dup` 替代 |
| **阈值按本机语料实测，不照抄** | 见 §十 校准数据 |
| **分母固定** | handoff 实测：分母随 hasTags 变会让无标签条目白拿 25% |

## 八、继承的坑（来自 memctx 实测）

| 坑 | 处理 |
|---|---|
| ISO 日期 → `Number()` → NaN | 一律用 unix 时间戳（int） |
| 分母随 hasTags 变 | 固定分母 |
| `search` 不带 scope | scope 必填（除显式 `--all`） |
| stdout 混日志 | JSON 走 stdout，摘要走 stderr |
| 邻接表文件名不带 hash | 用 SQLite 表，绕开 |
| e5 维度/前缀/池化 | 384 / `query:` `passage:` / mean（已正确） |
| 绝对名次随语料膨胀 | 评测用相对 gate，不用绝对数 |
| 注入内容污染下轮检索 | `stripInjected`（S3 实现） |
| 改 system 重锚代价极高 | 重锚只改尾部 |

## 九、参数（来自 memctx 线上默认）

```
候选 30   扩展 2 跳   扩展种子 5   扩展上限 20
稳定块 8 条   动态块 3 条   单条 300 字符（65% 头 + 35% 尾）
动态相对地板 0.75   动态绝对地板 0.18
漂移阈值 2   重锚上限 3
权重 0.50 内容 + 0.20 标签 + 0.15 关键词 + 0.15 图
衰减 0.5^(天数/180)，pinned 不衰减
图边：余弦 ≥0.94（related）/ ≥0.97（near_dup），每节点 top-5，相对剪枝 ≥最强边×0.99
去重：余弦 ≥0.96
矛盾判定：余弦 ≥0.94 且（否定词不匹配 或 关键数字冲突）
```


## 十、相似度阈值校准（本机实测，2026-11-09）

e5-small 在中文上的相似度分布**极其压缩**，阈值必须实测，不能照抄 memctx 的 0.90。

| 样本 | 相似度区间 |
|---|---|
| 13 条互不相关的中文技术记忆 | 0.853 – 0.917 |
| 10 条互不相关的通识记忆 | 0.877 – 0.926 |
| 同一事实的同义改写 | 0.985 |
| 同一事实的相反表述（很快/很慢） | 0.9665 |

**结论与取值**：

| 用途 | 阈值 | 理由 |
|---|---|---|
| `related` 图边 | **0.94** | 0.90 会把一半无关条目连起来（实测 16/36 对 ≥0.90） |
| `near_dup` | **0.97** | 只有真正的近似重复 |
| `contradicts` | ≥0.94 + 启发式 | 低于 0.94 的相似度不足以谈"矛盾" |
| 写入去重 | **0.96** | 合并而非新增的门槛 |

> **换模型/换语料必须重跑这张表。** memctx 的 0.90 是在它自己的语料上测的，直接搬过来会得到一张噪声图。


## 十一、S2 交付记录（2026-11-09）

### 拆分结果

| 层 | 函数 | 职责 | 纯函数 |
|---|---|---|---|
| ① 候选 | `candidates(con, qv, scope, ...)` | 语义召回（S3 加图扩展） | 否（读库） |
| ② 打分 | `score_item(row, sim, now)` | 单条打分 | ✅ |
| ② 排序 | `rank(cands, now)` | 批量排序 | ✅ |
| ③ 选择 | `select_flat(ranked, k, budget)` | top-k + token 预算 | ✅ |

### 回归验证

重构前后各跑 5 条查询（DB 从同一基线快照恢复）：

| 查询 | 顺序一致 | token 数 | 最大分数差 |
|---|---|---|---|
| huggingface 下载不了 | ✅ | 324 = 324 | 0.0000 |
| bge 模型快不快 | ✅ | 352 = 352 | 0.0000 |
| 中文维基条目结构 | ✅ | 351 = 351 | 0.0000 |
| torch cuda 怎么开 | ✅ | 373 = 373 | **0.0001** |
| 三库统一 schema | ✅ | 349 = 349 | 0.0000 |

> 唯一差异来自 `score_item` 里的 `now`（两次运行相隔数分钟），非代码变更。

### 新增 `mem.py selftest`

10 项纯函数不变量：单调性、确定性、self 衰减、verified 权重、importance 单调、token 预算、truncated 标记、预算内/外返回条数。全部通过。


## 十二、待办 3 交付：四分量打分（2026-11-09）

### 实现

```
基础分 = (0.50·content + 0.20·tags + 0.15·keyword + 0.15·graph) / 1.00   ← 固定分母
最终分 = 基础分 × decay(claim_type, self×0.33, importance) × trust(verified)
```

- `content` = 原始余弦
- `tags` / `keyword` = 查询 2-gram 覆盖率（CJK 取相邻二字，拉丁按词）
- `graph` = 邻居基础分 × 边权 的最大值（两遍排序）
- **分母固定为 1.0**，不随是否有标签变化（避免无标签条目白拿 25%）
- `verified` / `importance` 改为**乘数**，不占分母（相关性、可信度、新鲜度是正交的）

### 实测影响（5 条查询，13 条记忆）

| 查询 | top1 是否不变 | 前 3 重合 |
|---|---|---|
| huggingface 下载不了 | ✅ | 2/3 |
| bge 模型快不快 | ❌ | 1/3 |
| 中文维基条目结构 | ✅ | 3/3 |
| torch cuda 怎么开 | ✅ | 2/3 |
| 三库统一 schema | ❌ | 0/3 |

### ⚠️ 发现：原始余弦本身不可区分

13 条记忆的 top-1 分数从旧公式的 **0.896** 变成新公式的 **0.454**，但**排序区分度没变好**——因为 e5 的余弦区间只有 0.85–0.92，content 分量（权重 0.5）贡献恒在 0.42–0.46，几乎不区分。

**下一步需要**：候选集内做 min-max 归一化（或 z-score），让 content 分量真正拉开差距。绝对相似度保留作「无匹配」的地板判断。

### 新增 selftest 检查

`verified=0 应低于 verified=2`、`图分应加分（邻居存在时）`、`关键词不匹配不应同分`、`无标签不应白拿分`——共 11 项，全部通过。


## 十三、查阅原插件后的修正 + 待办 2（2026-11-09）

### 关键修正：不做归一化

原插件 `lib/score.mjs` 源码注释：

> e5 把相似度压在 0.84–0.87 的窄带里，内容分方差几乎为 0，排序被 graph/keyword 的方差主导

他们的对策 `hybridWeight`（向量+BM25 混合）**默认 0**——handoff §10 记录「盲测 8–9，放弃」。

**结论：窄带是接受现状，靠 graph/keyword 方差区分，不靠内容分。**

### 分母修正（对齐源码）

```
第一遍（无图分）：除以 wContent + wTags + wKeyword = 0.85
第二遍（有图分）：除以 全部四项 = 1.00
```

无图条目不会被"少除 0.15"而白拿优势。

### 待办 2 已实现

| 组件 | 实现 |
|---|---|
| `build_bm25(docs)` | 纯本地 BM25（k1=1.2, b=0.75），CJK 2-gram + 拉丁词 |
| `expand_graph(graph, seeds, depth=2)` | 受限 BFS，每节点 top-5，最多加 20 |
| `candidates(..., top_n=30)` | 语义取 top-30 → 图扩展 → 扩展项用 **BM25/max 补 similarity** |
| `--candidates N` | 可调语义候选上限（验证扩展用） |

**BM25 语料必须是全量 scope，不是候选集**——否则扩展项不在语料里，得分恒为 0（踩过）。

### 验证

| 场景 | 结果 |
|---|---|
| demo，`--candidates 1` | 语义 1 条 + 图扩展 2 条；扩展项 BM25 补分 0.976 |
| demo，默认 | 3 条，HF 簇排前二 |
| local-rag，`--candidates 1` | 仅 1 条（HF 条目无图边，正确不扩展） |


## 十四、S3 交付：retrieve / snapshot（2026-11-09）

### 设计原则

**不假设宿主形态**：`retrieve` 只返回 JSON（stable/tail/hash/insert），由宿主决定怎么插。

### 接口

```bash
mem.py retrieve --session <id> --query "<本轮原话>" --scope <scope>
mem.py snapshot --session <id> [--query ...]
```

### 输出契约

```json
{
  "stable": {"text": "...", "hash": "dbddb259", "ids": [...], "cache": "new|hit"},
  "tail":   {"text": "...", "ids": [...], "anchor": true, "dynamic": 3},
  "insert": {"system": "append", "afterUserText": "

"},
  "debug":  {"candidates": 13, "best": 0.62, "floor": 0.46, "drift": 0,
             "reanchors": 0, "elapsedMs": 866}
}
```

### 实现要点

| 机制 | 实现 |
|---|---|
| 稳定块缓存 | 首轮生成并落 `mem_sessions`，之后**原样返回**（不再检索，保证逐字节一致） |
| hash 校验 | `sha256(text)[:8]`，用于跨进程/跨重启验证 |
| 漂移检测 | 本轮 top1 不在 stable.ids → drift+1；在 → 归零 |
| 重锚 | drift≥2 且 reanchors<3 → 用 `last_query + query` 重检索，**只写尾部**，不碰 stable |
| 动态块 | 排除 stable/anchor，`floor = max(0.18, best×0.75)`，取前 3 |
| 截断 | 65% 头 + 35% 尾，单条 300 字符 |
| 剥注入 | 检索前剥掉 `[memctx:now]` / `[memctx:anchor]` 之后的内容 |
| 会话 GC | 7 天未更新自动删除 |

### 验证

| 场景 | 结果 |
|---|---|
| 首轮 | `cache=new`，stable 8 条，hash `dbddb259` |
| 同 query 第二轮 | `cache=hit`，hash 不变 |
| 换主题 | drift+1（top1 不在 stable 时） |
| 连续两次偏离 | **drift 达 2 → 重锚触发**，`reanchors=1`，`anchor=true`，drift 归零 |
| `snapshot` | 返回 5 条 `[memctx:compaction]` 文本 |
| 剥注入 | 输入含 `[memctx:now]` 的 query，落库的 `last_query` 已剥干净 |

### 已知限制

13 条记忆 + STABLE_K=8 时，stable 覆盖了大部分，**漂移很少触发**（top1 大概率已在 stable 里）。这是数据规模问题，不是逻辑问题——记忆上量后才会体现。


## 缓存经济学（实测，2026-11-10）

### 两条注入通道

| 通道 | 机制 | 内容变化时的代价 |
|---|---|---|
| `systemPromptOptions.sections` | pi diff，只发变化的 section | **从该 section 位置起全量 miss** |
| `return { message }` | 追加到对话尾 | 只付该段 token |

### 实测数据

用 env 开关让一个 section 每轮变化，两组各 10 轮：

| 组 | 每轮 input | cacheRead | system 条目 |
|---|---|---|---|
| section 稳定 | 219–525 | 3968 → 6272（持续增长） | 1 |
| section 每轮变 | 2319 → 4481 | **卡死 2048** | 10（每轮一个补丁） |

`2048` 正好是 section 之前的长度。补丁虽然追加在末尾，
但开头的 system 消息被重建，缓存从 section 处断开。

### 规则

```
内容永不变化  → system sections（首次之后 0 成本）
内容会变化    → tail（每次只付该段 token）
```

**不要因为"它很稳定"就放进 system。** 会话内变化 3–5 次的东西
（段索引、域激活）放 system 的代价是放 tail 的 1000 倍。

### 成本

deepseek-v4.1-flash（opencode-go）：miss $0.15/M，hit $0.003/M（1/50）。

折算：`等效 token = miss + cacheRead × 0.02`

| 场景（上下文 / 每轮新增） | 增幅 |
|---|---|
| 20K / 500 | +15.5% |
| 50K / 2000 | +4.6% |
| 100K / 2000 | +3.5% |
| 150K / 3000 | +2.3% |

100 轮 / 100K 上下文：基线 $0.0600 → 有插件 $0.0621。

**稳态开销 88% 来自 tail 每轮 123 token。**
**一次重锚（100K 上下文）= 25 轮正常开销 = $0.015。**

优化优先级：省重锚，不是省注入。

### 怎么测量

读会话 JSONL 里 `message.usage` 的 `input` / `cacheRead`。
`cacheRead` 卡在某个固定值 = 失效点在该位置。

详见 `文档/方法论-实验与测量.md`。

---
