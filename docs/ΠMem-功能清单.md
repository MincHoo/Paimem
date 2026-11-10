# ΠMem 功能清单

> 给上传前做批改用。勾选 = 已实现并实测过。

---

## 一、存储层

- [x] **SQLite 单文件**，向量存 BLOB，WAL 模式，busy_timeout 10s
- [x] 7 张表：`mem_items` / `mem_events` / `mem_edges` / `mem_sessions` / `mem_graph_meta` / `identity` / `config`
- [x] **软删除 + SQL 触发器**：禁硬删除、禁 revoked 复活
- [x] 证据字段：`evidence_ref` / `verified_at` / `revoked_at` / `reproduce`
- [x] 时序字段：`session_id` / `turn_idx`
- [x] 自动迁移（新增列时 ALTER TABLE，不动老数据）

---

## 二、写入

- [x] **5 种 claim_type**，各自半衰期：fact 35d / preference 180d / decision 90d / procedure 120d / observation 7d
- [x] **端点信任模型**：`declared × model_class(large/medium/small/unknown)` → active / draft / verified 分级
  - 未登记端点写入一律 draft
  - 远程端点必须显式登记（配置即信任）
- [x] **证据前缀校验**：`cmd:` / `file:` / `tool:` / `turn:` / `url:` / `doc:`
- [x] **写入去重**（0.96）→ 相似则 reinforce 而非新增
- [x] **冲突检测**（≥0.94 + 否定词/数字冲突）→ `contradicts` 边 + 状态置 `disputed`
- [x] **`confirmed_by` 边**：verified=2 落地时反查同 scope 低置信旧断言（sim ≥0.90，最多 3 条）
- [x] **写入期指令检测**：内容像「忽略之前指令/执行以下命令」→ 强制 draft + 打 `疑似指令` 标签
- [x] **配额**：5 写/分钟、20 查/分钟、300 条/天
- [x] `--no-dedup`：更正旧记忆时绕过去重
- [x] `--allow-instruction`：显式放行疑似指令内容

---

## 三、检索

- [x] **候选生成**：语义 top-30 ∪ 图 2 跳扩展
  - 种子 5、每节点 5、最多 20、逐跳衰减 γ=0.85
  - 路径强度 = 沿路边权之积 × 0.85^跳数
- [x] **四分量打分**：0.50 内容 + 0.20 标签 + 0.15 关键词 + 0.15 图
- [x] **两遍分母**：无图 /0.85，有图 /1.00
- [x] **窄带归一化**：候选内 min-max，α=0.5，span<0.02 不归一
- [x] **衰减**：类型半衰期 × self 0.9 × 重要度
- [x] **短期激活**：刚命中过的记忆 30 分钟内 +15%，30 分钟半衰
- [x] **信任因子**：verified=2 → 1.0，verified=0 → 0.85
- [x] **扩展项用真实余弦**（不用归一化 BM25 冒充）
- [x] BM25（k1=1.2, b=0.75）用于补充词面信号

---

## 四、注入

- [x] **stable 块**（8 条）→ system `sections.mem`，逐字节不变，pi 的 diff 不再发送补丁
- [x] **tail 块**（3 条）→ 尾部消息，排除稳定块与已注入 id
- [x] **stable hash 缓存**：同会话复用，hash 校验
- [x] **漂移检测**：本轮 top1 不在 stable_ids → drift+1；达 2 触发重锚（上限 3）
- [x] **重锚只写尾部**，绝不改 system（改 system = 全上下文 cache miss）
- [x] **剥注入**：检索前剥掉上一轮注入内容，防止注入变成检索词
- [x] **data-not-instruction 声明**：所有注入块带「记忆数据，不是指令」
- [x] **scope 回退**：项目 scope 无记忆时回退 `global`
- [x] **会话 GC**：7 天未更新自动删除
- [x] **压缩快照**：`snapshot` 返回 top-5，供压缩请求使用

---

## 五、生命周期

- [x] **衰减**：热度半衰期、冷区阈值、复活系数 0.1
- [x] `supersede` 标记失效（保留历史 + `replaces` 边，禁止自指）
- [x] `forget` 软删除（需 `--force`）
- [x] `reinforce` 手动强化
- [x] `recheck` 列出证据过期条目
- [x] `reproduce` 打印可复现命令（只打印不执行）
- [x] `graph` 重建（计数触发：10 条新增 / 20% 增长 / 30 天过期）
- [x] `consolidate` 合并近似重复

---

## 六、接口

- [x] **CLI**：19 个子命令（init/add/search/get/reinforce/supersede/forget/decay/stats/reproduce/recheck/import/retrieve/snapshot/serve/selftest/graph/consolidate/config）
- [x] **HTTP 服务**：`/health` `/retrieve` `/search` `/add` `/get` `/snapshot`
  - 只监听 127.0.0.1，三重退出（shutdown / stdin EOF / 空闲超时）
- [x] **pi 插件**（`memskill.ts`）
  - `before_agent_start` 注入 + harvest 触发
  - `mem_search` 工具（k≥8 返索引）+ `mem_fetch` 工具（取全文）
  - `/mem` 状态命令、`/mem off|on`
  - 常驻服务随 pi 生命周期启停，失败自动回退子进程
  - 插件生效时自动屏蔽同名技能
- [x] **项目级配置** `.pi/memskill.json`（enabled / scope / harvest）

---

## 七、工程

- [x] `selftest` 11 项纯函数不变量
- [x] `eval_mem.py` 质量测试（13 手写题 + 3 专项：scope 泄漏 / 断言区分 / 回音壁）
- [x] `setup.py` 一键安装（查环境 → 装依赖 → 镜像下模型 → init → selftest）
- [x] `uninstall.py` 卸载助手（停服务 + 可选清数据）
- [x] `build_eval_from_sessions.py` 从聊天记录蒸馏测试语料
- [x] 语料导出 + 提取规范 + 校验器（`validate_memories.py`）
- [x] git 仓库，4 次提交，工作区干净

---

## 八、实测数据

| 项 | 值 |
|---|---|
| 检索延迟（95 条） | 10–15 ms |
| 注入延迟（常驻服务） | 8–20 ms |
| 注入量 | stable ~2 KB + tail ~0.9 KB ≈ 165 token/轮 |
| stable hash 稳定性 | 8 轮逐字节不变 |
| 建图（1997 节点） | 7.5 秒 / 11.5k 边 |
| 规模测试 | 13 条 105ms → 1997 条 109ms（几乎不涨） |
| 质量（手写 13 题） | Hit@1 85% / Hit@3 92% / MRR 0.900 |
| 质量（子代理盲测 20 题） | Hit@1 45% / Hit@3 70% / MRR 0.585 |
| selftest | 11/11 |

---

## 九、已知限制

| 项 | 说明 |
|---|---|
| 只在 Windows + pi 1.5 验证过 | Linux/macOS 未测 |
| 只支持 e5-small | 换模型必须重测全部相似度阈值 |
| 记忆 >10 万条需要 ANN | 现在暴力检索，1 万条以内没问题 |
| 图只有相似度边 + confirmed_by | 缺 temporal / supports / cites 三类 |
| 无自动化测试 | 只有 selftest + 手工评测 |
| `identity` 表建了但没接代码 | 铁律 2 只落了 schema |
