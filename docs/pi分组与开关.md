# pi 的分组与开关：怎么把技能拆成独立组

## 结论先说

**pi 里的"组" = 来源（source）**，不是你自己划的标签。想让一组技能有独立开关，就把它们做成一个 **Pi 包（package）**。

| 来源 | 显示成什么 | 能否单独开关 |
|---|---|---|
| `~/.agents/skills/*` 自动发现 | 「全局」一组 | 只能逐个开，不能整组拆 |
| skills.sh 装的（`.skill-lock.json`） | 「SKILLS.SH」组 | 组级开关 |
| **Pi 包** | **包名一组** | **组级开关** ✅ |

---

## 我们实际做了什么

### 1. 记忆 → `pi-mem` 包

```
E:\本地数据库\源码-记忆\v0.1\pi-package\
├── package.json
├── memskill.ts              ← 插件
└── skills/memory/           ← 技能（SKILL.md + mem.py）
```

```json
{
  "name": "pi-mem",
  "pi": {
    "extensions": ["./memskill.ts"],
    "skills": ["./skills"]
  }
}
```

```bash
pi install "E:/本地数据库/源码-记忆/v0.1/pi-package"
```

**效果**：`pi config` 里出现 `pi-mem` 组，**一个开关同时管插件和技能**。
插件开着时会自动屏蔽同名技能，关掉插件技能就回来——不会再出现两套逻辑打架。

### 2. 视频技能 → `video-skills` 包

```
E:\本地数据库\技能包\video-skills\
└── package.json
```

```json
{
  "name": "video-skills",
  "pi": {
    "skills": [
      "C:/Users/Cros/.agents/skills/hyperframes",
      "C:/Users/Cros/.agents/skills/hyperframes-cli",
      "... 共 8 个"
    ]
  }
}
```

同时在 `~/.pi/agent/settings.json` 里**排除原路径**，避免重复加载：

```json
{
  "skills": [
    "-C:/Users/Cros/.agents/skills/hyperframes",
    "-C:/Users/Cros/.agents/skills/hyperframes-cli",
    "..."
  ]
}
```

```bash
pi install "E:/本地数据库/技能包/video-skills"
```

**效果**：8 个视频技能从「全局」组移到 `video-skills` 组，独立开关，且**文件还在原位**（`~/.agents/skills/` 不动，其他工具照常能读）。

---

## 踩过的坑

| 做法 | 结果 |
|---|---|
| 包内放 **junction / 符号链接** | ❌ 不加载（pi 不跟随） |
| 包内放**真实目录** | ✅ 加载 |
| `pi.skills` 用 **`../` 相对路径** | ❌ 只相对包根解析 |
| `pi.skills` 用 **绝对路径** | ✅ **加载**（这是关键发现） |
| `settings.skills` 写 `-绝对路径` | ✅ 排除生效（`extensions` 的 `-path` 反而无效） |
| 不排除原路径 | ❌ 同一技能加载两次 |

---

## 操作步骤（任意技能组）

```bash
# 1. 建包目录
mkdir -p "E:/本地数据库/技能包/<组名>"

# 2. 写 package.json（skills 用绝对路径）
cat > "E:/本地数据库/技能包/<组名>/package.json" <<'EOF'
{
  "name": "<组名>",
  "pi": { "skills": ["C:/Users/Cros/.agents/skills/技能A", "C:/Users/Cros/.agents/skills/技能B"] }
}
EOF

# 3. 在 settings.json 排除原路径（否则会重复）
#    "skills": ["-C:/Users/Cros/.agents/skills/技能A", "-C:/Users/Cros/.agents/skills/技能B"]

# 4. 安装
pi install "E:/本地数据库/技能包/<组名>"

# 5. 看结果
pi list
pi config        # Tab 切全局/项目
```

---

## 包管理命令

| 命令 | 作用 |
|---|---|
| `pi install <path>` | 装到用户级 settings |
| `pi install -l <path>` | 装到项目级 `.pi/settings.json` |
| `pi list` | 看已装包 |
| `pi remove <source>` | 卸载 |
| `pi config` | TUI，Tab 切全局/项目，逐个开关 |
| `pi -e <path>` | 临时加载一个包（不写 settings） |

**项目级开关**：`.pi/settings.json` 里写 `"skills": ["-..."]` 或 `"packages": [...]`，只在当前项目生效。

---

## 当前状态

```
pi-mem         → 插件 + 记忆技能         （独立开关）
video-skills   → 8 个视频技能            （独立开关）
全局           → figma, go-usage, web-search
SKILLS.SH      → web-search, pdf
```
