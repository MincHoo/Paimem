/**
 * memskill — 把本地记忆库接入 pi
 *
 * 三条通道：
 *   stable（8 条长期记忆）→ systemPromptOptions.sections.mem   （缓存零损失）
 *   tail  （3 条当前相关）→ return message                      （尾部追加，前缀不破）
 *   harvest（该总结了）    → 拼进 tail 末尾，不额外开一轮
 *
 * 常驻服务（与插件生命周期绑定）：
 *   session_start    → 拉起 `mem.py serve`（子进程，stdin 管道）
 *   session_shutdown → 关闭
 *   父进程崩溃       → stdin EOF → 服务自杀（不留孤儿）
 *   1 小时无请求     → 自动退出
 *   服务不可用       → 自动回退到每次起子进程（慢但可用）
 *
 * 环境变量：
 *   MEM_SCOPE      记忆 scope（默认按 cwd 推导）
 *   MEM_HARVEST=0  关闭自动总结（默认开）
 *   MEM_DAEMON=0   关闭常驻服务，只用子进程
 *   MEM_PORT       服务端口（默认 7788）
 *   MEM_DRY_RUN=1  只记录不注入
 *   MEM_DISABLE=1  完全关闭
 *   MEM_PY / MEM_PY_PATH / MEM_TIMEOUT / MEM_ENDPOINT
 */

import { type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { execFile, spawn, type ChildProcess } from "node:child_process";
import { promisify } from "node:util";
import { appendFile, mkdir } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const execFileAsync = promisify(execFile);

/** 扩展所在目录（包根），mem.py 与它同级 */
const PKG_DIR = dirname(fileURLToPath(import.meta.url));

/** Python 解释器：优先 MEM_PY，其次自动探测（Windows 商店占位符会失败） */
let pyBin = process.env.MEM_PY ?? "";
const PY_CANDIDATES = ["python3", "python"];

async function resolvePython(): Promise<string | null> {
	if (pyBin) return pyBin;
	for (const c of PY_CANDIDATES) {
		try {
			const { stdout } = await execFileAsync(c, ["-c", "print(1)"], {
				timeout: 5000,
				windowsHide: true,
			});
			if (stdout.trim() === "1") {
				pyBin = c;
				return c;
			}
		} catch {
			/* 试下一个 */
		}
	}
	return null;
}

const MEM_PY = process.env.MEM_PY_PATH ?? join(PKG_DIR, "mem.py");
const TIMEOUT = Number(process.env.MEM_TIMEOUT ?? 20000);
const DRY = process.env.MEM_DRY_RUN === "1";
const DISABLED = process.env.MEM_DISABLE === "1";
let harvestEnabled = process.env.MEM_HARVEST !== "0";
/** 写入记忆时声明的端点。留空 = 未登记 → 写入为 draft（更安全） */
const ENDPOINT = process.env.MEM_ENDPOINT ?? "";
/** 插件生效时要屏蔽的同名技能（避免"注入"和"技能"两套逻辑打架） */
const HIDE_SKILL = process.env.MEM_HIDE_SKILL ?? "memory";
/** 当前 scope 没有记忆时回退到哪个 scope */
const FALLBACK_SCOPE = process.env.MEM_SCOPE_FALLBACK ?? "global";
const USE_DAEMON = process.env.MEM_DAEMON !== "0";
const PORT = Number(process.env.MEM_PORT ?? 7788);
const BASE = `http://127.0.0.1:${PORT}`;

const MEM_HOME = process.env.MEM_HOME ?? join(homedir(), ".mem");
const LOG_PATH = join(MEM_HOME, "pi-adapter.log");

// ---- 触发阈值（实测：本会话 123 轮 / 453K token / 平均 3.7K 每轮）----
const TOKENS_FIRST = Number(process.env.MEM_TOKENS_FIRST ?? 60_000);
const TOKENS_STEP = Number(process.env.MEM_TOKENS_STEP ?? 40_000);
const TURNS_STEP = Number(process.env.MEM_TURNS_STEP ?? 10);
const TURNS_MIN_TOKENS = Number(process.env.MEM_TURNS_MIN_TOKENS ?? 2_000);
const CHARS_PER_TOKEN = 3.3;

type RetrieveOut = {
	v: number;
	stable: { text: string; hash: string; ids: string[]; cache: string };
	tail: { text: string; ids: string[]; anchor: boolean; dynamic: number };
	debug: {
		candidates: number;
		best: number;
		floor: number;
		drift: number;
		reanchors: number;
		elapsedMs: number;
	};
};

type Stats = {
	turns: number;
	cacheNew: number;
	cacheHit: number;
	stableBytes: number;
	tailBytes: number;
	injected: number;
	harvests: number;
	lastHash: string | null;
	lastTail: string;
	lastDebug: RetrieveOut["debug"] | null;
	errors: number;
	viaDaemon: number;
	viaSpawn: number;
};

const stats: Stats = {
	turns: 0,
	cacheNew: 0,
	cacheHit: 0,
	stableBytes: 0,
	tailBytes: 0,
	injected: 0,
	harvests: 0,
	lastHash: null,
	lastTail: "",
	lastDebug: null,
	errors: 0,
	viaDaemon: 0,
	viaSpawn: 0,
};

let enabled = !DISABLED;
let scope = process.env.MEM_SCOPE ?? "default";
let injectedIds: string[] = [];
let pendingCompact = false;
let skillHidden = false;

// ---- 常驻服务 ----
let daemon: ChildProcess | null = null;
let daemonReady: Promise<boolean> | null = null;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function health(ms = 600): Promise<boolean> {
	if (!USE_DAEMON) return false;
	try {
		const r = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(ms) });
		return r.ok;
	} catch {
		return false;
	}
}

async function ensureDaemon(): Promise<boolean> {
	if (!USE_DAEMON) return false;
	if (daemon && daemon.exitCode === null && (await health(300))) return true;
	if (await health(300)) return true; // 别的 pi 实例已经拉起
	try {
		const py = await resolvePython();
		if (!py) {
			await log({ t: Date.now(), event: "daemon_fail", reason: "no_python" });
			return false;
		}
		daemon = spawn(py, [MEM_PY, "serve", "--port", String(PORT)], {
			stdio: ["pipe", "pipe", "pipe"],
			windowsHide: true,
		});
		daemon.on("exit", () => {
			daemon = null;
		});
		// 吞掉输出，避免污染终端
		daemon.stdout?.resume();
		daemon.stderr?.resume();
		for (let i = 0; i < 80; i++) {
			await sleep(250);
			if (await health(300)) {
				await log({ t: Date.now(), event: "daemon_start", pid: daemon?.pid, port: PORT });
				return true;
			}
			if (daemon === null) break; // 进程已退出
		}
	} catch {
		/* fall through */
	}
	await log({ t: Date.now(), event: "daemon_fail", port: PORT });
	return false;
}

function stopDaemon() {
	const pid = daemon?.pid;
	try {
		daemon?.stdin?.end(); // 触发服务端 stdin EOF → 自杀
		daemon?.kill();
	} catch {
		/* ignore */
	}
	// Windows 下 kill 偶尔不生效，补一刀
	if (pid && process.platform === "win32") {
		try {
			spawn("taskkill", ["/F", "/PID", String(pid)], { windowsHide: true, stdio: "ignore" });
		} catch {
			/* ignore */
		}
	}
	daemon = null;
	daemonReady = null;
}

async function callDaemon(path: string, body: unknown): Promise<any | null> {
	try {
		const r = await fetch(`${BASE}${path}`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(body),
			signal: AbortSignal.timeout(TIMEOUT),
		});
		if (!r.ok) return null;
		return await r.json();
	} catch {
		return null;
	}
}

/** 统一入口：优先常驻服务，失败回退子进程 */
async function runMem(args: string[], path: string, body: unknown): Promise<any | null> {
	if (USE_DAEMON) {
		if (!daemonReady) daemonReady = ensureDaemon();
		const ok = await daemonReady;
		if (ok) {
			const r = await callDaemon(path, body);
			if (r) {
				stats.viaDaemon++;
				return r;
			}
			daemonReady = null; // 服务挂了，下次重连
		}
	}
	try {
		const py = await resolvePython();
		if (!py) {
			stats.errors++;
			await log({ t: Date.now(), event: "mem_error", error: "python_not_found" });
			return null;
		}
		const { stdout } = await execFileAsync(py, [MEM_PY, ...args], {
			timeout: TIMEOUT,
			maxBuffer: 16 * 1024 * 1024,
			windowsHide: true,
			env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
		});
		stats.viaSpawn++;
		return JSON.parse(stdout);
	} catch (e) {
		stats.errors++;
		await log({
			t: Date.now(),
			event: "mem_error",
			args: args.slice(0, 2),
			error: String(e).slice(0, 2000),
		});
		return null;
	}
}

/** 读取分层配置：全局 ~/.pi/agent/memskill.json + 项目 <cwd>/.pi/memskill.json */
function loadConfig(cwd: string): Record<string, any> {
	const out: Record<string, any> = {};
	const read = (p: string) => {
		try {
			if (!existsSync(p)) return {};
			return JSON.parse(readFileSync(p, "utf8")) ?? {};
		} catch {
			return {};
		}
	};
	Object.assign(out, read(join(MEM_HOME, "..", ".pi", "agent", "memskill.json")));
	Object.assign(out, read(join(homedir(), ".pi", "agent", "memskill.json")));
	Object.assign(out, read(join(cwd, ".pi", "memskill.json")));
	return out;
}

function resolveScope(cwd: string): string {
	if (process.env.MEM_SCOPE) return process.env.MEM_SCOPE;
	const norm = (s: string) => s.replace(/\\/g, "/").toLowerCase().replace(/\/+$/, "");
	if (norm(cwd) === norm(homedir())) return "home";
	return basename(cwd) || "default";
}

async function log(rec: Record<string, unknown>) {
	try {
		await mkdir(MEM_HOME, { recursive: true });
		await appendFile(LOG_PATH, JSON.stringify(rec) + "\n", "utf8");
	} catch {
		/* ignore */
	}
}

/** 数轮次与累计字符（n 通常 <2000） */
function scanSession(entries: any[]): { turns: number; chars: number } {
	let turns = 0;
	let chars = 0;
	for (const e of entries) {
		if (e.type === "custom_message" || e.type === "custom") continue;
		const m = e.message;
		if (!m) continue;
		if (m.role === "user") turns++;
		const c = m.content;
		if (typeof c === "string") chars += c.length;
		else if (Array.isArray(c)) {
			for (const b of c) {
				if (!b || typeof b !== "object") continue;
				if (typeof b.text === "string") chars += b.text.length;
				if (typeof b.thinking === "string") chars += b.thinking.length;
				if (b.type === "toolCall" && b.arguments) chars += JSON.stringify(b.arguments).length;
			}
		}
	}
	return { turns, chars };
}

function lastHarvest(entries: any[]): { turn: number; tokens: number } {
	for (let i = entries.length - 1; i >= 0; i--) {
		const e = entries[i];
		if (e.type === "custom" && e.customType === "mem_harvest" && e.data) {
			return { turn: e.data.turn ?? 0, tokens: e.data.tokens ?? 0 };
		}
	}
	return { turn: 0, tokens: 0 };
}

function harvestLine(turn: number, tokensK: number, reason: string): string {
	const cmd =
		`python "${MEM_PY}" add "<一句话事实>" --scope ${scope} --origin self` +
		(ENDPOINT ? ` --endpoint ${ENDPOINT}` : "") +
		` --claim-type <fact|preference|decision|procedure|observation>`;
	return (
		`[mem:harvest] 第 ${turn} 轮 / 约 ${tokensK}K token（${reason}）。` +
		`如有值得记忆的内容则记录 1~5 条，无则不记。命令：${cmd}` +
		`；若发现旧记忆已过时，先 add --no-dedup 写新的，再 supersede <旧id> <新id> 更正。`
	);
}

export default function (pi: ExtensionAPI) {
	pi.on("session_start", async (_e, ctx) => {
		// 首次运行：库还没建，提示怎么装
		if (!existsSync(join(MEM_HOME, "mem.db"))) {
			ctx.ui.notify(
				`ΠMem 还没有初始化。
运行：python "${MEM_PY}" init
或先跑 setup.py 装依赖和模型。`,
				"warning",
			);
		}
		stats.turns = 0;
		stats.cacheNew = 0;
		stats.cacheHit = 0;
		stats.stableBytes = 0;
		stats.tailBytes = 0;
		stats.injected = 0;
		stats.harvests = 0;
		stats.lastHash = null;
		stats.lastTail = "";
		stats.lastDebug = null;
		stats.errors = 0;
		stats.viaDaemon = 0;
		stats.viaSpawn = 0;
		injectedIds = [];
		pendingCompact = false;
		const cwd = ctx.sessionManager.getCwd?.() ?? process.cwd();
		const cfg = loadConfig(cwd);
		if (cfg.enabled === false) enabled = false;
		if (typeof cfg.harvest === "boolean") harvestEnabled = cfg.harvest;
		if (typeof cfg.scope === "string" && cfg.scope) scope = cfg.scope;
		else scope = resolveScope(cwd);
		// 后台拉起常驻服务，不阻塞会话启动
		if (enabled && USE_DAEMON && !daemonReady) daemonReady = ensureDaemon();
	});

	// 插件退出 / 重载 / 切会话 → 关掉服务
	pi.on("session_shutdown", async (event) => {
		if (event.reason === "reload" || event.reason === "quit") stopDaemon();
		await log({ t: Date.now(), event: "shutdown", reason: event.reason });
	});

	pi.on("session_before_compact", async (event) => {
		if (!harvestEnabled) return;
		pendingCompact = true;
		await log({
			t: Date.now(),
			event: "compact_before",
			reason: event.reason,
			tokensBefore: event.preparation?.tokensBefore,
		});
	});

	pi.on("before_agent_start", async (event, ctx) => {
		if (!enabled) return;
		if (event.messageType !== "user") return;
		const query = (event.prompt ?? "").trim();
		if (!query) return;

		const sid = ctx.sessionManager.getSessionId();
		const args = (sc: string) => [
			"retrieve",
			"--session",
			sid,
			"--query",
			query,
			"--scope",
			sc,
			...(injectedIds.length ? ["--exclude-ids", injectedIds.join(",")] : []),
		];
		let out = (await runMem(args(scope), "/retrieve", {
			session: sid,
			query,
			scope,
			exclude_ids: injectedIds,
		})) as RetrieveOut | null;
		if (!out) return;
		// 本项目还没有记忆 → 回退到全局
		let usedScope = scope;
		const fbTried = (out.debug?.candidates ?? 0) === 0 && !!FALLBACK_SCOPE && FALLBACK_SCOPE !== scope;
		if (fbTried) {
			const fb = (await runMem(args(FALLBACK_SCOPE), "/retrieve", {
				session: sid,
				query,
				scope: FALLBACK_SCOPE,
				exclude_ids: injectedIds,
			})) as RetrieveOut | null;
			if (fb && (fb.debug?.candidates ?? 0) > 0) {
				out = fb;
				usedScope = FALLBACK_SCOPE;
			}
		}

		stats.turns++;
		const cacheState = out.stable.cache ?? "empty";
		if (cacheState === "new") stats.cacheNew++;
		else if (cacheState === "hit") stats.cacheHit++;
		const hashChanged = out.stable.hash !== stats.lastHash;
		stats.lastHash = out.stable.hash;
		stats.lastDebug = out.debug;

		const stableText = out.stable.text ?? "";
		let tailText = out.tail.text ?? "";

		// 插件生效 → 关掉同名技能，避免两套逻辑打架
		const opts = event.systemPromptOptions;
		if (opts?.skills?.length) {
			const n0 = opts.skills.length;
			opts.skills = opts.skills.filter(
				(s: any) => s?.name !== HIDE_SKILL && !String(s?.baseDir ?? "").includes(`/skills/${HIDE_SKILL}`),
			);
			if (opts.skills.length !== n0) skillHidden = true;
		}

		// stable → system sections
		if (opts && stableText) {
			opts.sections = opts.sections ?? {};
			opts.sections.mem = stableText;
			stats.stableBytes = Buffer.byteLength(stableText, "utf8");
		}

		// 钩子 2/3：token / 轮次
		const entries = ctx.sessionManager.getEntries();
		const { turns: prevTurns, chars } = scanSession(entries);
		const turns = prevTurns + 1;
		const tokens = Math.round(chars / CHARS_PER_TOKEN);
		const lh = lastHarvest(entries);
		let harvestReason = "";
		if (harvestEnabled) {
			if (pendingCompact) harvestReason = "压缩前";
			else if (tokens - lh.tokens >= (lh.tokens === 0 ? TOKENS_FIRST : TOKENS_STEP))
				harvestReason = "上下文增长";
			else if (turns - lh.turn >= TURNS_STEP && tokens - lh.tokens >= TURNS_MIN_TOKENS)
				harvestReason = "多轮对话";
		}
		if (harvestReason) {
			const line = harvestLine(turns, Math.round(tokens / 1000), harvestReason);
			tailText = tailText ? tailText + "\n\n" + line : line;
			stats.harvests++;
			pendingCompact = false;
			try {
				ctx.sessionManager.appendCustomEntry("mem_harvest", {
					turn: turns,
					tokens,
					reason: harvestReason,
				});
			} catch {
				/* ignore */
			}
		}

		const willInject = Boolean(tailText) && tailText !== stats.lastTail;
		const tailBytes = willInject ? Buffer.byteLength(tailText, "utf8") : 0;
		if (willInject) {
			stats.tailBytes += tailBytes;
			stats.injected++;
			stats.lastTail = tailText;
			for (const id of out.tail.ids ?? []) {
				if (!injectedIds.includes(id)) injectedIds.push(id);
			}
			if (injectedIds.length > 200) injectedIds = injectedIds.slice(-200);
		}

		await log({
			t: Date.now(),
			event: "turn",
			sid,
			scope: usedScope,
			fbTried,
			turn: stats.turns,
			sessTurns: turns,
			tokens,
			cache: cacheState,
			hash: out.stable.hash,
			hashChanged,
			stableBytes: Buffer.byteLength(stableText, "utf8"),
			tailBytes,
			dyn: out.tail.dynamic,
			anchor: out.tail.anchor,
			harvest: harvestReason || undefined,
			skillHidden,
			candidates: out.debug.candidates,
			best: out.debug.best,
			drift: out.debug.drift,
			elapsedMs: out.debug.elapsedMs,
			dry: DRY,
		});

		if (DRY) {
			ctx.ui.notify(
				`[mem dry] stable ${Buffer.byteLength(stableText, "utf8")}B (${out.stable.cache}) / tail ${tailBytes}B / ${tokens}K / 第${turns}轮`,
				"info",
			);
			return;
		}
		if (willInject) {
			return { message: { customType: "mem", content: tailText, display: false } };
		}
	});

	// 拉取式检索
	pi.registerTool({
		name: "mem_search",
		label: "记忆检索",
		description:
			"检索本地长期记忆（跨会话的经验/决策/事实/偏好）。当用户提到「之前」「上次」「我们不是说过」或需要回忆既往结论时使用。",
		parameters: {
			type: "object",
			properties: {
				query: { type: "string", description: "检索语句，用自然语言" },
				k: { type: "number", description: "返回条数，默认 5" },
			},
			required: ["query"],
		} as any,
		async execute(_id: string, params: any) {
			const k = Math.max(1, Math.min(Number(params.k) || 5, 20));
			const r = await runMem(
				["search", String(params.query), "--scope", scope, "-k", String(k)],
				"/search",
				{ query: String(params.query), scope, k },
			);
			if (!r) {
				return { content: [{ type: "text", text: "记忆库不可用" }], details: {} };
			}
			const items = (r.results ?? []).map(
				(x: any, i: number) =>
					`${i + 1}. [${x.id}] (${x.claim_type}, ${Number(x.score).toFixed(2)}) ${x.content}`,
			);
			return {
				content: [{ type: "text", text: items.length ? items.join("\n") : "没有匹配的记忆" }],
				details: { count: items.length },
			};
		},
	});

	pi.registerCommand("mem", {
		description: "记忆库状态（/mem off 关闭、/mem on 开启）",
		handler: async (args, ctx) => {
			const a = args.trim().toLowerCase();
			if (a === "off") {
				enabled = false;
				ctx.ui.notify("记忆注入已关闭", "warning");
				return;
			}
			if (a === "on") {
				enabled = true;
				ctx.ui.notify("记忆注入已开启", "info");
				return;
			}
			const d = stats.lastDebug;
			const live = await health(400);
			ctx.ui.notify(
				[
					`状态: ${enabled ? "开启" : "关闭"}${DRY ? " (dry)" : ""}${harvestEnabled ? " +harvest" : ""}`,
					`scope: ${scope}`,
					`技能: ${skillHidden ? `已屏蔽 ${HIDE_SKILL}（插件接管）` : "未屏蔽"}`,
					`服务: ${live ? `常驻 (${BASE})` : "未运行（回退子进程）"}  daemon ${stats.viaDaemon} / spawn ${stats.viaSpawn}`,
					`轮次: ${stats.turns}  (stable new ${stats.cacheNew} / hit ${stats.cacheHit})`,
					`stable: ${stats.stableBytes}B  hash ${stats.lastHash ?? "-"}`,
					`tail: 累计 ${stats.tailBytes}B / 注入 ${stats.injected} 次 / harvest ${stats.harvests} 次`,
					d
						? `最近: ${d.candidates} 候选 best=${Number(d.best).toFixed(3)} drift=${d.drift} ${d.elapsedMs}ms`
						: "最近: -",
					`错误: ${stats.errors}`,
				].join("\n"),
				"info",
			);
		},
	});
}
