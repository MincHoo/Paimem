#!/usr/bin/env python3
"""mem-skill 安装脚本：检查环境 → 装依赖 → 下模型 → 自检

用法:
    python setup.py                # 完整安装
    python setup.py --check        # 只检查，不安装
    python setup.py --model-only   # 只下载模型
"""
import argparse
import os
import subprocess
import sys
import urllib.request

MIN_PY = (3, 9)
PKGS = ["numpy", "onnxruntime", "tokenizers"]
MIRROR = "https://hf-mirror.com/Xenova/multilingual-e5-small/resolve/main"
FILES = [
    ("onnx/model_int8.onnx", "onnx/model_int8.onnx", 118_000_000),
    ("tokenizer.json", "tokenizer.json", 17_000_000),
]

HOME = os.environ.get("MEM_HOME", os.path.join(os.path.expanduser("~"), ".mem"))
MODEL_DIR = os.environ.get("MEM_MODEL", os.path.join(HOME, "models", "e5-small-onnx"))


def log(msg):
    print(msg, flush=True)


def check_python():
    if sys.version_info < MIN_PY:
        log(f"✗ Python {MIN_PY[0]}.{MIN_PY[1]}+ 必需，当前 {sys.version.split()[0]}")
        return False
    log(f"✓ Python {sys.version.split()[0]}")
    return True


def check_pkgs(auto_install=True):
    missing = []
    for p in PKGS:
        try:
            __import__(p)
            log(f"✓ {p} 已安装")
        except ImportError:
            missing.append(p)
    if not missing:
        return True
    log(f"✗ 缺少: {', '.join(missing)}")
    if not auto_install:
        return False
    log("→ 正在安装...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", *missing])
    return r.returncode == 0


def download(url, dest, size_hint):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or size_hint)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                pct = done * 100 // max(total, 1)
                sys.stdout.write(f"\r  {os.path.basename(dest)} {pct:3d}% "
                                 f"({done // 2**20}/{total // 2**20} MB)")
                sys.stdout.flush()
        sys.stdout.write("\n")
        os.replace(tmp, dest)
        return True
    except Exception as e:
        sys.stdout.write("\n")
        log(f"✗ 下载失败 {os.path.basename(dest)}: {type(e).__name__} {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def check_model(auto_download=True):
    ok = True
    for _, rel, hint in FILES:
        dest = os.path.join(MODEL_DIR, rel)
        if os.path.isfile(dest) and os.path.getsize(dest) > hint * 0.5:
            log(f"✓ 模型文件 {rel}")
            continue
        ok = False
        log(f"✗ 缺少 {rel}")
        if auto_download:
            url = f"{MIRROR}/{rel}"
            log(f"→ 下载 {url}")
            if not download(url, dest, hint):
                return False
            ok = True
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查")
    ap.add_argument("--model-only", action="store_true", help="只下模型")
    a = ap.parse_args()

    log("=== mem-skill 安装检查 ===")
    log(f"数据目录: {HOME}")
    log(f"模型目录: {MODEL_DIR}")
    log("")

    if not check_python():
        sys.exit(1)

    if a.model_only:
        ok = check_model()
    elif a.check:
        ok = check_pkgs(auto_install=False) and check_model(auto_download=False)
    else:
        ok = check_pkgs() and check_model()

    if not ok:
        log("\n✗ 安装未完成，请按上面的提示处理")
        sys.exit(1)

    log("\n→ 初始化数据库...")
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "mem.py"), "init"])
    if r.returncode != 0:
        log("✗ 初始化失败")
        sys.exit(1)

    log("\n→ 自检...")
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "mem.py"), "selftest"])
    if r.returncode != 0:
        log("✗ 自检失败")
        sys.exit(1)

    log("\n=== 安装完成 ===")
    log("试一下：")
    log('  python mem.py add "测试记忆" --scope demo --origin user --evidence turn:t1')
    log('  python mem.py search "测试" --scope demo')


if __name__ == "__main__":
    main()
