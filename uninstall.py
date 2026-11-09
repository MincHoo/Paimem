#!/usr/bin/env python3
"""ΠMem 卸载助手

用法:
    python uninstall.py            # 只停服务 + 显示移除命令
    python uninstall.py --purge    # 同时删除 ~/.mem（记忆 + 模型，约 130MB）
    python uninstall.py --purge --yes   # 不询问

注意：pi 没有卸载钩子，所以删包要手动执行 `pi remove <包路径>`。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request

HOME = os.environ.get("MEM_HOME", os.path.join(os.path.expanduser("~"), ".mem"))
PORT = int(os.environ.get("MEM_PORT", 7788))


def health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def stop_daemon():
    h = health()
    if not h:
        print(f"✓ 没有在跑的服务（127.0.0.1:{PORT}）")
        return
    pid = h.get("pid")
    print(f"→ 停掉服务 pid={pid}")
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, 15)
        print("✓ 服务已停止")
    except Exception as e:
        print(f"✗ 停止失败（可能已退出）: {e}")


def purge(yes=False):
    if not os.path.isdir(HOME):
        print(f"✓ 数据目录不存在: {HOME}")
        return
    size = 0
    for root, _, files in os.walk(HOME):
        for f in files:
            try:
                size += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    print(f"→ 将删除 {HOME}（{size / 2**20:.0f} MB）")
    print("  包含：记忆数据库、模型、配置、日志")
    if not yes:
        ans = input("  确认删除？输入 yes: ").strip().lower()
        if ans != "yes":
            print("已取消")
            return
    shutil.rmtree(HOME, ignore_errors=True)
    print("✓ 数据已删除")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge", action="store_true", help="删除 ~/.mem")
    ap.add_argument("--yes", action="store_true", help="不再询问")
    a = ap.parse_args()

    print("=== ΠMem 卸载 ===\n")
    stop_daemon()
    if a.purge:
        purge(a.yes)

    print("\n最后一步（手动）：")
    print('  pi remove "<安装时使用的路径>"')
    print("\n检查残留：")
    print("  pi list")
    print("  curl -s http://127.0.0.1:%d/health   # 应连不上" % PORT)


if __name__ == "__main__":
    main()
