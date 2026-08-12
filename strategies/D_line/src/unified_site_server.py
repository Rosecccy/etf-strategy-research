from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "D" / "src" / "build_unified_site.py"
SITE_INDEX = ROOT / "D" / "site" / "index.html"
HOST = "127.0.0.1"
PORT = 8765


def rebuild() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, str(BUILDER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    message = (result.stdout or result.stderr).strip()
    return result.returncode == 0, message


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self.send_response(302)
            self.send_header("Location", "/D/site/live")
            self.end_headers()
            return
        if route in {"/D/site/live", "/D/site/live/"}:
            body = SITE_INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if route == "/api/rebuild":
            ok, message = rebuild()
            body = json.dumps({"ok": ok, "message": message}, ensure_ascii=False).encode("utf-8")
            self.send_response(200 if ok else 500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self) -> None:
        if urlparse(self.path).path.startswith("/D/site/"):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        print(f"[dashboard] {self.address_string()} {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动C/S/D三线策略候选看板")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok, message = rebuild()
    if not ok:
        raise SystemExit(f"看板数据生成失败：{message}")

    handler = partial(DashboardHandler, directory=str(ROOT))
    server = ThreadingHTTPServer((HOST, PORT), handler)
    url = f"http://{HOST}:{PORT}/D/site/live"
    print("=" * 60)
    print("C / S / D 三线策略候选看板")
    print(url)
    print("关闭本窗口即可停止网站。")
    print("=" * 60)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
