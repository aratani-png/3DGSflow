"""3DGS Workflow ローカルサーバー.

起動: python workflow_server.py
ブラウザ: http://localhost:8050
"""

import http.server
import json
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PORT = 8050
HTML_PATH = Path(__file__).parent / "workflow.html"


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PATH.read_bytes())
        elif parsed.path == "/api/ls":
            self._handle_ls(parsed)
        else:
            # favicon等は空で返す
            self.send_response(204)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        body = self._read_body()

        if parsed.path == "/api/mkdir":
            self._handle_mkdir(body)
        elif parsed.path == "/api/rmdir":
            self._handle_rmdir(body)
        else:
            self.send_error(404)

    # ---------- API handlers ----------

    def _handle_ls(self, parsed):
        """指定パスのフォルダ一覧を返す."""
        qs = parse_qs(parsed.query)
        target = qs.get("path", [""])[0]
        if not target:
            self._json_response({"error": "path required"}, 400)
            return
        p = Path(target)
        if not p.exists():
            self._json_response({"exists": False, "children": []})
            return
        children = []
        try:
            for item in sorted(p.iterdir()):
                if item.is_dir():
                    children.append({"name": item.name, "type": "dir"})
        except PermissionError:
            pass
        self._json_response({"exists": True, "path": str(p), "children": children})

    def _handle_mkdir(self, body):
        """フォルダを作成する."""
        paths = body.get("paths", [])
        if isinstance(paths, str):
            paths = [paths]
        created = []
        errors = []
        for p_str in paths:
            p = Path(p_str)
            try:
                p.mkdir(parents=True, exist_ok=True)
                created.append(str(p))
            except Exception as e:
                errors.append({"path": p_str, "error": str(e)})
        self._json_response({"created": created, "errors": errors})

    def _handle_rmdir(self, body):
        """空フォルダを削除する (安全のため空のみ)."""
        path = body.get("path", "")
        if not path:
            self._json_response({"error": "path required"}, 400)
            return
        p = Path(path)
        if not p.exists():
            self._json_response({"error": "not found"}, 404)
            return
        if not p.is_dir():
            self._json_response({"error": "not a directory"}, 400)
            return
        try:
            # 中身がある場合は削除しない
            if any(p.iterdir()):
                self._json_response({"error": "directory not empty"}, 400)
                return
            p.rmdir()
            self._json_response({"removed": str(p)})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ---------- Helpers ----------

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, fmt, *args):
        # API呼び出しだけログ表示
        msg = fmt % args if args else fmt
        if "/api/" in str(msg):
            print(f"  API: {msg}")


def main():
    print(f"3DGS Workflow Server")
    print(f"  http://localhost:{PORT}")
    print(f"  Ctrl+C で停止")
    print()
    server = http.server.HTTPServer(("", PORT), Handler)
    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました。")


if __name__ == "__main__":
    main()
