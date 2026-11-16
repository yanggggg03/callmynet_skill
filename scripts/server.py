"""Local HTTP panel and model API; no cloud service or LLM API dependency."""
import argparse
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from core import ROOT, MODEL_IDS, analyze_waveforms, export_bundle, file_rows, load_config, run_models, status, validate_records
from demo import make_demo
from excel_writer import write_workbook
import io

MODEL_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "CallMyNet/1.0"

    def origin_allowed(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True
        expected = f"http://{self.headers.get('Host', '')}"
        return origin == expected or origin in load_config(self.server.config_path).get("allowed_origins", [])

    def send(self, code, content, kind="application/json; charset=utf-8", filename=None):
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        if self.headers.get("Origin") and self.origin_allowed():
            self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        self.send(204 if self.origin_allowed() else 403, b"")

    def do_GET(self):
        try:
            path = urlparse(self.path).path
            if path == "/api/health":
                self.send(200, status(load_config(self.server.config_path)))
            elif path == "/api/demo":
                self.send(200, make_demo())
            elif path.startswith("/api/templates/"):
                name = path.rsplit("/", 1)[-1]
                if name == "input.xlsx":
                    columns = ("mag", "deg")
                elif name == "waveforms.xlsx":
                    columns = ("run_id", "sample_id", "scenario_id", "model_id", "time_s", "power_w", "target_w", "source")
                else:
                    return self.send(404, {"error": "Unknown template"})
                memory = io.BytesIO()
                write_workbook(memory, [("input", columns, [])])
                self.send(200, memory.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", name)
            elif path in ("/", "/index.html"):
                self.send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path.startswith("/assets/"):
                destination = (ROOT / path.lstrip("/")).resolve()
                if not destination.is_relative_to(ROOT / "assets") or not destination.is_file():
                    return self.send(404, {"error": "Not found"})
                self.send(200, destination.read_bytes(), mimetypes.guess_type(destination)[0] or "application/octet-stream")
            else:
                self.send(404, {"error": "Not found"})
        except (ValueError, OSError, KeyError) as exc:
            self.send(400, {"error": str(exc)})

    def do_POST(self):
        try:
            if not self.origin_allowed():
                return self.send(403, {"error": "来源未允许，请在config/models.json配置allowed_origins"})
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 24_000_000:
                return self.send(413, {"error": "请求大小须在24MB以内"})
            data = json.loads(self.rfile.read(size).decode("utf-8"))
            path = urlparse(self.path).path
            if path == "/api/predict":
                if not MODEL_LOCK.acquire(blocking=False):
                    return self.send(409, {"error": "模型正在处理另一项请求，请稍后重试"})
                try:
                    bundle = run_models(load_config(self.server.config_path), data)
                finally:
                    MODEL_LOCK.release()
                self.send(200, bundle)
            elif path == "/api/read-input":
                self.send(200, {"records": validate_records(file_rows(data), data.get("magnitude_unit", "linear"))})
            elif path == "/api/analyze-waveforms":
                if "content_base64" in data:
                    data["rows"] = file_rows(data)
                self.send(200, analyze_waveforms(data))
            elif path == "/api/export":
                self.send(200, export_bundle(data), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "matching_results.xlsx")
            else:
                self.send(404, {"error": "Not found"})
        except Exception as exc:
            self.send(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description="三模型调节比较面板")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--config", default=str(ROOT / "config/models.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    service = ThreadingHTTPServer((args.host, args.port), Handler)
    service.config_path = args.config
    print(f"Panel: http://{args.host}:{args.port}", flush=True)
    print(json.dumps(status(config), ensure_ascii=False), flush=True)
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.server_close()


if __name__ == "__main__":
    main()
