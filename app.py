"""Local dev server for the tabletop-perception app.

The real app is `public/index.html` — a static site that runs the three models
entirely in the browser via ONNX Runtime Web and Transformers.js + WebGPU.
This file serves `public/` at `http://localhost:8088/` for local testing.

Production is deployed via GitHub Pages at
https://neevs.io/aipi540-tabletop-perception/.

Usage:
    python app.py
    # then open http://localhost:8088/
"""
from __future__ import annotations

import http.server
import os
import socketserver
import sys
from pathlib import Path

PORT = int(os.environ.get("PORT", 8088))
APP_DIR = Path(__file__).resolve().parent / "public"


def main() -> int:
    if not APP_DIR.exists():
        print(f"error: {APP_DIR} does not exist. Run `python setup.py` first.", file=sys.stderr)
        return 1
    os.chdir(APP_DIR)
    with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"Serving tabletop-perception app at http://localhost:{PORT}/")
        print("Chrome 113+ required for WebGPU (classifier + VLM inference).")
        print("Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
