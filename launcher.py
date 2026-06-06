from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def resource_path(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


def user_workspace() -> Path:
    workspace = Path.home() / "BatchIsoFixer"
    for folder in (
        workspace / "input",
        workspace / "input" / "raw",
        workspace / "output",
        workspace / "output" / "fixed",
        workspace / "output" / "modified",
    ):
        folder.mkdir(parents=True, exist_ok=True)
    return workspace


def wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def open_when_ready(url: str, host: str, port: int) -> None:
    if wait_for_port(host, port):
        webbrowser.open(url)


def main() -> None:
    host = "127.0.0.1"
    port = int(os.environ.get("BATCH_ISO_FIXER_PORT", "8501"))
    app_path = resource_path("app.py")
    workspace = user_workspace()
    os.chdir(workspace)

    url = f"http://{host}:{port}"
    threading.Thread(target=open_when_ready, args=(url, host, port), daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--global.developmentMode",
        "false",
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
