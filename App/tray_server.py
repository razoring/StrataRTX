import os
import sys
import time
import json
import webbrowser
import subprocess
import threading
import urllib.request
from pathlib import Path
import pystray
from PIL import Image, ImageDraw

APP_DIR = Path(__file__).resolve().parent
VENV_PYTHON = APP_DIR / ".venv" / "Scripts" / "python.exe"
SERVER_SCRIPT = APP_DIR / "serve" / "server.py"
CONFIG_FILE = APP_DIR / "strata-coder-iq1_m.json"
PORT = 8080

def create_tray_icon():
    ico_path = APP_DIR / "strata.ico"
    if ico_path.exists():
        try:
            return Image.open(str(ico_path))
        except Exception:
            pass
    # 64x64 icon with modern gradient circle and stylized 'S'
    img = Image.new("RGBA", (64, 64), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Background circle (deep indigo / cyan accent)
    draw.ellipse((4, 4, 60, 60), fill=(24, 28, 42, 255), outline=(56, 189, 248, 255), width=3)
    # Inner glowing dot / core
    draw.ellipse((20, 20, 44, 44), fill=(56, 189, 248, 255))
    draw.ellipse((26, 26, 38, 38), fill=(255, 255, 255, 255))
    return img

class StrataTrayApp:
    def __init__(self):
        self.server_proc = None
        self.running = True

    def start_server(self):
        env = os.environ.copy()
        env["STRATA_ARENA_PIN_GIB"] = "8"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["STRATA_STAGE_TRIM"] = "1"
        env["CUDA_P2P_DISABLE"] = "1"

        cmd = [
            str(VENV_PYTHON),
            str(SERVER_SCRIPT),
            "--engine", "strata",
            "--config", str(CONFIG_FILE),
            "--port", str(PORT)
        ]

        # Windows CREATE_NO_WINDOW flag
        creationflags = 0x08000000

        self.server_proc = subprocess.Popen(
            cmd,
            cwd=str(APP_DIR),
            env=env,
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def stop_server(self):
        # 1. Attempt graceful shutdown via HTTP endpoint
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/shutdown",
                data=b"{}",
                headers={"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{PORT}"}
            )
            with urllib.request.urlopen(req, timeout=1.5):
                pass
        except Exception:
            pass

        # 2. Terminate server_proc if tracked
        if self.server_proc and self.server_proc.poll() is None:
            try:
                subprocess.run(["taskkill", "/F", "/PID", str(self.server_proc.pid), "/T"],
                               capture_output=True, creationflags=0x08000000)
            except Exception:
                pass
            self.server_proc = None

        # 3. Terminate any orphan strata.exe or server.py python instances
        subprocess.run(["taskkill", "/F", "/IM", "strata.exe", "/T"],
                       capture_output=True, creationflags=0x08000000)

    def open_webui(self, icon=None, item=None):
        webbrowser.open(f"http://127.0.0.1:{PORT}/")

    def open_logs(self, icon=None, item=None):
        log_path = APP_DIR / "strata-coder-iq1_m.log"
        if log_path.exists():
            os.startfile(str(log_path))

    def restart_server(self, icon=None, item=None):
        def _do_restart():
            # Try warm live restart endpoint first (instant, keeps port open)
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/restart",
                    data=b"{}",
                    headers={"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{PORT}"}
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass

            # Fallback to cold full restart of the server process
            self.stop_server()
            time.sleep(2)
            self.start_server()

        threading.Thread(target=_do_restart, daemon=True).start()

    def on_exit(self, icon, item):
        self.running = False
        try:
            self.stop_server()
        finally:
            icon.stop()
            os._exit(0)

    def run(self):
        self.start_server()

        menu = pystray.Menu(
            pystray.MenuItem("Open WebUI (http://127.0.0.1:8080)", self.open_webui, default=True),
            pystray.MenuItem("Restart Engine", self.restart_server),
            pystray.MenuItem("View Log", self.open_logs),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit Strata", self.on_exit)
        )

        icon = pystray.Icon(
            name="StrataServer",
            icon=create_tray_icon(),
            title="Strata AI Server (Dual RTX 3060)",
            menu=menu
        )

        icon.run()

if __name__ == "__main__":
    app = StrataTrayApp()
    try:
        app.run()
    except KeyboardInterrupt:
        app.stop_server()
