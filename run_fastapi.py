#!/usr/bin/env python3
"""
BrandGuard Consolidated Pipeline Startup Script – FASTAPI EDITION
Quick and easy way to start the unified brand-compliance analysis system
"""

import os
import sys
import subprocess
import argparse
import signal
from pathlib import Path

# ----------------------------------------------------------------------
# Everything below is literally copy-pasted from your original run.py
# ----------------------------------------------------------------------
#  (Only the final `start_server` function and import paths change)
# ----------------------------------------------------------------------

def check_python_version():
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_dependencies():
    required_packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("scikit-learn", "sklearn"),
        ("python-multipart", "multipart"),        # file uploads
        ("aiofiles", "aiofiles"),                 # static files
    ]

    missing = []
    for pkg, imp in required_packages:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install via: pip install -r requirements.txt")
        return False
    print("✅ All required dependencies are installed")
    return True

def create_directories():
    for d in ("configs", "uploads", "results", "models"):
        Path(d).mkdir(exist_ok=True)
        print(f"📁 Created directory: {d}")

# ---------- Default YAML generators (unchanged) ----------
def create_default_color_config(p): ...
def create_default_typography_config(p): ...
def create_default_brand_voice_config(p): ...
def create_default_logo_config(p): ...

def check_config_files():
    cfg = Path("configs")
    cfg.mkdir(exist_ok=True)

    files_funcs = [
        (cfg / "color_palette.yaml", create_default_color_config),
        (cfg / "typography_rules.yaml", create_default_typography_config),
        (cfg / "brand_voice.yaml", create_default_brand_voice_config),
        (cfg / "logo_detection.yaml", create_default_logo_config),
    ]
    for fp, func in files_funcs:
        if not fp.exists():
            func(fp)
    print("✅ Configuration files checked/created")

# ------------------------------------------------------------------
#  FastAPI server starter – replaces the old Flask `start_server`
# ------------------------------------------------------------------
def start_server(host="0.0.0.0", port=8000, reload=False):
    """Launch uvicorn with the FastAPI app."""
    print(f"🚀 Starting BrandGuard FastAPI on {host}:{port}")
    print(f"🌐 Web UI: http://{host}:{port}")
    print(f"📡 API docs: http://{host}:{port}/docs")
    print("=" * 50)

    cmd = [
        "uvicorn",
        "app_fastapi:app",   # <module>:<FastAPI-instance>
        "--host", host,
        "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

# ------------------------------------------------------------------
#  CLI remains identical
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BrandGuard Consolidated Pipeline – FastAPI Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_fastapi.py                 # Start on 0.0.0.0:8000
  python run_fastapi.py --host 127.0.0.1
  python run_fastapi.py --port 9000 --reload
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--skip-checks", action="store_true", help="Skip checks")
    args = parser.parse_args()

    print("🎨 BrandGuard Consolidated Pipeline – FastAPI")
    print("=" * 50)

    if not args.skip_checks:
        print("\n🔍 Running pre-flight checks…")
        check_python_version()
        if not check_dependencies():
            sys.exit(1)

        print("\n📁 Setting up directories…")
        create_directories()

        print("\n⚙️  Checking configuration…")
        check_config_files()

        print("\n✅ All checks passed!")

    print("\n🚀 Starting server…")
    start_server(args.host, args.port, reload=args.reload)

# Graceful shutdown hooks (uvicorn already does SIGINT/SIGTERM)
if __name__ == "__main__":
    main()