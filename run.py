#!/usr/bin/env python3
"""
BrandGuard Consolidated Pipeline Startup Script — FastAPI Edition
Quick and easy way to start the unified brand compliance analysis system
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


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
        ("slowapi", "slowapi"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("scikit-learn", "sklearn"),
        ("python-multipart", "multipart"),
        ("aiofiles", "aiofiles"),
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


def check_config_files():
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    files_funcs = [
        (config_dir / "color_palette.yaml", _create_default_color_config),
        (config_dir / "typography_rules.yaml", _create_default_typography_config),
        (config_dir / "brand_voice.yaml", _create_default_brand_voice_config),
        (config_dir / "logo_detection.yaml", _create_default_logo_config),
    ]
    for fp, func in files_funcs:
        if not fp.exists():
            func(fp)
    print("✅ Configuration files checked/created")


# ---------------------------------------------------------------------------
# Default YAML generators (unchanged from original run.py)
# ---------------------------------------------------------------------------

def _create_default_color_config(config_path):
    content = """# Default Brand Color Palette Configuration
brand_palette:
  name: "Default"
  primary_colors:
    - name: "Primary Blue"
      hex: "#1E40AF"
      rgb: [30, 64, 175]
      tolerance: 0.1
    - name: "Primary Dark"
      hex: "#1E293B"
      rgb: [30, 41, 59]
      tolerance: 0.1
  secondary_colors:
    - name: "Secondary Gray"
      hex: "#64748B"
      rgb: [100, 116, 139]
      tolerance: 0.1
    - name: "Light Gray"
      hex: "#F1F5F9"
      rgb: [241, 245, 249]
      tolerance: 0.1
  accent_colors:
    - name: "Accent Orange"
      hex: "#F97316"
      rgb: [249, 115, 22]
      tolerance: 0.1
    - name: "Accent Green"
      hex: "#10B981"
      rgb: [16, 185, 129]
      tolerance: 0.1
"""
    with open(config_path, "w") as f:
        f.write(content)


def _create_default_typography_config(config_path):
    content = """# Default Typography Rules Configuration
typography_rules:
  approved_fonts:
    - "Arial"
    - "Helvetica"
    - "Times New Roman"
    - "Georgia"
  max_font_size: 72
  min_font_size: 8
  preferred_font_families:
    - "sans-serif"
    - "serif"
  forbidden_fonts: []
  line_height_ratio: 1.5
  letter_spacing: 0.0
"""
    with open(config_path, "w") as f:
        f.write(content)


def _create_default_brand_voice_config(config_path):
    content = """# Default Brand Voice Configuration
brand_voice:
  formality_score: 60
  confidence_level: "balanced"
  warmth_score: 50
  energy_score: 50
  readability_level: "grade8"
  persona_type: "general"
  allow_emojis: false
  allow_slang: false
  no_financial_guarantees: true
  no_medical_claims: true
  no_competitor_bashing: true
"""
    with open(config_path, "w") as f:
        f.write(content)


def _create_default_logo_config(config_path):
    content = """# Default Logo Detection Configuration
logo_detection:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  max_detections: 100
  min_logo_size: 20
  max_logo_size: 500
  placement_rules: {}
"""
    with open(config_path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Server starter — uvicorn (replaces Flask app.run)
# ---------------------------------------------------------------------------

def start_server(host: str = "0.0.0.0", port: int = 5001, reload: bool = False):
    """Launch uvicorn with the FastAPI app."""
    print(f"🚀 Starting BrandGuard FastAPI on {host}:{port}")
    print(f"🌐 Web UI:    http://{host}:{port}")
    print(f"📡 API docs:  http://{host}:{port}/docs")
    print(f"🔍 Health:    http://{host}:{port}/api/health")
    print("=" * 50)

    cmd = ["uvicorn", "app:app", "--host", host, "--port", str(port)]
    if reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BrandGuard Consolidated Pipeline — FastAPI Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                     # Start on 0.0.0.0:5001
  python run.py --host 127.0.0.1   # Localhost only
  python run.py --port 8080        # Custom port
  python run.py --reload           # Auto-reload on code changes (dev)
  python run.py --skip-checks      # Skip pre-flight checks
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind (default: 5001)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency and config checks")
    args = parser.parse_args()

    print("🎨 BrandGuard Consolidated Pipeline")
    print("=" * 50)

    if not args.skip_checks:
        print("\n🔍 Running pre-flight checks...")
        check_python_version()
        if not check_dependencies():
            sys.exit(1)

        print("\n📁 Setting up directories...")
        create_directories()

        print("\n⚙️  Checking configuration...")
        check_config_files()

        print("\n✅ All checks passed!")

    print("\n🚀 Starting server...")
    start_server(args.host, args.port, args.reload)


if __name__ == "__main__":
    main()
