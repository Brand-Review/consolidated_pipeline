#!/usr/bin/env python3
"""
BrandGuard Consolidated Pipeline Startup Script
Quick and easy way to start the unified brand compliance analysis system
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        ('flask', 'flask'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('pillow', 'PIL'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('scikit-learn', 'sklearn')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing dependencies: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['configs', 'uploads', 'results', 'models']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def check_config_files():
    """Check if configuration files exist, create defaults if not"""
    config_dir = Path('configs')
    
    if not config_dir.exists():
        config_dir.mkdir(exist_ok=True)
    
    # Create default color palette config if it doesn't exist
    color_config = config_dir / 'color_palette.yaml'
    if not color_config.exists():
        create_default_color_config(color_config)
    
    # Create default typography config if it doesn't exist
    typography_config = config_dir / 'typography_rules.yaml'
    if not typography_config.exists():
        create_default_typography_config(typography_config)
    
    # Create default brand voice config if it doesn't exist
    voice_config = config_dir / 'brand_voice.yaml'
    if not voice_config.exists():
        create_default_brand_voice_config(voice_config)
    
    # Create default logo detection config if it doesn't exist
    logo_config = config_dir / 'logo_detection.yaml'
    if not logo_config.exists():
        create_default_logo_config(logo_config)
    
    print("✅ Configuration files checked/created")

def create_default_color_config(config_path):
    """Create default color palette configuration"""
    config_content = """# Default Brand Color Palette Configuration
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
    with open(config_path, 'w') as f:
        f.write(config_content)

def create_default_typography_config(config_path):
    """Create default typography configuration"""
    config_content = """# Default Typography Rules Configuration
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
    with open(config_path, 'w') as f:
        f.write(config_content)

def create_default_brand_voice_config(config_path):
    """Create default brand voice configuration"""
    config_content = """# Default Brand Voice Configuration
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
    with open(config_path, 'w') as f:
        f.write(config_content)

def create_default_logo_config(config_path):
    """Create default logo detection configuration"""
    config_content = """# Default Logo Detection Configuration
logo_detection:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  max_detections: 100
  min_logo_size: 20
  max_logo_size: 500
  placement_rules: {}
"""
    with open(config_path, 'w') as f:
        f.write(config_content)

def start_server(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask server"""
    print(f"🚀 Starting BrandGuard Pipeline on {host}:{port}")
    print(f"🌐 Web interface: http://{host}:{port}")
    print(f"📡 API base: http://{host}:{port}/api")
    print(f"🔍 Health check: http://{host}:{port}/api/health")
    print("\n" + "="*50)
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development' if debug else 'production'
    
    # Start the server
    try:
        from app import app
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=debug
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='BrandGuard Consolidated Pipeline Startup Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Start with default settings
  python run.py --host 127.0.0.1  # Start on localhost only
  python run.py --port 8080       # Start on port 8080
  python run.py --debug           # Start in debug mode
        """
    )
    
    parser.add_argument(
        '--host', 
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--skip-checks', 
        action='store_true',
        help='Skip dependency and configuration checks'
    )
    
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
        
        print("\n⚙️ Checking configuration...")
        check_config_files()
        
        print("\n✅ All checks passed!")
    
    print("\n🚀 Starting server...")
    start_server(args.host, args.port, args.debug)

if __name__ == '__main__':
    main()
