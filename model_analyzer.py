#!/usr/bin/env python3
"""
BrandGuard Model Analyzer
Analyzes all models, libraries, and their sizes in the project
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import psutil
import platform

def get_file_size(path):
    """Get file size in human readable format"""
    try:
        size_bytes = os.path.getsize(path)
        return format_size(size_bytes)
    except:
        return "Unknown"

def format_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def get_directory_size(path):
    """Get total size of directory"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
    except:
        pass
    return total_size

def analyze_huggingface_models():
    """Analyze Hugging Face cached models"""
    hf_cache_dir = Path.home() / ".cache" / "huggingface"
    models_info = []
    
    if not hf_cache_dir.exists():
        return models_info, 0
    
    hub_dir = hf_cache_dir / "hub"
    if not hub_dir.exists():
        return models_info, 0
    
    total_size = 0
    
    for model_dir in hub_dir.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("models--"):
            model_name = model_dir.name.replace("models--", "").replace("--", "/")
            size = get_directory_size(model_dir)
            total_size += size
            
            # Try to get more info about the model
            config_file = None
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file == "config.json":
                        config_file = os.path.join(root, file)
                        break
                if config_file:
                    break
            
            model_info = {
                "name": model_name,
                "size": format_size(size),
                "size_bytes": size,
                "path": str(model_dir),
                "config_file": config_file
            }
            
            # Try to read config for more details
            if config_file and os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        model_info["model_type"] = config.get("model_type", "unknown")
                        model_info["architecture"] = config.get("architectures", ["unknown"])[0] if config.get("architectures") else "unknown"
                except:
                    pass
            
            models_info.append(model_info)
    
    # Sort by size (largest first)
    models_info.sort(key=lambda x: x["size_bytes"], reverse=True)
    return models_info, total_size

def analyze_local_models():
    """Analyze local model files"""
    model_extensions = ['.pt', '.pth', '.onnx', '.pdmodel', '.pdiparams', '.safetensors', '.bin']
    models_info = []
    
    current_dir = Path.cwd()
    
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if any(file.endswith(ext) for ext in model_extensions):
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                
                model_info = {
                    "name": file,
                    "size": format_size(size),
                    "size_bytes": size,
                    "path": file_path,
                    "type": "local"
                }
                
                # Try to identify model type
                if file.endswith('.pt') or file.endswith('.pth'):
                    model_info["type"] = "PyTorch"
                elif file.endswith('.pdmodel'):
                    model_info["type"] = "PaddlePaddle"
                elif file.endswith('.onnx'):
                    model_info["type"] = "ONNX"
                elif file.endswith('.safetensors'):
                    model_info["type"] = "SafeTensors"
                
                models_info.append(model_info)
    
    # Sort by size (largest first)
    models_info.sort(key=lambda x: x["size_bytes"], reverse=True)
    return models_info

def analyze_installed_libraries():
    """Analyze installed Python libraries"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            packages = json.loads(result.stdout)
            
            # Filter ML-related packages
            ml_keywords = ['torch', 'tensorflow', 'transformers', 'ultralytics', 'paddle', 
                          'opencv', 'numpy', 'pandas', 'scikit', 'fastapi', 'flask', 'vllm']
            
            ml_packages = []
            for package in packages:
                if any(keyword in package['name'].lower() for keyword in ml_keywords):
                    ml_packages.append({
                        "name": package['name'],
                        "version": package['version']
                    })
            
            return ml_packages
    except:
        pass
    return []

def get_system_info():
    """Get system information"""
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "total_memory": format_size(psutil.virtual_memory().total),
        "available_memory": format_size(psutil.virtual_memory().available),
        "disk_usage": format_size(psutil.disk_usage('/').total)
    }

def main():
    print("🔍 BrandGuard Model Analyzer")
    print("=" * 50)
    
    # System Info
    print("\n🖥️  SYSTEM INFORMATION")
    print("-" * 30)
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Analyze Hugging Face Models
    print("\n🤗 HUGGING FACE MODELS")
    print("-" * 30)
    hf_models, hf_total = analyze_huggingface_models()
    
    if hf_models:
        print(f"Total Hugging Face Models: {len(hf_models)}")
        print(f"Total Size: {format_size(hf_total)}")
        print("\nLargest Models:")
        for i, model in enumerate(hf_models[:10]):  # Show top 10
            print(f"{i+1:2d}. {model['name']:<40} {model['size']:>8}")
            if 'model_type' in model:
                print(f"     Type: {model.get('model_type', 'unknown')} | Arch: {model.get('architecture', 'unknown')}")
    else:
        print("No Hugging Face models found")
    
    # Analyze Local Models
    print("\n📁 LOCAL MODEL FILES")
    print("-" * 30)
    local_models = analyze_local_models()
    
    if local_models:
        local_total = sum(model["size_bytes"] for model in local_models)
        print(f"Total Local Models: {len(local_models)}")
        print(f"Total Size: {format_size(local_total)}")
        print("\nAll Local Models:")
        for i, model in enumerate(local_models):
            rel_path = os.path.relpath(model["path"])
            print(f"{i+1:2d}. {model['name']:<30} {model['size']:>8} ({model['type']})")
            print(f"     Path: {rel_path}")
    else:
        print("No local model files found")
        local_total = 0
    
    # Analyze Libraries
    print("\n📚 INSTALLED ML LIBRARIES")
    print("-" * 30)
    libraries = analyze_installed_libraries()
    
    if libraries:
        print(f"Total ML Libraries: {len(libraries)}")
        for lib in libraries:
            print(f"• {lib['name']:<25} v{lib['version']}")
    else:
        print("Could not retrieve library information")
    
    # Summary
    print("\n📊 SUMMARY")
    print("-" * 30)
    total_models_size = hf_total + local_total
    
    print(f"Hugging Face Models: {format_size(hf_total)}")
    print(f"Local Models:       {format_size(local_total)}")
    print(f"Total Models:       {format_size(total_models_size)}")
    print(f"Total Models Count: {len(hf_models) + len(local_models)}")
    
    # Free Tier Analysis
    print("\n🆓 FREE TIER COMPATIBILITY")
    print("-" * 30)
    
    # AWS Free Tier limits
    aws_storage_limit = 30 * 1024 * 1024 * 1024  # 30 GB
    aws_ram_limit = 1 * 1024 * 1024 * 1024  # 1 GB
    
    # Hugging Face Spaces limits
    hf_storage_limit = 50 * 1024 * 1024 * 1024  # 50 GB
    hf_ram_limit = 16 * 1024 * 1024 * 1024  # 16 GB
    
    print("AWS Free Tier (t2.micro):")
    print(f"  Storage Limit: 30 GB")
    print(f"  RAM Limit: 1 GB")
    print(f"  Models Fit: {'✅ YES' if total_models_size < aws_storage_limit else '❌ NO'}")
    print(f"  Size Used: {format_size(total_models_size)} / 30 GB")
    
    print("\nHugging Face Spaces Free:")
    print(f"  Storage Limit: 50 GB")
    print(f"  RAM Limit: 16 GB")
    print(f"  Models Fit: {'✅ YES' if total_models_size < hf_storage_limit else '❌ NO'}")
    print(f"  Size Used: {format_size(total_models_size)} / 50 GB")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 30)
    
    if total_models_size > aws_storage_limit:
        print("❌ Your models are too large for AWS Free Tier")
        print("✅ Consider Hugging Face Spaces for free deployment")
        
        # Find large models to potentially remove
        large_models = [m for m in hf_models if m["size_bytes"] > 1024 * 1024 * 1024]  # > 1GB
        if large_models:
            print("\nLarge models you could remove to fit free tiers:")
            for model in large_models:
                print(f"  • {model['name']} ({model['size']})")
    
    if total_models_size < hf_storage_limit:
        print("✅ Your models fit within Hugging Face Spaces free tier")
        print("✅ Ready for free deployment!")
    
    # Save detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_info": sys_info,
        "huggingface_models": hf_models,
        "local_models": local_models,
        "libraries": libraries,
        "totals": {
            "hf_models_count": len(hf_models),
            "local_models_count": len(local_models),
            "hf_models_size_bytes": hf_total,
            "local_models_size_bytes": local_total,
            "total_size_bytes": total_models_size
        }
    }
    
    report_file = "model_analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()
