#!/usr/bin/env python3
"""
Auto Remove Unused Models Script
Automatically removes Llava and BERT models that are no longer needed
"""

import os
import shutil
from pathlib import Path

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

def remove_unused_models():
    """Remove Llava and BERT models"""
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    # Models to remove
    models_to_remove = [
        "models--llava-hf--llava-1.5-7b-hf",  # Llava model
        "models--facebook--bart-large-mnli",   # BART model (BERT-based)
        "models--s-nlp--xlmr_formality_classifier",  # XLM-R model (BERT-based)
    ]
    
    print("🗑️  Removing Unused Models")
    print("=" * 40)
    
    total_freed = 0
    removed_count = 0
    
    for model_dir_name in models_to_remove:
        model_path = hf_cache_dir / model_dir_name
        
        if model_path.exists():
            # Get size before removal
            size = get_directory_size(model_path)
            size_formatted = format_size(size)
            
            print(f"\nRemoving: {model_dir_name}")
            print(f"Size: {size_formatted}")
            print(f"Path: {model_path}")
            
            try:
                # Remove the directory
                shutil.rmtree(model_path)
                print(f"✅ Successfully removed {size_formatted}")
                total_freed += size
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove: {e}")
        else:
            print(f"\n⚠️  Model not found: {model_dir_name}")
    
    print(f"\n📊 SUMMARY")
    print("-" * 30)
    print(f"Models removed: {removed_count}")
    print(f"Total space freed: {format_size(total_freed)}")
    
    return total_freed

def main():
    print("🔍 BrandGuard Model Cleanup")
    print("=" * 40)
    print("Removing the following unused models:")
    print("• Llava-1.5-7B-HF (26.3 GB)")
    print("• BART-Large-MNLI (3.0 GB)")
    print("• XLM-R Formality Classifier (2.1 GB)")
    print("\nProceeding with automatic removal...")
    
    freed_space = remove_unused_models()
    print(f"\n🎉 Cleanup completed!")
    print(f"Space freed: {format_size(freed_space)}")
    print(f"\nYour models should now fit in AWS Free Tier (30 GB limit)!")
    
    # Run the analyzer again to show updated stats
    print(f"\n🔍 Running updated analysis...")
    os.system("python model_analyzer.py")

if __name__ == "__main__":
    main()
