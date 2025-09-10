#!/usr/bin/env python3
"""
Safe Server Startup Script
Prevents semaphore leaks and multiprocessing issues
"""

import os
import sys
import multiprocessing
import atexit
import signal
import logging

# Set environment variables BEFORE importing any ML libraries
def setup_environment():
    """Set up environment variables to prevent multiprocessing issues"""
    print("🔧 Setting up environment for safe server startup...")
    
    # Disable tokenizers parallelism to prevent semaphore leaks
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Configure PyTorch for single-threaded operation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Disable multiprocessing for various libraries
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU usage
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Set multiprocessing start method to prevent issues
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, that's fine
        pass
    
    print("✅ Environment configured for single-threaded operation")

def cleanup_resources():
    """Clean up all resources to prevent semaphore leaks"""
    try:
        print("🧹 Cleaning up resources...")
        
        # Clean up any active multiprocessing processes
        try:
            active_children = multiprocessing.active_children()
            for process in active_children:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
        except Exception as e:
            print(f"⚠️ Warning: Error cleaning up processes: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("✅ Resource cleanup completed")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n🛑 Received signal {sig}, shutting down gracefully...")
    cleanup_resources()
    sys.exit(0)

def main():
    """Main startup function"""
    print("🚀 Starting BrandGuard Pipeline API (Safe Mode)")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Register cleanup functions
    atexit.register(cleanup_resources)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Import and run the app
        print("📦 Importing application modules...")
        from app import app, pipeline
        
        print("🔧 Initializing pipeline...")
        if pipeline:
            print("✅ Pipeline initialized successfully")
        else:
            print("⚠️ Pipeline initialization failed, but continuing...")
        
        print("🌐 Starting Flask server...")
        print("📍 Server will be available at: http://localhost:5003")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run the Flask app with safe settings
        app.run(
            debug=False,  # Disable debug mode
            host='0.0.0.0',
            port=5003,
            threaded=True,
            use_reloader=False,  # Disable reloader to prevent multiprocessing issues
            processes=1  # Single process only
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        cleanup_resources()
    except Exception as e:
        print(f"❌ Server crashed: {e}")
        import traceback
        print(f"📋 Full traceback: {traceback.format_exc()}")
        cleanup_resources()
        sys.exit(1)

if __name__ == '__main__':
    main()
