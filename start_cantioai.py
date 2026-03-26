#!/usr/bin/env python3
"""
Unified Startup Script for CantioAI Complete System
Can start the complete system, backend API, or WebUI frontend
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

def setup_environment():
    """Setup the environment"""
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Ensure logs directory exists
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)

def start_complete_system():
    """Start the complete CantioAI system"""
    print("[START] Starting CantioAI Complete System...")
    print("=" * 50)

    try:
        from src.main import main
        main()
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal, shutting down...")
    except Exception as e:
        print(f"❌ Error running complete system: {e}")
        return 1

    return 0

def start_backend_api():
    """Start just the backend API server"""
    print("[BACKEND] Starting CantioAI Backend API...")
    print("=" * 50)

    try:
        # Try to use the integrated main first
        from src.main import app
        import uvicorn

        # Get configuration
        from src.utils.config_integrated import get_integrated_config
        config = get_integrated_config()

        host = config.get('webui', {}).get('host', "0.0.0.0")
        port = config.get('webui', {}).get('port', 7860)
        workers = config.get('deployment', {}).get('resources', {}).get('cpu_limit', 1)
        if isinstance(workers, str):
            workers = int(workers) if workers.isdigit() else 1

        print(f"🌐 Server Configuration:")
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   Workers: {workers}")
        print(f"   Mode: {config.get('system', {}).get('mode', 'production')}")
        print("=" * 50)

        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )

    except ImportError as e:
        print(f"[WARN] Integrated components not available ({e}), falling back to simple backend")
        # Fall back to simple backend
        try:
            from src.webui.main_simple import app
            import uvicorn

            print("[WEB] Starting Simple Backend API on http://0.0.0.0:7860")
            print("[DOCS] API Documentation: http://0.0.0.0:7860/docs")
            print("=" * 50)

            uvicorn.run(
                app,
                host="0.0.0.0",
                port=7860,
                workers=1,
                log_level="info"
            )
        except Exception as e2:
            print(f"❌ Error starting backend: {e2}")
            return 1

    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal, shutting down API...")
    except Exception as e:
        print(f"❌ Error running backend API: {e}")
        return 1

    return 0

def start_frontend_dev():
    """Start the frontend development server"""
    print("[FRONTEND] Starting CantioAI Frontend Development Server...")
    print("=" * 50)

    frontend_dir = Path(__file__).parent / "frontend"
    if not frontend_dir.exists():
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return 1

    # Check if Node.js/npm is available
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("[ERROR] Node.js not found. Please install Node.js to run the frontend.")
            return 1
    except FileNotFoundError:
        print("[ERROR] Node.js not found. Please install Node.js to run the frontend.")
        return 1

    try:
        print("[INSTALL] Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)

        print("[START] Starting frontend development server...")
        print("[WEB] Frontend will be available at: http://localhost:3000")
        print("[LINK] Make sure the backend API is running on http://localhost:7860")
        print("=" * 50)

        subprocess.run(["npm", "run", "dev"], cwd=frontend_dir)

    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal, shutting down frontend...")
    except Exception as e:
        print(f"[ERROR] Error running frontend: {e}")
        return 1

    return 0

def start_full_stack():
    """Start the complete frontend + backend stack"""
    print("[FULL-STACK] Starting CantioAI Full Stack (Backend + Frontend)")
    print("=" * 60)

    # Start backend in background
    print("[BACKEND] Starting backend API...")
    backend_proc = subprocess.Popen(
        [sys.executable, __file__, "--backend"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait a moment for backend to start
    print("[WAIT] Waiting for backend to start...")
    time.sleep(3)

    # Check if backend started successfully
    if backend_proc.poll() is not None:
        stdout, stderr = backend_proc.communicate()
        print(f"[ERROR] Backend failed to start:")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return 1

    print("[SUCCESS] Backend API started")

    try:
        # Start frontend
        print("⚛️  Starting frontend...")
        frontend_result = start_frontend_dev()

        # If frontend exits, terminate backend
        if frontend_result != 0:
            backend_proc.terminate()
            backend_proc.wait(timeout=5)

        return frontend_result

    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal, shutting down full stack...")
    finally:
        # Clean up backend process
        if backend_proc.poll() is None:  # Still running
            print("[SHUTDOWN] Stopping backend API...")
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[WARN] Backend did not terminate gracefully, forcing...")
                backend_proc.kill()

    return 0

def show_help():
    """Show help information"""
    print("""
CantioAI Complete System Startup Script

Usage:
  python start_cantioai.py [OPTIONS]

Options:
  --complete, -c   Start the complete integrated system (default)
  --backend, -b    Start only the backend API server
  --frontend, -f   Start only the frontend development server
  --full-stack, -fs Start both backend and frontend (full stack)
  --help, -h       Show this help information

Examples:
  python start_cantioai.py              # Complete system
  python start_cantioai.py --backend    # Backend API only
  python start_cantioai.py --frontend   # Frontend only
  python start_cantioai.py --full-stack # Backend + Frontend

The system will:
  - Load configuration from configs/integrated/cantioai.yaml
  - Initialize all system components in proper order
  - Provide health monitoring and performance metrics
  - Support graceful shutdown via Ctrl+C
""")

def main():
    """Main entry point"""
    setup_environment()

    parser = argparse.ArgumentParser(
        description="CantioAI Complete System Startup Script",
        add_help=False
    )
    parser.add_argument("--complete", "-c", action="store_true", help="Start complete system")
    parser.add_argument("--backend", "-b", action="store_true", help="Start backend API")
    parser.add_argument("--frontend", "-f", action="store_true", help="Start frontend")
    parser.add_argument("--full-stack", "-fs", action="store_true", help="Start full stack")
    parser.add_argument("--help", "-h", action="store_true", help="Show help")

    args = parser.parse_args()

    # Show help if requested or no arguments
    if args.help or len(sys.argv) == 1:
        show_help()
        return 0

    # Determine what to start
    if args.backend:
        return start_backend_api()
    elif args.frontend:
        return start_frontend_dev()
    elif args.full_stack:
        return start_full_stack()
    else:  # Default to complete system
        return start_complete_system()

if __name__ == "__main__":
    sys.exit(main())