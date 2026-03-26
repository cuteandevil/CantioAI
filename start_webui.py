#!/usr/bin/env python3
"""
Startup script for CantioAI WebUI Backend
"""

import sys
import os
import uvicorn
from src.webui.main import app

def main():
    """Start the WebUI backend server"""
    print("Starting CantioAI Web Interface Backend...")
    print("=" * 50)

    # Import config to get server settings
    try:
        from src.utils.config import get_config
        config = get_config()
        webui_config = config.get('webui', {})
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        webui_config = {}

    # Get server configuration
    server_config = webui_config.get('server', {})
    host = server_config.get('host', "0.0.0.0")
    port = server_config.get('port', 7860)
    workers = server_config.get('workers', 1)

    print(f"Server Configuration:")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Workers: {workers}")
    print(f"  Mode: {webui_config.get('mode', 'development')}")
    print("=" * 50)

    # Start the server
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nShutting down WebUI backend...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()