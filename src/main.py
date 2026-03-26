"""
CantioAI Complete System Main Entry Point
Integrates all stages into a unified system
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

# Initialize logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/cantioai.log', encoding='utf-8')
    ]
)

logger = logging.getLogger("CantioAI")

# Import system components
try:
    from src.utils.config_integrated import get_integrated_config, reload_config
    from src.utils.system_initializer import initialize_cantioai_system, get_system_status
    from src.utils.system_monitor import start_system_monitoring, get_system_health
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some integrated components not available: {e}")
    COMPONENTS_AVAILABLE = False

# Global state
_system_initialized = False
_shutdown_event = asyncio.Event()


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        _shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def initialize_system() -> bool:
    """Initialize the complete CantioAI system"""
    global _system_initialized

    try:
        logger.info("[START] Starting CantioAI Complete System...")
        logger.info("=" * 60)

        # Load and log configuration
        if COMPONENTS_AVAILABLE:
            config = get_integrated_config()
            logger.info(f"System Mode: {config.get('system', {}).get('mode', 'unknown')}")
            logger.info(f"Device: {config.get('system', {}).get('device', 'unknown')}")
            logger.info(f"Model Architecture: {config.get('model', {}).get('architecture', 'unknown')}")
            logger.info(f"Inference Mode: {config.get('inference', {}).get('mode', 'unknown')}")
            logger.info(f"Target Latency: {config.get('inference', {}).get('target_latency', 0)} ms")
            logger.info(f"WebUI Enabled: {config.get('webui', {}).get('enabled', False)}")

        # Initialize system components
        if COMPONENTS_AVAILABLE:
            logger.info("Initializing system components...")
            success = initialize_cantioai_system()
            if not success:
                logger.error("[ERROR] System initialization failed")
                return False

            # Log initialization status
            status = get_system_status()
            logger.info(f"Initialized stages: {len(status['initialized_stages'])}")
            logger.info(f"System ready: {status['is_complete']}")

        # Start monitoring
        if COMPONENTS_AVAILABLE:
            logger.info("[MONITOR] Starting system monitoring...")
            start_system_monitoring()

            # Give monitoring a moment to start
            await asyncio.sleep(1)

            health = get_system_health()
            logger.info(f"System health: {health.get('status', 'unknown')}")
        else:
            logger.info("[MONITOR] System monitoring skipped (components not available)")

        _system_initialized = True
        logger.info("[SUCCESS] CantioAI Complete System initialized successfully!")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        return False


async def run_system() -> None:
    """Run the main system loop"""
    logger.info("[RUN] CantioAI Complete System is running...")
    logger.info("Press Ctrl+C to initiate graceful shutdown")

    try:
        # Wait for shutdown signal
        await _shutdown_event.wait()

    except asyncio.CancelledError:
        logger.info("[CANCEL] System run loop cancelled")
    except Exception as e:
        logger.error(f"[ERROR] Error in system run loop: {e}")


async def shutdown_system() -> None:
    """Gracefully shutdown the system"""
    logger.info("[SHUTDOWN] Initiating graceful system shutdown...")
    logger.info("=" * 60)

    try:
        # Stop monitoring
        from src.utils.system_monitor import stop_system_monitoring
        stop_system_monitoring()
        logger.info("System monitoring stopped")

        # Log final status
        if COMPONENTS_AVAILABLE:
            status = get_system_status()
            health = get_system_health()
            logger.info("[STATUS] Final system status:")
            logger.info(f"  - Initialized stages: {len(status['initialized_stages'])}")
            logger.info(f"  - System complete: {status['is_complete']}")
            logger.info(f"  - Health status: {health.get('status', 'unknown')}")

        logger.info("[SUCCESS] CantioAI Complete System shutdown completed")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during system shutdown: {e}")


def main() -> None:
    """Main entry point"""
    try:
        # Setup signal handlers
        setup_signal_handlers()

        # Run the async main function
        async def async_main():
            # Initialize system
            if not await initialize_system():
                logger.error("❌ Failed to initialize system. Exiting.")
                sys.exit(1)

            # Run system
            await run_system()

            # Shutdown system
            await shutdown_system()

        # Run the async event loop
        asyncio.run(async_main())

    except KeyboardInterrupt:
        logger.info("[INTERRUPT] Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"[FATAL] Fatal error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()