"""
System Initializer for CantioAI Complete System
Handles proper initialization order and dependency management
"""

import logging
import time
from typing import Dict, List, Callable, Any
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class InitializationStage(Enum):
    """Stages of system initialization"""
    CONFIGURATION = 1
    CORE_UTILS = 2
    DATA_LAYER = 3
    MODEL_LAYER = 4
    TRAINING_LAYER = 5
    INFERENCE_LAYER = 6
    SERVICE_LAYER = 7
    UI_LAYER = 8
    MONITORING = 9
    COMPLETE = 10


class SystemInitializer:
    """Manages system initialization with proper dependency ordering"""

    def __init__(self):
        self.initialized_stages: set = set()
        self.initialization_functions: Dict[InitializationStage, List[Callable]] = {
            InitializationStage.CONFIGURATION: [],
            InitializationStage.CORE_UTILS: [],
            InitializationStage.DATA_LAYER: [],
            InitializationStage.MODEL_LAYER: [],
            InitializationStage.TRAINING_LAYER: [],
            InitializationStage.INFERENCE_LAYER: [],
            InitializationStage.SERVICE_LAYER: [],
            InitializationStage.UI_LAYER: [],
            InitializationStage.MONITORING: [],
        }
        self.initialization_status: Dict[InitializationStage, bool] = {}
        self.start_time = None
        self.logger = logging.getLogger(__name__)

    def register_initialization_function(
        self,
        stage: InitializationStage,
        func: Callable,
        name: str = None
    ) -> None:
        """Register a function to be called during a specific initialization stage"""
        if stage not in self.initialization_functions:
            self.initialization_functions[stage] = []

        func_name = name or func.__name__ if hasattr(func, '__name__') else str(func)
        self.initialization_functions[stage].append((func, func_name))
        self.logger.debug(f"Registered initialization function '{func_name}' for stage {stage.name}")

    def initialize_stage(self, stage: InitializationStage) -> bool:
        """Initialize a specific stage"""
        if stage in self.initialized_stages:
            self.logger.debug(f"Stage {stage.name} already initialized")
            return True

        if stage not in self.initialization_functions:
            self.logger.error(f"No initialization functions registered for stage {stage.name}")
            return False

        self.logger.info(f"Initializing stage: {stage.name}")
        stage_start = time.time()

        try:
            # Execute all initialization functions for this stage
            for func, func_name in self.initialization_functions[stage]:
                try:
                    self.logger.debug(f"[EXEC] Executing initialization function: {func_name}")
                    func()
                    self.logger.debug(f"[DONE] Completed initialization function: {func_name}")
                except Exception as e:
                    self.logger.error(f"[ERROR] Failed to execute initialization function '{func_name}': {e}")
                    return False

            # Mark stage as initialized
            self.initialized_stages.add(stage)
            self.initialization_status[stage] = True

            stage_time = time.time() - stage_start
            self.logger.info(f"[SUCCESS] Stage {stage.name} initialized successfully in {stage_time:.2f}s")
            return True

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to initialize stage {stage.name}: {e}")
            self.initialization_status[stage] = False
            return False

    def initialize_system(self) -> bool:
        """Initialize the complete system in proper order"""
        self.start_time = time.time()
        self.logger.info("[INIT] Starting CantioAI Complete System initialization...")

        # Define initialization order (respecting dependencies)
        initialization_order = [
            InitializationStage.CONFIGURATION,
            InitializationStage.CORE_UTILS,
            InitializationStage.DATA_LAYER,
            InitializationStage.MODEL_LAYER,
            InitializationStage.TRAINING_LAYER,
            InitializationStage.INFERENCE_LAYER,
            InitializationStage.SERVICE_LAYER,
            InitializationStage.UI_LAYER,
            InitializationStage.MONITORING,
        ]

        # Initialize each stage in order
        for stage in initialization_order:
            if not self.initialize_stage(stage):
                self.logger.error(f"[ERROR] System initialization failed at stage {stage.name}")
                return False

        # Mark system as complete
        self.initialized_stages.add(InitializationStage.COMPLETE)
        self.initialization_status[InitializationStage.COMPLETE] = True

        total_time = time.time() - self.start_time
        self.logger.info(f"[SUCCESS] CantioAI Complete System initialized successfully in {total_time:.2f}s")
        return True

    def get_initialization_status(self) -> Dict[str, Any]:
        """Get the current initialization status"""
        return {
            "initialized_stages": [stage.name for stage in self.initialized_stages],
            "stage_status": {stage.name: status for stage, status in self.initialization_status.items()},
            "is_complete": InitializationStage.COMPLETE in self.initialized_stages,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0
        }

    def is_stage_initialized(self, stage: InitializationStage) -> bool:
        """Check if a specific stage is initialized"""
        return stage in self.initialized_stages

    def wait_for_stage(self, stage: InitializationStage, timeout: float = 30.0) -> bool:
        """Wait for a specific stage to be initialized"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_stage_initialized(stage):
                return True
            time.sleep(0.1)
        return False


# Global initializer instance
_system_initializer = None


def get_system_initializer() -> SystemInitializer:
    """Get the system initializer (singleton pattern)"""
    global _system_initializer
    if _system_initializer is None:
        _system_initializer = SystemInitializer()
    return _system_initializer


def initialize_cantioai_system() -> bool:
    """Initialize the complete CantioAI system"""
    initializer = get_system_initializer()
    return initializer.initialize_system()


def get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    initializer = get_system_initializer()
    return initializer.get_initialization_status()


if __name__ == "__main__":
    # Test the system initializer
    logging.basicConfig(level=logging.INFO)

    try:
        initializer = SystemInitializer()

        # Register some dummy initialization functions
        def dummy_config_init():
            logger.info("[DUMMY] Dummy config initialization")

        def dummy_model_init():
            logger.info("[DUMMY] Dummy model initialization")

        initializer.register_initialization_function(
            InitializationStage.CONFIGURATION,
            dummy_config_init,
            "config_loader"
        )
        initializer.register_initialization_function(
            InitializationStage.MODEL_LAYER,
            dummy_model_init,
            "model_loader"
        )

        # Initialize system
        success = initializer.initialize_system()

        if success:
            print("[PASS] System initialization test passed")
            status = initializer.get_initialization_status()
            print(f"  Initialized stages: {status['initialized_stages']}")
            print(f"  Is complete: {status['is_complete']}")
        else:
            print("[FAIL] System initialization test failed")

    except Exception as e:
        print(f"[FAIL] System initializer test failed: {e}")