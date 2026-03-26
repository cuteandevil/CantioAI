"""
Data preprocessing script for CantioAI.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

from cantioai.src.data.preprocess import (
    WorldFeatureExtractor,
    extract_features_from_dir
)
from cantioai.src.utils.logging import setup_logging
from cantioai.src.utils.config import load_config

logger = logging.getLogger(__name__)


def main():
    """Main entry point for data preprocessing."""
    parser = argparse.ArgumentParser(
        description="Extract features from audio files using WORLD vocoder"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input audio files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save extracted features"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (None for stdout only)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=args.log_file
    )

    logger.info("Starting CantioAI data preprocessing")
    logger.info(f"Arguments: {args}")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Extract preprocessing parameters
        prep_config = config.get("feature", {})

        # Initialize feature extractor
        extractor = WorldFeatureExtractor(**prep_config)

        # Process directory
        logger.info(
            f"Processing audio files from {args.input_dir} "
            f"to {args.output_dir}"
        )
        extractor.process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )

        logger.info("Data preprocessing completed successfully")

    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()