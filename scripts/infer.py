"""
Inference script for CantioAI model.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import soundfile as sf
from torchaudio.transforms import Resample

from cantioai.src.models.hybrid_svc import HybridSVC
from cantioai.src.inference.synthesizer import (
    VocoderInference,
    synthesize_from_file,
    batch_synthesize
)
from cantioai.src.utils.logging import setup_logging
from cantioai.src.utils.config import load_config

logger = logging.getLogger(__name__)


def main():
    """Main entry point for model inference."""
    parser = argparse.ArgumentParser(
        description="Run inference with CantioAI hybrid source-filter + neural vocoder model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input features (phoneme features, F0, speaker ID)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save output audio (.wav, .flac)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for synthesis (number of utterances to process)"
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cpu', 'cuda', etc.)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--f0-is-hz",
        action="store_true",
        help="Whether input F0 values are in Hz (True) or normalized (False)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=args.log_file
    )

    logger.info("Starting CantioAI model inference")
    logger.info(f"Arguments: {args}")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Setup device
        if args.device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            device = torch.device(args.device)

        logger.info(f"Using device: {device}")

        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Load model
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        logger.info(f"Loading model from checkpoint: {model_path}")

        # Initialize inference
        inferencer = VocoderInference(
            model_path=model_path,
            config_path=args.config,
            device=device
        )

        logger.info(f"VocoderInference initialized from {model_path}")

        # Load input features
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input features not found: {input_path}")

        logger.info(f"Loading input features from {input_path}")

        # Load features based on extension
        if input_path.suffix() == ".npy":
            phoneme_features = np.load(input_path)
            f0 = np.load(input_path)
            spk_id = np.load(input_path)
        else:  # Assume .pt
            phoneme_features = torch.load(input_path)
            f0 = torch.load(input_path)
            spk_id = torch.load(input_path).long()

        # Handle F0 unit
        if args.f0_is_hz:
            f0 = f0  # Already in Hz
            f0_hz = None
        else:
            # Generate constant F0 if not provided (e.g., for zero-shot)
            f0_hz = config.get("inference", {}).get("default_f0_hz", 220.0)
            f0 = np.full_like(phoneme_features, f0_hz) if isinstance(phoneme_features, np.ndarray) else \
                   torch.full_like(phoneme_features, f0_hz)

        # Run inference
        logger.info("Running inference...")
        start_time = time.time()

        # Single file synthesis
        if not args.batch_size or args.batch_size == 1:
            logger.info("Synthesizing from single file")
            waveform = inferencer.synthesize_from_file(
                input_path=input_path,
                f0_path=None,
                spk_id_path=None,
                output_path=args.output_path,
                device=device
            )

        # Batch synthesis
        elif args.batch_size and args.batch_size > 1:
            logger.info("Synthesizing from batch")
            # Process input as dictionary
            input_dict = {
                "phoneme_features": input_path,
                "f0": f0,
                "spk_id": spk_id
            }
            waveform = inferencer.batch_synthesize(
                batch=input_dict,
                device=device,
            )

        # Save output audio
        output_path = Path(args.output_path)
        if not output_path.exists():
            raise FileNotFoundError(f"Output directory not found: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving audio to {output_path}")

        # Optional: write to file
        if args.output_path:
            sf.write(
                str(output_path),
                waveform,
                samplerate=int(device),
                subtype='PCM_WAVE'
            )

        # E timing
        end_time = time.time() - start_time
        avg_time = end_time / 3600  # hours
        logger.info(
            f"Inference completed in {end_time:.2f}s - "
            f"Waveform saved to {output_path}"
        )

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()