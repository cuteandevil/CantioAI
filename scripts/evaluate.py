"""
Evaluation script for CantioAI model.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import json

from cantioai.src.models.hybrid_svc import HybridSVC
from cantioai.src.data.dataset import CantioAIDataset
from cantioai.src.inference.synthesizer import VocoderInference
from cantioai.src.utils.logging import setup_logging
from cantioai.src.utils.config import load_config

logger = logging.getLogger(__name__)


def evaluate_spectral_envelope(
    model: HybridSVC,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate spectral envelope prediction accuracy.

    Args:
        model: Trained HybridSVC model
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            # Move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            sp_pred, _, _ = model(
                phoneme_features=batch["phoneme_features"],
                f0=batch["f0"],
                spk_id=batch["spk_id"],
                f0_is_hz=True,
                return_quantized_f0=False
            )

            # Collect predictions and targets
            all_preds.append(sp_pred.cpu().numpy())
            all_targets.append(batch["target_sp"].cpu().numpy())

    # Concatenate
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Flatten for metric computation
    preds_flat = preds.reshape(-1, preds.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])

    # Compute metrics
    mse = mean_squared_error(targets_flat, preds_flat)
    mae = mean_absolute_error(targets_flat, preds_flat)
    rmse = np.sqrt(mse)

    # Correlation per dimension
    correlations = []
    for i in range(preds.shape[-1]):
        if np.std(preds_flat[:, i]) > 0 and np.std(targets_flat[:, i]) > 0:
            corr, _ = pearsonr(preds_flat[:, i], targets_flat[:, i])
            correlations.append(corr)
        else:
            correlations.append(0.0)
    mean_corr = np.mean(correlations) if correlations else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mean_correlation": mean_corr,
        "num_samples": len(all_targets),
        "num_frames": preds.shape[0]
    }


def evaluate_f0_accuracy(
    model: HybridSVC,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate F0 prediction accuracy (if model predicts F0).

    Args:
        model: Trained HybridSVC model
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on

    Returns:
        Dictionary of F0 evaluation metrics
    """
    # Check if model predicts F0
    if not hasattr(model, 'predictor') or not hasattr(model.predictor, 'fc2'):
        return {"error": "Model does not predict F0"}

    model.eval()
    all_preds = []
    all_targets = []
    all_quants = []

    with torch.no_grad():
        for batch in data_loader:
            # Move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            sp_pred, f0_quant, extras = model(
                phoneme_features=batch["phoneme_features"],
                f0=batch["f0"],
                spk_id=batch["spk_id"],
                f0_is_hz=True,
                return_quantized_f0=True
            )

            # Collect predictions and targets
            if f0_quant is not None:
                all_preds.append(f0_quant.cpu().numpy())
                all_targets.append(batch["f0"].cpu().numpy())
                all_quants.append(extras.get("f0_quant", np.zeros_like(f0_quant.cpu().numpy())))

    if not all_preds:
        return {"error": "No F0 predictions found"}

    # Concatenate
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    quants = np.concatenate(all_quants, axis=0) if all_quants else np.zeros_like(preds)

    # Flatten for metric computation
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    quants_flat = quants.reshape(-1)

    # Compute metrics
    mse = mean_squared_error(targets_flat, preds_flat)
    mae = mean_absolute_error(targets_flat, preds_flat)
    rmse = np.sqrt(mse)

    # Quantization accuracy (how close predictions are to quantized values)
    quant_mse = mean_squared_error(quants_flat, preds_flat)
    quant_mae = mean_absolute_error(quants_flat, preds_flat)

    # Correlation
    if np.std(preds_flat) > 0 and np.std(targets_flat) > 0:
        corr, _ = pearsonr(preds_flat, targets_flat)
    else:
        corr = 0.0

    return {
        "f0_mse": mse,
        "f0_mae": mae,
        "f0_rmse": rmse,
        "f0_correlation": corr,
        "f0_quant_mse": quant_mse,
        "f0_quant_mae": quant_mae,
        "num_samples": len(all_targets),
        "num_frames": preds.shape[0]
    }


def synthesize_and_save(
    model: HybridSVC,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: Union[str, Path],
    vocoder_config: Dict[str, Any]
) -> None:
    """
    Synthesize audio from model predictions and save to disk.

    Args:
        model: Trained HybridSVC model
        data_loader: DataLoader for evaluation data
        device: Device to run synthesis on
        output_dir: Directory to save synthesized audio
        vocoder_config: Configuration for WORLD vocoder
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize vocoder
    from cantioai.src.inference.vocoder import WORLDVocoder
    vocoder = WORLDVocoder(**vocoder_config)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            sp_pred, f0_quant, extras = model(
                phoneme_features=batch["phoneme_features"],
                f0=batch["f0"],
                spk_id=batch["spk_id"],
                f0_is_hz=True,
                return_quantized_f0=True
            )

            # Use quantized F0 for synthesis if available, else original
            f0_to_use = f0_quant if f0_quant is not None else batch["f0"]

            # Synthesize for each item in batch
            for i in range(f0_to_use.shape[0]):
                # Extract single item
                f0_item = f0_to_use[i:i+1]  # (1, T, 1)
                sp_item = sp_pred[i:i+1]   # (1, T, D_sp)
                spk_item = batch["spk_id"][i:i+1]  # (1,)

                # Synthesize
                waveform = vocoder.synthesize_from_features(
                    f0=f0_item.squeeze(0).cpu().numpy(),  # (T,)
                    sp_pred=sp_item.squeeze(0).cpu().numpy(),  # (T, D_sp)
                    ap=None  # Assume voiced speech
                )

                # Save audio
                output_path = output_dir / f"batch{batch_idx:04d}_item{i:02d}.wav"
                sf.write(str(output_path), waveform, samplerate=vocoder.sample_rate)

    logger.info(f"Synthesized audio saved to {output_dir}")


def main():
    """Main entry point for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate CantioAI hybrid source-filter + neural vocoder model"
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
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing preprocessed features"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Whether to synthesize audio from predictions"
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
        help="Device to run evaluation on ('cpu', 'cuda', etc.)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=args.log_file
    )

    logger.info("Starting CantioAI model evaluation")
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

        # Load model
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        logger.info(f"Loading model from checkpoint: {model_path}")

        # Initialize model
        model_config = config.get("model", {})
        model = HybridSVC(
            phoneme_feature_dim=model_config.get("phoneme_feature_dim", 32),
            spectral_envelope_dim=model_config.get("spectral_envelope_dim", 60),
            speaker_embed_dim=model_config.get("speaker_embed_dim", 128),
            n_speakers=model_config.get("n_speakers", 100),
            use_pitch_quantizer=model_config.get("use_pitch_quantizer", True),
            pitch_quantizer_config=model_config.get("pitch_quantizer", {}),
            **{
                k: v for k, v in model_config.items()
                if k not in ["phoneme_feature_dim", "spectral_envelope_dim",
                           "speaker_embed_dim", "n_speakers", "use_pitch_quantizer",
                           "pitch_quantizer"]
            }
        )
        # Load checkpoint
        checkpoint_epoch = model.load_checkpoint(model_path, load_optimizer=False)
        model.to(device)
        logger.info(f"Loaded model from epoch {checkpoint_epoch}")

        # Initialize dataset
        data_config = config.get("data", {})
        dataset = CantioAIDataset(
            data_dir=args.data_dir,
            split=args.split,
            phoneme_feature_dim=model_config.get("phoneme_feature_dim", 32),
            spectral_envelope_dim=model_config.get("spectral_envelope_dim", 60),
            augment=False  # No augmentation for evaluation
        )

        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=config.get("training", {}).get("batch_size", 16),
            shuffle=False,
            num_workers=config.get("training", {}).get("num_workers", 4),
            pin_memory=config.get("training", {}).get("pin_memory", True)
        )
        logger.info(
            f"Evaluation dataset loaded: {len(dataset)} samples "
            f"from {args.split} split"
        )

        # Run evaluation
        results = {}

        # Spectral envelope evaluation
        logger.info("Evaluating spectral envelope prediction...")
        sp_metrics = evaluate_spectral_envelope(model, data_loader, device)
        results.update({"sp_" + k: v for k, v in sp_metrics.items()})
        logger.info(f"Spectral envelope metrics: {sp_metrics}")

        # F0 evaluation (if applicable)
        logger.info("Evaluating F0 prediction...")
        f0_metrics = evaluate_f0_accuracy(model, data_loader, device)
        results.update({"f0_" + k: v for k, v in f0_metrics.items()})
        logger.info(f"F0 metrics: {f0_metrics}")

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {results_path}")

        # Optional: synthesize audio
        if args.synthesize:
            logger.info("Synthesizing audio from predictions...")
            vocoder_config = {
                "sample_rate": config.get("feature", {}).get("sample_rate", 24000),
                "frame_period": config.get("feature", {}).get("frame_period", 5.0),
                "fft_size": config.get("feature", {}).get("fft_size", 1024),
                "f0_floor": config.get("feature", {}).get("f0_floor", 71.0),
                "f0_ceil": config.get("feature", {}).get("f0_ceil", 800.0),
                "num_mcep": config.get("feature", {}).get("num_mcep", 60)
            }
            synthesize_and_save(
                model=model,
                data_loader=data_loader,
                device=device,
                output_dir=output_dir / "synthesized",
                vocoder_config=vocoder_config
            )

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()