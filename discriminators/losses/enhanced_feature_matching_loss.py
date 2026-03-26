"""Enhanced Feature Matching Loss with pyramid matching and adaptive weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class FeaturePyramidMatching(nn.Module):
    """Feature pyramid matching for multi-scale feature comparison."""

    def __init__(self, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels

    def forward(self, real_features: Dict[str, List[List[torch.Tensor]]],
                fake_features: Dict[str, List[List[torch.Tensor]]]) -> torch.Tensor:
        """
        Compute feature pyramid matching loss.

        Args:
            real_features: Features from real data
            fake_features: Features from fake data

        Returns:
            Pyramid matching loss
        """
        total_loss = 0.0

        # Process each discriminator type (MPD, MSD, etc.)
        for disc_type in real_features.keys():
            if disc_type not in fake_features:
                continue

            real_disc_features = real_features[disc_type]
            fake_disc_features = fake_features[disc_type]

            # Compute pyramid loss for this discriminator type
            pyramid_loss = self._compute_pyramid_loss(
                real_disc_features, fake_disc_features
            )
            total_loss += pyramid_loss

        return total_loss

    def _compute_pyramid_loss(self, real_features: List[List[torch.Tensor]],
                             fake_features: List[List[torch.Tensor]]) -> torch.Tensor:
        """Compute pyramid loss for a single discriminator type."""
        total_loss = 0.0

        # For each discriminator (period/scale)
        for disc_idx, (real_disc, fake_disc) in enumerate(zip(real_features, fake_features)):
            # For each layer in the discriminator
            for layer_idx, (real_layer, fake_layer) in enumerate(zip(real_disc, fake_disc)):
                # Compute loss at different pyramid levels
                layer_loss = 0.0
                for level in range(self.num_levels):
                    # Downsample features for this level
                    if level > 0:
                        pool_size = 2 ** level
                        real_down = F.avg_pool1d(
                            real_layer.transpose(1, 2),
                            kernel_size=pool_size,
                            stride=pool_size,
                            padding=pool_size//2
                        ).transpose(1, 2)

                        fake_down = F.avg_pool1d(
                            fake_layer.transpose(1, 2),
                            kernel_size=pool_size,
                            stride=pool_size,
                            padding=pool_size//2
                        ).transpose(1, 2)
                    else:
                        real_down = real_layer
                        fake_down = fake_layer

                    # Compute loss at this level
                    if real_down.shape[-1] > 0 and fake_down.shape[-1] > 0:
                        diff = F.l1_loss(real_down, fake_down)
                        # Weight by level (higher levels get less weight)
                        level_weight = 1.0 / (2 ** level)
                        layer_loss += level_weight * diff

                total_loss += layer_loss

        return total_loss


class EnhancedFeatureMatchingLoss(nn.Module):
    """
    Enhanced Feature Matching Loss:
    Matches generator features to real data features in discriminator intermediate layers
    with pyramid matching and adaptive weighting.

    Design points:
    1. Multi-scale feature matching: Different layer features
    2. Adaptive weights: Different layer features have different importance
    3. Feature normalization: Normalize features for stable training
    4. Pyramid matching: Multi-resolution feature comparison
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        # Feature matching layer weights (learnable parameters)
        self.layer_weights = nn.ParameterDict({
            f"layer_{i}": nn.Parameter(torch.tensor(weight))
            for i, weight in enumerate(config.get("layer_weights", [1.0, 0.8, 0.6, 0.4, 0.2]))
        })

        # Feature normalization method
        self.normalization = config.get("normalization", "layer_norm")

        # Loss function type
        self.loss_fn_type = config.get("loss_type", "l1")

        # Feature pyramid matching
        self.feature_pyramid_matching = FeaturePyramidMatching(
            num_levels=config.get("num_pyramid_levels", 3)
        )

        # Discriminator-type specific weights
        self.disc_weights = nn.ParameterDict({
            disc_type: nn.Parameter(torch.tensor(weight))
            for disc_type, weight in config.get("disc_weights", {
                "mpd": 1.0,
                "msd": 0.8,
                "source_filter": 0.5,
                "control_consistency": 0.3
            }).items()
        })

    def forward(self, real_features: Dict[str, List[List[torch.Tensor]]],
                fake_features: Dict[str, List[List[torch.Tensor]]],
                feature_types: Union[str, List[str]] = "all") -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute enhanced feature matching loss.

        Args:
            real_features: Features from real data
            fake_features: Features from fake data
            feature_types: Feature types to match ("all" or list of types)

        Returns:
            Feature matching loss and detailed loss information
        """
        total_loss = 0.0
        loss_details = {}

        # Determine which feature types to process
        if feature_types == "all":
            types_to_process = list(real_features.keys())
        else:
            types_to_process = [ft for ft in feature_types if ft in real_features]

        # Process each feature type
        for feature_name in types_to_process:
            if feature_name not in fake_features:
                continue

            real_feat = real_features[feature_name]
            fake_feat = fake_features[feature_name]

            # Get layer weight for this feature type
            layer_weight = self.layer_weights.get(
                f"layer_{list(real_features.keys()).index(feature_name)}",
                1.0
            )

            # Get discriminator weight
            disc_weight = self.disc_weights.get(feature_name, 1.0)

            # Compute base feature matching loss
            base_loss, base_details = self._compute_base_feature_matching(
                real_feat, fake_feat
            )

            # Apply pyramid matching
            pyramid_loss = self.feature_pyramid_matching(
                {feature_name: real_feat},
                {feature_name: fake_feat}
            )

            # Combine losses
            feature_loss = (base_loss + pyramid_loss) * layer_weight * disc_weight
            total_loss += feature_loss

            # Store detailed losses
            loss_details[f"{feature_name}_base_loss"] = base_loss.item()
            loss_details[f"{feature_name}_pyramid_loss"] = pyramid_loss.item()
            loss_details[f"{feature_name}_layer_weight"] = layer_weight.item()
            loss_details[f"{feature_name}_disc_weight"] = disc_weight.item()
            loss_details[f"{feature_name}_total_loss"] = feature_loss.item()

            # Add base loss details
            for key, value in base_details.items():
                loss_details[f"{feature_name}_{key}"] = value

        return total_loss, loss_details

    def _compute_base_feature_matching(self, real_features: List[List[torch.Tensor]],
                                      fake_features: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute base feature matching loss.

        Args:
            real_features: Real features from discriminator
            fake_features: Fake features from discriminator

        Returns:
            Base feature matching loss and details
        """
        loss_dict = {}
        total_loss = 0.0

        # Process each discriminator (period/scale)
        for disc_idx, (real_disc, fake_disc) in enumerate(zip(real_features, fake_features)):
            # Process each layer in the discriminator
            for layer_idx, (real_layer, fake_layer) in enumerate(zip(real_disc, fake_disc)):
                # Skip if shapes don't match
                if real_layer.shape != fake_layer.shape:
                    continue

                # Normalize features if required
                if self.normalization == "layer_norm":
                    real_layer = F.layer_norm(real_layer, real_layer.shape[1:])
                    fake_layer = F.layer_norm(fake_layer, fake_layer.shape[1:])
                elif self.normalization == "instance_norm":
                    real_layer = F.instance_norm(real_layer.unsqueeze(1)).squeeze(1)
                    fake_layer = F.instance_norm(fake_layer.unsqueeze(1)).squeeze(1)

                # Compute feature matching loss
                if self.loss_fn_type == "l1":
                    diff = F.l1_loss(real_layer, fake_layer, reduction='none')
                elif self.loss_fn_type == "l2":
                    diff = F.mse_loss(real_layer, fake_layer, reduction='none')
                else:  # cosine similarity
                    # Flatten for cosine embedding loss
                    real_flat = real_layer.view(real_layer.size(0), -1)
                    fake_flat = fake_layer.view(fake_layer.size(0), -1)
                    target = torch.ones(real_flat.size(0)).to(real_layer.device)
                    diff = F.cosine_embedding_loss(real_flat, fake_flat, target, reduction='none')
                    # Reshape back to original shape for consistency
                    diff = diff.view(-1, 1, 1).expand_as(real_layer)

                # Mean over all dimensions except batch
                loss = diff.mean(dim=list(range(1, diff.dim())))  # [B]
                loss = loss.mean()  # Scalar

                total_loss += loss
                loss_dict[f"disc_{disc_idx}_layer_{layer_idx}"] = loss.item()

        return total_loss, loss_dict

    def compute_adaptive_weights(self, real_features: Dict[str, List[List[torch.Tensor]]],
                                fake_features: Dict[str, List[List[torch.Tensor]]],
                                current_epoch: int) -> Dict[str, float]:
        """
        Compute adaptive weights for feature matching loss.

        Args:
            real_features: Features from real data
            fake_features: Features from fake data
            current_epoch: Current training epoch

        Returns:
            Adaptive weights for each feature type
        """
        adaptive_weights = {}

        # Compute feature distances for each type
        feature_distances = {}
        for feature_name in real_features.keys():
            if feature_name not in fake_features:
                continue

            real_feat = real_features[feature_name]
            fake_feat = fake_features[feature_name]

            # Compute average feature distance
            total_distance = 0.0
            count = 0

            for real_disc, fake_disc in zip(real_feat, fake_feat):
                for real_layer, fake_layer in zip(real_disc, fake_disc):
                    if real_layer.shape == fake_layer.shape:
                        distance = F.l1_loss(real_layer, fake_layer).item()
                        total_distance += distance
                        count += 1

            avg_distance = total_distance / max(count, 1)
            feature_distances[feature_name] = avg_distance

        # Compute adaptive weights (inverse distance weighting)
        if feature_distances:
            max_distance = max(feature_distances.values())
            min_distance = min(feature_distances.values())
            distance_range = max_distance - min_distance

            for feature_name, distance in feature_distances.items():
                if distance_range > 0:
                    # Normalize distance to [0, 1] and invert
                    normalized_distance = (distance - min_distance) / distance_range
                    adaptive_weight = 2.0 - normalized_distance  # Range [1, 2]
                else:
                    adaptive_weight = 1.0

                # Apply epoch-based scaling
                if current_epoch < 10:
                    # Early training: uniform weights
                    final_weight = 1.0
                else:
                    # Later training: use adaptive weights
                    final_weight = adaptive_weight

                adaptive_weights[feature_name] = final_weight

        return adaptive_weights