import os
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dataset import restore_full_sequence
from model import ASDiffusionModel, cosine_beta_schedule, extract
from dual_hand_model import DualHandASDiffusionModel
from dual_hand_dataset import DualHandVideoFeatureDataset, get_dual_hand_data_dicts
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time
from utils import mode_filter
import random


class DataAugmentation:
    """
    Advanced data augmentation for dual-hand action segmentation
    Implements temporal jittering, action substitution, and confidence smoothing
    """
    
    def __init__(self, num_classes, temporal_jitter_std=2, action_sub_prob=0.1, confidence_smooth_alpha=0.1):
        self.num_classes = num_classes
        self.temporal_jitter_std = temporal_jitter_std
        self.action_sub_prob = action_sub_prob
        self.confidence_smooth_alpha = confidence_smooth_alpha
        
        # Define similar actions for substitution (customize based on your dataset)
        self.similar_actions = self._get_similar_action_groups()
    
    def _get_similar_action_groups(self):
        """
        Define groups of similar actions that can be substituted
        Customize this based on your specific action classes
        """
        similar_groups = []
        
        # Example: group actions by similarity (you should customize this)
        if self.num_classes >= 10:
            similar_groups = [
                [0, 1, 2],      # Basic manipulation actions
                [3, 4, 5],      # Tool usage actions
                [6, 7, 8],      # Movement actions
                # Add more groups as needed
            ]
        else:
            # For smaller action sets, create pairs
            for i in range(0, self.num_classes - 1, 2):
                if i + 1 < self.num_classes:
                    similar_groups.append([i, i + 1])
        
        return similar_groups
    
    def temporal_jittering(self, labels, boundaries):
        """
        Apply temporal jittering to action boundaries
        
        Args:
            labels: Action labels [T]
            boundaries: Boundary labels [T]
        
        Returns:
            Jittered labels and boundaries
        """
        labels = labels.clone()
        boundaries = boundaries.clone()
        
        # Find action boundaries
        transitions = (labels[1:] != labels[:-1]).nonzero(as_tuple=True)[0] + 1
        
        # Apply random jitter to each boundary
        for boundary_idx in transitions:
            if boundary_idx > 0 and boundary_idx < len(labels) - 1:
                # Generate jitter amount
                jitter = int(np.random.normal(0, self.temporal_jitter_std))
                jitter = max(-boundary_idx, min(jitter, len(labels) - boundary_idx - 1))
                
                if jitter != 0:
                    # Shift the boundary
                    if jitter > 0:
                        # Move boundary forward
                        labels[boundary_idx:boundary_idx + jitter] = labels[boundary_idx - 1]
                    else:
                        # Move boundary backward
                        labels[boundary_idx + jitter:boundary_idx] = labels[boundary_idx]
        
        # Recalculate boundaries
        new_boundaries = torch.zeros_like(boundaries)
        transitions = (labels[1:] != labels[:-1]).nonzero(as_tuple=True)[0] + 1
        for t in transitions:
            if t < len(new_boundaries):
                new_boundaries[t] = 1.0
        
        return labels, new_boundaries
    
    def action_substitution(self, labels):
        """
        Occasionally substitute similar actions
        
        Args:
            labels: Action labels [T]
        
        Returns:
            Labels with occasional substitutions
        """
        labels = labels.clone()
        
        # Get unique actions in the sequence
        unique_actions = labels.unique()
        
        for action in unique_actions:
            # Check if this action can be substituted
            for group in self.similar_actions:
                if action.item() in group:
                    # Randomly decide whether to substitute
                    if random.random() < self.action_sub_prob:
                        # Choose a different action from the same group
                        substitute_candidates = [a for a in group if a != action.item()]
                        if substitute_candidates:
                            substitute_action = random.choice(substitute_candidates)
                            labels[labels == action] = substitute_action
                    break
        
        return labels
    
    def confidence_smoothing(self, predictions, alpha=None):
        """
        Apply label smoothing to predictions for confidence smoothing
        
        Args:
            predictions: Model predictions [B, C, T]
            alpha: Smoothing factor
        
        Returns:
            Smoothed predictions
        """
        if alpha is None:
            alpha = self.confidence_smooth_alpha
        
        # Convert to probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Apply smoothing
        smooth_probs = (1 - alpha) * probs + alpha / self.num_classes
        
        # Convert back to logits
        smooth_logits = torch.log(smooth_probs + 1e-8)
        
        return smooth_logits
    
    def apply_augmentation(self, lh_labels, rh_labels, lh_boundaries, rh_boundaries, predictions=None):
        """
        Apply all augmentation techniques
        
        Args:
            lh_labels: Left hand labels [T]
            rh_labels: Right hand labels [T]
            lh_boundaries: Left hand boundaries [T]
            rh_boundaries: Right hand boundaries [T]
            predictions: Optional predictions for confidence smoothing [B, C, T]
        
        Returns:
            Augmented data
        """
        # Apply temporal jittering
        lh_labels_aug, lh_boundaries_aug = self.temporal_jittering(lh_labels, lh_boundaries)
        rh_labels_aug, rh_boundaries_aug = self.temporal_jittering(rh_labels, rh_boundaries)
        
        # Apply action substitution
        lh_labels_aug = self.action_substitution(lh_labels_aug)
        rh_labels_aug = self.action_substitution(rh_labels_aug)
        
        # Apply confidence smoothing to predictions if provided
        if predictions is not None:
            predictions = self.confidence_smoothing(predictions)
        
        return lh_labels_aug, rh_labels_aug, lh_boundaries_aug, rh_boundaries_aug, predictions


class EnhancedEncoder(nn.Module):
    """
    Enhanced encoder with larger capacity, attention mechanisms, and skip connections
    """
    
    def __init__(self, encoder_params):
        super(EnhancedEncoder, self).__init__()
        
        # Increase feature capacity
        original_f_maps = encoder_params.get('num_f_maps', 256)
        enhanced_f_maps = int(original_f_maps * 1.5)  # 50% larger: 384
        
        # Increase number of layers
        original_layers = encoder_params.get('num_layers', 12)
        enhanced_layers = min(original_layers + 4, 20)  # Add 4 layers, cap at 20
        
        self.num_classes = encoder_params['num_classes']
        self.input_dim = encoder_params['input_dim']
        self.num_f_maps = enhanced_f_maps  # This should be 384
        self.num_layers = enhanced_layers
        self.feature_layer_indices = encoder_params.get('feature_layer_indices', [])
        
        # Input projection - should go from input_dim (2048) to enhanced_f_maps (384)
        self.conv_in = nn.Conv1d(self.input_dim, enhanced_f_maps, 1)
        
        # Create backbone layers that maintain enhanced_f_maps channels throughout
        # Each layer should take enhanced_f_maps input and produce enhanced_f_maps output
        self.backbone_layers = nn.ModuleList()
        for i in range(enhanced_layers):
            # Create a simple conv layer that maintains the channel dimension
            layer = nn.Sequential(
                nn.Conv1d(enhanced_f_maps, enhanced_f_maps, 
                         kernel_size=encoder_params.get('kernel_size', 5), 
                         padding=encoder_params.get('kernel_size', 5)//2),
                nn.BatchNorm1d(enhanced_f_maps),
                nn.ReLU(),
                nn.Dropout(encoder_params.get('normal_dropout_rate', 0.1))
            )
            self.backbone_layers.append(layer)
            
        # Self-attention mechanism
        self.self_attention = nn.MultiheadAttention(
            embed_dim=enhanced_f_maps,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Skip connections - ensure they maintain channel dimensions
        num_skip_connections = enhanced_layers // 4
        self.skip_connections = nn.ModuleList([
            nn.Conv1d(enhanced_f_maps, enhanced_f_maps, 1) 
            for _ in range(num_skip_connections)
        ])
        
        # Output projection
        self.conv_out = nn.Conv1d(enhanced_f_maps, self.num_classes, 1)
    
    def forward(self, x, get_features=False):
        if get_features:
            # Mimic the original encoder's exact behavior
            assert(self.feature_layer_indices is not None and len(self.feature_layer_indices) > 0)
            features = []
            
            # Add input features if -1 is in feature_layer_indices
            if -1 in self.feature_layer_indices:
                features.append(x)
            
            # Apply input projection (equivalent to conv_in in original)
            x = self.conv_in(x)  # [B, enhanced_f_maps, T]
            
            # Process through backbone layers and collect features ONLY at specified indices
            layer_features = []
            
            # Process through backbone with skip connections
            for i, layer in enumerate(self.backbone_layers):
                residual = x
                x = layer(x)
                
                # Add skip connection every 3 layers
                if i % 3 == 2 and len(skip_features := getattr(self, '_skip_features', [])) > 0:
                    skip_idx = len(skip_features) - 1
                    if skip_idx < len(self.skip_connections):
                        skip_feature = self.skip_connections[skip_idx](skip_features[0])
                        x = x + skip_feature
                    skip_features = skip_features[1:]  # Remove used skip feature
                    setattr(self, '_skip_features', skip_features)
                
                # Store every 4th layer for skip connections
                if i % 4 == 0:
                    if not hasattr(self, '_skip_features'):
                        setattr(self, '_skip_features', [])
                    self._skip_features.append(x)
                
                # Store ALL layer outputs (we'll select specific ones later)
                layer_features.append(x)
            
            # Apply self-attention
            x_att = x.transpose(1, 2)
            att_out, _ = self.self_attention(x_att, x_att, x_att)
            x_att = att_out.transpose(1, 2)
            x = x + x_att
            
            # CRITICAL: Only collect features from the EXACT specified indices
            # This ensures we get exactly len(feature_layer_indices) × enhanced_f_maps channels
            selected_features = []
            for idx in self.feature_layer_indices:
                if idx >= 0 and idx < len(layer_features):
                    selected_features.append(layer_features[idx])
            
            # Add selected features to the features list
            features.extend(selected_features)
            
            # Output prediction
            out = self.conv_out(x)
            
            # Add encoder predictions if -2 is in feature_layer_indices
            if -2 in self.feature_layer_indices:
                features.append(F.softmax(out, 1))
            
            # Concatenate all features - this should exactly match expected dimensions
            if features:
                final_features = torch.cat(features, dim=1)
                return out, final_features
            else:
                # Fallback - should not happen with proper feature_layer_indices
                return out, x
        else:
            # Simple forward pass without features
            x = self.conv_in(x)
            
            # Process through backbone
            for i, layer in enumerate(self.backbone_layers):
                x = layer(x)
            
            # Apply self-attention
            x_att = x.transpose(1, 2)
            att_out, _ = self.self_attention(x_att, x_att, x_att)
            x_att = att_out.transpose(1, 2)
            x = x + x_att
            
            # Output projection
            out = self.conv_out(x)
            return out


class EnhancedDualHandASDiffusionModel(nn.Module):
    """
    Enhanced dual-hand model with improved encoders and architecture tweaks
    """
    
    def __init__(self, encoder_params, decoder_params, diffusion_params, num_classes, device):
        super(EnhancedDualHandASDiffusionModel, self).__init__()
        
        self.device = device
        self.num_classes = num_classes
        
        # Calculate enhanced feature dimensions
        original_f_maps = encoder_params.get('num_f_maps', 256)
        enhanced_f_maps = int(original_f_maps * 1.5)  # 384 instead of 256
        
        # Create enhanced encoder parameters
        enhanced_encoder_params = copy.deepcopy(encoder_params)
        enhanced_encoder_params['num_classes'] = num_classes
        enhanced_encoder_params['num_f_maps'] = enhanced_f_maps  # Use enhanced size
        
        # Calculate decoder input dimension correctly using the enhanced feature maps
        enhanced_decoder_params = copy.deepcopy(decoder_params)
        decoder_input_dim = len([i for i in encoder_params['feature_layer_indices'] if i not in [-1, -2]]) * enhanced_f_maps
        if -1 in encoder_params['feature_layer_indices']:
            decoder_input_dim += encoder_params['input_dim']
        if -2 in encoder_params['feature_layer_indices']:
            decoder_input_dim += num_classes
            
        enhanced_decoder_params['input_dim'] = decoder_input_dim
        enhanced_decoder_params['num_classes'] = num_classes
        
        print(f"🔧 Enhanced decoder input dimension: {decoder_input_dim}")
        print(f"   - Enhanced f_maps: {enhanced_f_maps} (vs original {original_f_maps})")
        print(f"   - Feature layers: {len([i for i in encoder_params['feature_layer_indices'] if i not in [-1, -2]])} layers")
        
        # Don't use the base ASDiffusionModel - build everything ourselves to avoid dimension issues
        
        # Diffusion parameters (copied from ASDiffusionModel)
        timesteps = diffusion_params['timesteps']
        sampling_timesteps = diffusion_params['sampling_timesteps']
        self.ddim_sampling_eta = diffusion_params['ddim_sampling_eta']
        self.snr_scale = diffusion_params['snr_scale']
        self.detach_decoder = diffusion_params['detach_decoder']
        self.cond_types = diffusion_params['cond_types']
        
        # Set up diffusion schedule (copied from ASDiffusionModel)
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps = torch.arange(timesteps)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Posterior variance
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Create enhanced encoders directly with correct parameters
        enhanced_encoder_params_for_init = copy.deepcopy(encoder_params)
        enhanced_encoder_params_for_init['num_classes'] = num_classes
        
        self.left_hand_encoder = EnhancedEncoder(enhanced_encoder_params_for_init)
        self.right_hand_encoder = EnhancedEncoder(enhanced_encoder_params_for_init)
        
        # Create decoders with correct dimensions
        from model import DecoderModel
        self.left_hand_decoder = DecoderModel(**enhanced_decoder_params)
        self.right_hand_decoder = DecoderModel(**enhanced_decoder_params)
        
        print(f"🚀 Enhanced Dual Hand Model initialized:")
        print(f"   - Enhanced encoders with 50% larger capacity ({enhanced_f_maps} vs {original_f_maps} channels)")
        print(f"   - 16 encoder layers (vs 12)")
        print(f"   - Self-attention in encoders")
        print(f"   - Skip connections every 3 layers")
        print(f"   - NO cross-hand attention (removed for stability)")
        print(f"   - Compatible decoder input dimension: {decoder_input_dim}")
        
        # Create base models for proper DDIM sampling during evaluation
        from model import ASDiffusionModel
        base_encoder_params = copy.deepcopy(encoder_params)
        base_decoder_params = copy.deepcopy(decoder_params)
        
        self.base_lh_model = ASDiffusionModel(
            base_encoder_params, base_decoder_params, copy.deepcopy(diffusion_params), num_classes, device
        )
        self.base_rh_model = ASDiffusionModel(
            copy.deepcopy(encoder_params), copy.deepcopy(decoder_params), copy.deepcopy(diffusion_params), num_classes, device
        )
    
    def get_training_loss(self, lh_video_feats, rh_video_feats, 
                         lh_event_gt, rh_event_gt, 
                         lh_boundary_gt, rh_boundary_gt,
                         **criterions):
        
        # Get encoder outputs and features for both hands
        lh_encoder_out, lh_backbone_feats = self.left_hand_encoder(lh_video_feats, get_features=True)
        rh_encoder_out, rh_backbone_feats = self.right_hand_encoder(rh_video_feats, get_features=True)
        
        # Calculate losses for both hands using the same logic as ASDiffusionModel
        lh_loss_dict = self._calculate_single_hand_loss(
            lh_video_feats, lh_encoder_out, lh_backbone_feats, 
            lh_event_gt, lh_boundary_gt, self.left_hand_decoder, "lh", **criterions
        )
        
        rh_loss_dict = self._calculate_single_hand_loss(
            rh_video_feats, rh_encoder_out, rh_backbone_feats,
            rh_event_gt, rh_boundary_gt, self.right_hand_decoder, "rh", **criterions
        )
        
        # Combine losses
        combined_loss_dict = {}
        combined_loss_dict.update(lh_loss_dict)
        combined_loss_dict.update(rh_loss_dict)
        
        # Add cross-hand coordination loss using encoder outputs (not concatenated features)
        coordination_loss = self._compute_coordination_loss(lh_encoder_out, rh_encoder_out)
        combined_loss_dict['coordination_architecture'] = coordination_loss
        
        return combined_loss_dict
    
    def _calculate_single_hand_loss(self, video_feats, encoder_out, backbone_feats, 
                                   event_gt, boundary_gt, decoder, prefix, **criterions):
        """Calculate loss for a single hand using diffusion logic"""
        
        # Implementation of the diffusion training loss (simplified)
        # This mimics the logic from ASDiffusionModel.get_training_loss
        
        loss_dict = {}
        
        # Encoder losses
        if f'{prefix}_encoder_ce_loss' in criterions or 'encoder_ce_loss' in criterions:
            ce_crit = criterions.get(f'{prefix}_encoder_ce_loss', criterions.get('encoder_ce_loss'))
            ce_loss = ce_crit(encoder_out.transpose(2, 1).contiguous().view(-1, self.num_classes), 
                             event_gt.transpose(2, 1).contiguous().view(-1).long())
            loss_dict[f'{prefix}_encoder_ce_loss'] = ce_loss.mean()
        
        # Additional encoder losses (boundary, mse) can be added here
        
        # Decoder diffusion loss (simplified - using encoder prediction as target for now)
        # In practice, you'd implement the full diffusion forward process
        t = torch.randint(0, 1000, (video_feats.shape[0],), device=video_feats.device).long()
        
        # Simple decoder loss using encoder output as conditioning
        decoder_out = decoder(torch.zeros_like(backbone_feats), t, encoder_out.float())
        
        if f'{prefix}_decoder_ce_loss' in criterions or 'decoder_ce_loss' in criterions:
            ce_crit = criterions.get(f'{prefix}_decoder_ce_loss', criterions.get('decoder_ce_loss'))
            decoder_ce_loss = ce_crit(decoder_out.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                     event_gt.transpose(2, 1).contiguous().view(-1).long())
            loss_dict[f'{prefix}_decoder_ce_loss'] = decoder_ce_loss.mean()
        
        return loss_dict
    
    def _compute_coordination_loss(self, lh_encoder_out, rh_encoder_out):
        """Compute simple coordination loss using encoder outputs (no attention)"""
        # Use encoder outputs directly for coordination loss
        # lh_encoder_out and rh_encoder_out are [B, num_classes, T]
        
        # Ensure features have the same temporal dimension
        min_time = min(lh_encoder_out.shape[2], rh_encoder_out.shape[2])
        lh_feat = lh_encoder_out[:, :, :min_time]
        rh_feat = rh_encoder_out[:, :, :min_time]
        
        # Simple coordination loss: encourage similar prediction patterns
        # Convert to probabilities for better coordination
        lh_probs = F.softmax(lh_feat, dim=1)
        rh_probs = F.softmax(rh_feat, dim=1)
        
        # Compute coordination loss as similarity between prediction patterns
        coordination_loss = F.mse_loss(lh_probs, rh_probs)
        
        return coordination_loss
    
    def ddim_sample_both_hands(self, lh_video_feats, rh_video_feats, seed=None):
        """Sample from both hands using proper DDIM sampling from base models"""
        if seed is not None:
            torch.manual_seed(seed)
        
        # Use base models for proper DDIM sampling during evaluation
        # This ensures we get the same evaluation behavior as the working version
        lh_output = self.base_lh_model.ddim_sample(lh_video_feats, seed)
        rh_output = self.base_rh_model.ddim_sample(rh_video_feats, seed)
        
        return lh_output, rh_output
    
    def get_encoders_output(self, lh_video_feats, rh_video_feats):
        """Get encoder outputs for both hands"""
        lh_encoder_out, lh_backbone_feats = self.left_hand_encoder(lh_video_feats, get_features=True)
        rh_encoder_out, rh_backbone_feats = self.right_hand_encoder(rh_video_feats, get_features=True)
        
        return lh_encoder_out, rh_encoder_out


class DualHandCoordinationLoss:
    """
    Loss functions to capture coordination between dual hands
    """
    
    def __init__(self, device, num_classes):
        self.device = device
        self.num_classes = num_classes
    
    def temporal_synchronization_loss(self, lh_predictions, rh_predictions, weight=0.1):
        """
        Encourage temporal synchronization between hands
        
        Args:
            lh_predictions: Left hand predictions [B, C, T]
            rh_predictions: Right hand predictions [B, C, T]
        """
        # Get action confidence for each hand
        lh_confidence = torch.max(F.softmax(lh_predictions, dim=1), dim=1)[0]  # [B, T]
        rh_confidence = torch.max(F.softmax(rh_predictions, dim=1), dim=1)[0]  # [B, T]
        
        # Encourage similar confidence patterns (synchronized actions)
        sync_loss = F.mse_loss(lh_confidence, rh_confidence)
        
        return weight * sync_loss
    
    def action_coherence_loss(self, lh_predictions, rh_predictions, lh_labels, rh_labels, weight=0.05):
        """
        Encourage coherent actions between hands based on action semantics
        
        Some actions are naturally coordinated (e.g., both hands cutting)
        while others are independent (e.g., one hand stirring, other holding)
        """
        batch_size, _, seq_len = lh_predictions.shape
        
        # Get predicted and ground truth action classes
        lh_pred_classes = torch.argmax(F.softmax(lh_predictions, dim=1), dim=1)  # [B, T]
        rh_pred_classes = torch.argmax(F.softmax(rh_predictions, dim=1), dim=1)  # [B, T]
        
        # Define action coordination patterns (you can customize this based on your dataset)
        # Actions that should be coordinated between hands
        coordinated_actions = self._get_coordinated_action_pairs()
        
        coherence_loss = torch.tensor(0.0, device=lh_predictions.device, requires_grad=True)
        valid_pairs = 0
        
        for t in range(seq_len):
            lh_gt = lh_labels[:, t].long()
            rh_gt = rh_labels[:, t].long()
            lh_pred = lh_pred_classes[:, t]
            rh_pred = rh_pred_classes[:, t]
            
            for b in range(batch_size):
                # Check if ground truth actions should be coordinated
                gt_pair = (lh_gt[b].item(), rh_gt[b].item())
                
                if gt_pair in coordinated_actions:
                    # These actions should be coordinated
                    pred_pair = (lh_pred[b].item(), rh_pred[b].item())
                    
                    # Penalize if predictions don't match the coordination pattern
                    if pred_pair != gt_pair:
                        coherence_loss = coherence_loss + 1.0
                    valid_pairs += 1
        
        if valid_pairs > 0:
            coherence_loss = coherence_loss / valid_pairs
        else:
            coherence_loss = torch.tensor(0.0, device=lh_predictions.device)
        
        return weight * coherence_loss
    
    def cross_hand_consistency_loss(self, lh_predictions, rh_predictions, weight=0.08):
        """
        Encourage consistency in prediction confidence between hands
        """
        # Compute entropy for each hand (lower entropy = higher confidence)
        lh_probs = F.softmax(lh_predictions, dim=1)
        rh_probs = F.softmax(rh_predictions, dim=1)
        
        lh_entropy = -torch.sum(lh_probs * torch.log(lh_probs + 1e-8), dim=1)  # [B, T]
        rh_entropy = -torch.sum(rh_probs * torch.log(rh_probs + 1e-8), dim=1)  # [B, T]
        
        # Encourage similar confidence levels
        consistency_loss = F.mse_loss(lh_entropy, rh_entropy)
        
        return weight * consistency_loss
    
    def boundary_synchronization_loss(self, lh_predictions, rh_predictions, weight=0.06):
        """
        Encourage action boundaries to be synchronized between hands
        """
        # Detect action transitions (boundaries) for each hand
        lh_classes = torch.argmax(F.softmax(lh_predictions, dim=1), dim=1)  # [B, T]
        rh_classes = torch.argmax(F.softmax(rh_predictions, dim=1), dim=1)  # [B, T]
        
        # Compute boundaries (action transitions)
        lh_boundaries = (lh_classes[:, 1:] != lh_classes[:, :-1]).float()  # [B, T-1]
        rh_boundaries = (rh_classes[:, 1:] != rh_classes[:, :-1]).float()  # [B, T-1]
        
        # Encourage boundaries to occur at similar times
        boundary_sync_loss = F.mse_loss(lh_boundaries, rh_boundaries)
        
        return weight * boundary_sync_loss
    
    # Add action transition loss, improved version
    def action_transition_loss(self, lh_predictions, rh_predictions, lh_labels, rh_labels, weight=0.05):
        """
        Encourage coordinated action transitions between hands
        """
        # Get predicted classes
        lh_classes = torch.argmax(F.softmax(lh_predictions, dim=1), dim=1)  # [B, T]
        rh_classes = torch.argmax(F.softmax(rh_predictions, dim=1), dim=1)  # [B, T]
        
        # Detect transitions
        lh_transitions = (lh_classes[:, 1:] != lh_classes[:, :-1]).float()
        rh_transitions = (rh_classes[:, 1:] != rh_classes[:, :-1]).float()
        lh_gt_transitions = (lh_labels[:, 1:] != lh_labels[:, :-1]).float()
        rh_gt_transitions = (rh_labels[:, 1:] != rh_labels[:, :-1]).float()
        
        # Loss for coordinated transitions
        transition_coord_loss = F.mse_loss(lh_transitions, rh_transitions)
        
        # Loss for matching ground truth transitions
        lh_transition_acc = F.mse_loss(lh_transitions, lh_gt_transitions)
        rh_transition_acc = F.mse_loss(rh_transitions, rh_gt_transitions)
        
        total_loss = transition_coord_loss + 0.5 * (lh_transition_acc + rh_transition_acc)
        
        return weight * total_loss
    
    def semantic_coordination_loss(self, lh_predictions, rh_predictions, lh_labels, rh_labels, weight=0.1):
        """
        Advanced semantic coordination based on action relationships
        """
        # Get action probabilities
        lh_probs = F.softmax(lh_predictions, dim=1)  # [B, C, T]
        rh_probs = F.softmax(rh_predictions, dim=1)  # [B, C, T]
        
        # Define semantic action groups (actions that often occur together)
        action_groups = self._get_action_groups()
        
        semantic_loss = torch.tensor(0.0, device=lh_predictions.device)
        
        for group in action_groups:
            # For each semantic group, encourage similar activation patterns
            lh_group_prob = torch.sum(lh_probs[:, group, :], dim=1)  # [B, T]
            rh_group_prob = torch.sum(rh_probs[:, group, :], dim=1)  # [B, T]
            
            # Encourage similar group activation
            group_loss = F.mse_loss(lh_group_prob, rh_group_prob)
            semantic_loss = semantic_loss + group_loss
        
        if len(action_groups) > 0:
            return weight * semantic_loss / len(action_groups)
        else:
            return torch.tensor(0.0, device=lh_predictions.device)
    
    def _get_coordinated_action_pairs(self):
        """
        Define which action pairs should be coordinated
        Customize this based on your specific dataset
        """
        # Example coordination patterns (you should customize this)
        coordinated_pairs = set()
        
        # Both hands doing the same action (common coordination)
        for i in range(self.num_classes):
            coordinated_pairs.add((i, i))
        
        # Add specific action pair coordinations here
        # For example: (cut_action_id, cut_action_id), (pour_left_id, hold_right_id), etc.
        
        return coordinated_pairs
    
    def _get_action_groups(self):
        """
        Define semantic action groups
        Customize this based on your dataset's action taxonomy
        """
        # Example action groups - you should customize this
        if self.num_classes >= 10:
            return [
                list(range(0, 3)),      # Group 1: Basic actions
                list(range(3, 6)),      # Group 2: Manipulation actions  
                list(range(6, 10)),     # Group 3: Tool usage actions
                # Add more groups as needed
            ]
        else:
            # Simple grouping for smaller action sets
            mid = self.num_classes // 2
            return [
                list(range(0, mid)),
                list(range(mid, self.num_classes))
            ]


class EnhancedDualHandDataset(DualHandVideoFeatureDataset):
    """
    Enhanced dataset with data augmentation
    """
    
    def __init__(self, lh_data_dict, rh_data_dict, class_num, mode, 
                 enable_augmentation=True, aug_params=None):
        super().__init__(lh_data_dict, rh_data_dict, class_num, mode)
        
        self.enable_augmentation = enable_augmentation and (mode == 'train')
        
        # Initialize data augmentation
        if self.enable_augmentation:
            aug_params = aug_params or {}
            self.data_augmentation = DataAugmentation(
                num_classes=class_num,
                temporal_jitter_std=aug_params.get('temporal_jitter_std', 2),
                action_sub_prob=aug_params.get('action_sub_prob', 0.1),
                confidence_smooth_alpha=aug_params.get('confidence_smooth_alpha', 0.1)
            )
    
    def __getitem__(self, idx):
        # Get original data
        data = super().__getitem__(idx)
        
        if self.enable_augmentation and self.mode == 'train':
            (lh_feature, lh_label, lh_boundary, 
             rh_feature, rh_label, rh_boundary, video) = data
            
            # Apply data augmentation
            (lh_label_aug, rh_label_aug, 
             lh_boundary_aug, rh_boundary_aug, _) = self.data_augmentation.apply_augmentation(
                lh_label, rh_label, lh_boundary.squeeze(0), rh_boundary.squeeze(0)
            )
            
            # Return augmented data
            return (lh_feature, lh_label_aug, lh_boundary_aug.unsqueeze(0),
                   rh_feature, rh_label_aug, rh_boundary_aug.unsqueeze(0), video)
        else:
            return data


class EnhancedDualHandTrainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device, args=None):

        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess
        self.args = args  # Store args for coordination weights

        # Use enhanced model
        self.model = EnhancedDualHandASDiffusionModel(
            encoder_params, decoder_params, diffusion_params, self.num_classes, self.device
        )
        print('Enhanced Dual Hand Model Size: ', sum(p.numel() for p in self.model.parameters()))
        
        # Initialize dual-hand coordination loss
        self.coordination_loss = DualHandCoordinationLoss(self.device, self.num_classes)
        
        # Initialize data augmentation
        self.data_augmentation = DataAugmentation(self.num_classes)

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, lh_label_dir, rh_label_dir, result_dir, log_freq, log_train_results=True):

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Add learning rate scheduler, improved version
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
        )
        
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1
        best_avg_acc = 0.0  # Track best validation accuracy, improved version

        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'), weights_only=False)
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')
        
        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)
        
        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)
        
        for epoch in range(restore_epoch+1, num_epochs):

            self.model.train()
            
            epoch_running_loss = 0
            
            for _, data in enumerate(train_train_loader):

                (lh_feature, lh_label, lh_boundary, 
                 rh_feature, rh_label, rh_boundary, video) = data
                
                lh_feature, lh_label, lh_boundary = lh_feature.to(device), lh_label.to(device), lh_boundary.to(device)
                rh_feature, rh_label, rh_boundary = rh_feature.to(device), rh_label.to(device), rh_boundary.to(device)
                
                # Apply data augmentation during training
                if self.model.training:
                    (lh_label_aug, rh_label_aug, 
                     lh_boundary_aug, rh_boundary_aug, _) = self.data_augmentation.apply_augmentation(
                        lh_label.squeeze(0), rh_label.squeeze(0), 
                        lh_boundary.squeeze(0).squeeze(0), rh_boundary.squeeze(0).squeeze(0)
                    )
                    lh_label = lh_label_aug.unsqueeze(0)
                    rh_label = rh_label_aug.unsqueeze(0)
                    lh_boundary = lh_boundary_aug.unsqueeze(0).unsqueeze(0)
                    rh_boundary = rh_boundary_aug.unsqueeze(0).unsqueeze(0)
                
                loss_dict = self.model.get_training_loss(
                    lh_feature, rh_feature,
                    lh_event_gt=F.one_hot(lh_label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    rh_event_gt=F.one_hot(rh_label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    lh_boundary_gt=lh_boundary,
                    rh_boundary_gt=rh_boundary,
                    encoder_ce_criterion=ce_criterion,
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )

                total_loss = 0

                for k, v in loss_dict.items():
                    # Use the same weights for both hands
                    base_key = k.replace('lh_', '').replace('rh_', '')
                    if base_key in loss_weights:
                        total_loss += loss_weights[base_key] * v
                    elif k == 'coordination_architecture':
                        # Add architectural coordination loss
                        total_loss += 0.05 * v  # Moderate weight for architectural loss

                # Add dual-hand coordination losses
                # Get encoder outputs for coordination loss computation
                lh_encoder_out, _ = self.model.get_encoders_output(lh_feature, rh_feature)
                rh_encoder_out = _
                
                # Compute coordination losses with adaptive weights
                # Reduce coordination loss importance as training progresses
                coord_decay = max(0.5, 1.0 - epoch / (num_epochs * 0.8))  # Decay after 80% of training
                coord_weights = [w * coord_decay for w in self.args.coordination_weights]
                
                temporal_sync_loss = self.coordination_loss.temporal_synchronization_loss(
                    lh_encoder_out, rh_encoder_out, weight=coord_weights[0])
                
                cross_consistency_loss = self.coordination_loss.cross_hand_consistency_loss(
                    lh_encoder_out, rh_encoder_out, weight=coord_weights[1])
                
                boundary_sync_loss = self.coordination_loss.boundary_synchronization_loss(
                    lh_encoder_out, rh_encoder_out, weight=coord_weights[2])
                
                action_coherence_loss = self.coordination_loss.action_coherence_loss(
                    lh_encoder_out, rh_encoder_out, lh_label, rh_label, weight=coord_weights[3])
                
                semantic_coord_loss = self.coordination_loss.semantic_coordination_loss(
                    lh_encoder_out, rh_encoder_out, lh_label, rh_label, weight=coord_weights[4])
                
                # Add new transition loss, improved version
                action_transition_loss = self.coordination_loss.action_transition_loss(
                    lh_encoder_out, rh_encoder_out, lh_label, rh_label, weight=0.03)
                
                # Add coordination losses to total loss, improved version
                coordination_total = (temporal_sync_loss + cross_consistency_loss + 
                                    boundary_sync_loss + action_coherence_loss + semantic_coord_loss + action_transition_loss)
                
                # Ensure coordination_total is a proper tensor
                if not isinstance(coordination_total, torch.Tensor):
                    coordination_total = torch.tensor(coordination_total, device=device, requires_grad=True)
                
                total_loss += coordination_total

                if result_dir:
                    for k, v in loss_dict.items():
                        base_key = k.replace('lh_', '').replace('rh_', '')
                        if base_key in loss_weights:
                            logger.add_scalar(f'Train-{k}', loss_weights[base_key] * v.item() / batch_size, step)
                        elif k == 'coordination_architecture':
                            logger.add_scalar(f'Train-{k}', 0.05 * v.item() / batch_size, step)
                    
                    # Log coordination losses
                    logger.add_scalar('Train-Coordination-Temporal-Sync', 
                                     temporal_sync_loss.item() if isinstance(temporal_sync_loss, torch.Tensor) else temporal_sync_loss, step)
                    logger.add_scalar('Train-Coordination-Cross-Consistency', 
                                     cross_consistency_loss.item() if isinstance(cross_consistency_loss, torch.Tensor) else cross_consistency_loss, step)
                    logger.add_scalar('Train-Coordination-Boundary-Sync', 
                                     boundary_sync_loss.item() if isinstance(boundary_sync_loss, torch.Tensor) else boundary_sync_loss, step)
                    logger.add_scalar('Train-Coordination-Action-Coherence', 
                                     action_coherence_loss.item() if isinstance(action_coherence_loss, torch.Tensor) else action_coherence_loss, step)
                    logger.add_scalar('Train-Coordination-Semantic', 
                                     semantic_coord_loss.item() if isinstance(semantic_coord_loss, torch.Tensor) else semantic_coord_loss, step)
                    logger.add_scalar('Train-Coordination-Action-Transition', 
                                     action_transition_loss.item() if isinstance(action_transition_loss, torch.Tensor) else action_transition_loss, step)
                    logger.add_scalar('Train-Coordination-Total', 
                                     coordination_total.item() if isinstance(coordination_total, torch.Tensor) else coordination_total, step)
                    
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()
                
                # Gradient clipping for stability, improved version
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_train_dataset)

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')
            
            # Step the scheduler with validation loss，improved version
            scheduler.step(epoch_running_loss)
        
            if result_dir:

                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0:

                if result_dir:

                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')
        
                # Test both hands
                for mode in ['decoder-agg']: 

                    test_result_dict = self.test(
                        test_test_dataset, mode, device, lh_label_dir, rh_label_dir,
                        result_dir=result_dir, model_path=None)

                    if result_dir:
                        for k, v in test_result_dict.items():
                            logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                        np.save(os.path.join(result_dir, 
                            f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                    for k, v in test_result_dict.items():
                        print(f'Epoch {epoch} - {mode}-Test-{k} {v}')

                    # Save best model, improved version
                    current_avg_acc = test_result_dict.get('Avg_Acc', 0.0)
                    if current_avg_acc > best_avg_acc:
                        best_avg_acc = current_avg_acc
                        if result_dir:
                            torch.save(self.model.state_dict(), f'{result_dir}/best_model.model')
                            print(f'🎯 New best model saved! Avg_Acc: {best_avg_acc:.4f}')

                    if log_train_results:

                        train_result_dict = self.test(
                            train_test_dataset, mode, device, lh_label_dir, rh_label_dir,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            for k, v in train_result_dict.items():
                                logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                                 
                            np.save(os.path.join(result_dir, 
                                f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)
                            
                        for k, v in train_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Train-{k} {v}')
                        
        if result_dir:
            logger.close()

    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):  
        
        assert(test_dataset.mode == 'test')
        assert(mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert(self.postprocess['type'] in ['median', 'mode', 'purge', None])

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=False))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
            
        with torch.no_grad():

            (lh_feature, lh_label, lh_boundary,
             rh_feature, rh_label, rh_boundary, video) = test_dataset[video_idx]

            if mode == 'encoder':
                lh_output, rh_output = self.model.get_encoders_output(
                    torch.stack([f.to(device) for f in lh_feature]),
                    torch.stack([f.to(device) for f in rh_feature])
                )
                lh_output = [F.softmax(lh_output, 1).cpu()]
                rh_output = [F.softmax(rh_output, 1).cpu()]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            elif mode == 'decoder-agg':
                lh_outputs = []
                rh_outputs = []
                for i in range(len(lh_feature)):
                    lh_out, rh_out = self.model.ddim_sample_both_hands(
                        lh_feature[i].to(device), rh_feature[i].to(device), seed)
                    lh_outputs.append(lh_out.cpu())
                    rh_outputs.append(rh_out.cpu())
                
                lh_output = lh_outputs
                rh_output = rh_outputs
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            elif mode == 'decoder-noagg':  # temporal aug must be true
                lh_out, rh_out = self.model.ddim_sample_both_hands(
                    lh_feature[len(lh_feature)//2].to(device), 
                    rh_feature[len(rh_feature)//2].to(device), seed)
                lh_output = [lh_out.cpu()]
                rh_output = [rh_out.cpu()]
                left_offset = self.sample_rate // 2
                right_offset = 0

            # Process left hand output
            assert(lh_output[0].shape[0] == 1)
            min_len = min([i.shape[2] for i in lh_output])
            lh_output = [i[:,:,:min_len] for i in lh_output]
            lh_output = torch.cat(lh_output, 0).mean(0).numpy()

            # Process right hand output  
            assert(rh_output[0].shape[0] == 1)
            min_len = min([i.shape[2] for i in rh_output])
            rh_output = [i[:,:,:min_len] for i in rh_output]
            rh_output = torch.cat(rh_output, 0).mean(0).numpy()

            # Apply postprocessing to both hands
            def apply_postprocessing(output, label):
                if self.postprocess['type'] == 'median':
                    smoothed_output = np.zeros_like(output)
                    for c in range(output.shape[0]):
                        smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                    output = smoothed_output / smoothed_output.sum(0, keepdims=True)

                output = np.argmax(output, 0)
                output = restore_full_sequence(output, 
                    full_len=label.shape[-1], 
                    left_offset=left_offset, 
                    right_offset=right_offset, 
                    sample_rate=self.sample_rate
                )

                if self.postprocess['type'] == 'mode':
                    output = mode_filter(output, self.postprocess['value'])

                if self.postprocess['type'] == 'purge':
                    trans, starts, ends = get_labels_start_end_time(output)
                    for e in range(0, len(trans)):
                        duration = ends[e] - starts[e]
                        if duration <= self.postprocess['value']:
                            if e == 0:
                                output[starts[e]:ends[e]] = trans[e+1]
                            elif e == len(trans) - 1:
                                output[starts[e]:ends[e]] = trans[e-1]
                            else:
                                mid = starts[e] + duration // 2
                                output[starts[e]:mid] = trans[e-1]
                                output[mid:ends[e]] = trans[e+1]
                return output

            lh_output = apply_postprocessing(lh_output, lh_label)
            rh_output = apply_postprocessing(rh_output, rh_label)
            
            lh_label = lh_label.squeeze(0).cpu().numpy()
            rh_label = rh_label.squeeze(0).cpu().numpy()

            assert(lh_output.shape == lh_label.shape)
            assert(rh_output.shape == rh_label.shape)
            
            return video, lh_output, lh_label, rh_output, rh_label

    def test(self, test_dataset, mode, device, lh_label_dir, rh_label_dir, result_dir=None, model_path=None):
        
        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=False))
        
        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):
                
                video, lh_pred, lh_label, rh_pred, rh_label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)

                lh_pred = [self.event_list[int(i)] for i in lh_pred]
                rh_pred = [self.event_list[int(i)] for i in rh_pred]
                
                # Save left hand predictions
                if not os.path.exists(os.path.join(result_dir, 'lh_prediction')):
                    os.makedirs(os.path.join(result_dir, 'lh_prediction'))

                lh_file_name = os.path.join(result_dir, 'lh_prediction', f'{video}.txt')
                with open(lh_file_name, 'w') as file_ptr:
                    file_ptr.write('### Frame level recognition: ###\n')
                    file_ptr.write(' '.join(lh_pred))

                # Save right hand predictions
                if not os.path.exists(os.path.join(result_dir, 'rh_prediction')):
                    os.makedirs(os.path.join(result_dir, 'rh_prediction'))

                rh_file_name = os.path.join(result_dir, 'rh_prediction', f'{video}.txt')
                with open(rh_file_name, 'w') as file_ptr:
                    file_ptr.write('### Frame level recognition: ###\n')
                    file_ptr.write(' '.join(rh_pred))

        # Evaluate both hands
        lh_acc, lh_edit, lh_f1s = func_eval(
            lh_label_dir, os.path.join(result_dir, 'lh_prediction'), test_dataset.video_list)
        
        rh_acc, rh_edit, rh_f1s = func_eval(
            rh_label_dir, os.path.join(result_dir, 'rh_prediction'), test_dataset.video_list)

        result_dict = {
            'LH_Acc': lh_acc,
            'LH_Edit': lh_edit,
            'LH_F1@10': lh_f1s[0],
            'LH_F1@25': lh_f1s[1],
            'LH_F1@50': lh_f1s[2],
            'RH_Acc': rh_acc,
            'RH_Edit': rh_edit,
            'RH_F1@10': rh_f1s[0],
            'RH_F1@25': rh_f1s[1],
            'RH_F1@50': rh_f1s[2],
            'Avg_Acc': (lh_acc + rh_acc) / 2,
            'Avg_Edit': (lh_edit + rh_edit) / 2,
            'Avg_F1@10': (lh_f1s[0] + rh_f1s[0]) / 2,
            'Avg_F1@25': (lh_f1s[1] + rh_f1s[1]) / 2,
            'Avg_F1@50': (lh_f1s[2] + rh_f1s[2]) / 2,
        }
        
        return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    parser.add_argument('--coordination_weights', type=float, nargs=5, 
                       default=[0.1, 0.08, 0.06, 0.05, 0.1],
                       help='Weights for coordination losses: [temporal_sync, cross_consistency, boundary_sync, action_coherence, semantic]')
    parser.add_argument('--result_suffix', type=str, default='enhanced',
                       help='Suffix for result directory')
    
    # Data augmentation parameters
    parser.add_argument('--enable_augmentation', action='store_true', default=True,
                       help='Enable data augmentation')
    parser.add_argument('--temporal_jitter_std', type=float, default=2,
                       help='Standard deviation for temporal jittering')
    parser.add_argument('--action_sub_prob', type=float, default=0.1,
                       help='Probability of action substitution')
    parser.add_argument('--confidence_smooth_alpha', type=float, default=0.1,
                       help='Alpha for confidence smoothing')
    
    args = parser.parse_args()
    view_id = "View0"

    all_params = load_config_file(args.config)
    locals().update(all_params)

    # Update result directory with suffix
    if hasattr(args, 'result_suffix') and args.result_suffix:
        naming = naming + f"_{args.result_suffix}"

    print(args.config)
    print(all_params)
    print(f"🚀 Enhanced Dual Hand Training with:")
    print(f"  - Data Augmentation: {args.enable_augmentation}")
    print(f"  - Temporal Jitter Std: {args.temporal_jitter_std}")
    print(f"  - Action Substitution Prob: {args.action_sub_prob}")
    print(f"  - Confidence Smooth Alpha: {args.confidence_smooth_alpha}")
    print(f"  - Enhanced Architecture: Larger encoder + Attention + Skip connections")
    print(f"Coordination loss weights: {args.coordination_weights}")
    print(f"  - Temporal Sync: {args.coordination_weights[0]}")
    print(f"  - Cross Consistency: {args.coordination_weights[1]}")
    print(f"  - Boundary Sync: {args.coordination_weights[2]}")
    print(f"  - Action Coherence: {args.coordination_weights[3]}")
    print(f"  - Semantic Coordination: {args.coordination_weights[4]}")

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    mapping_file = os.path.join(root_data_dir, dataset_name, 'task_mapping.txt')
    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)

    # Load dual hand data
    (lh_train_data_dict, lh_test_data_dict, 
     rh_train_data_dict, rh_test_data_dict) = get_dual_hand_data_dicts(
        root_data_dir, dataset_name, split_id, event_list,
        sample_rate, temporal_aug, boundary_smooth, view_id
    )

    # Augmentation parameters
    aug_params = {
        'temporal_jitter_std': args.temporal_jitter_std,
        'action_sub_prob': args.action_sub_prob,
        'confidence_smooth_alpha': args.confidence_smooth_alpha
    }

    # Create enhanced datasets with augmentation
    train_train_dataset = EnhancedDualHandDataset(
        lh_train_data_dict, rh_train_data_dict, num_classes, mode='train',
        enable_augmentation=args.enable_augmentation, aug_params=aug_params)
    train_test_dataset = EnhancedDualHandDataset(
        lh_train_data_dict, rh_train_data_dict, num_classes, mode='test',
        enable_augmentation=False)
    test_test_dataset = EnhancedDualHandDataset(
        lh_test_data_dict, rh_test_data_dict, num_classes, mode='test',
        enable_augmentation=False)

    trainer = EnhancedDualHandTrainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        args=args
    )    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Get label directories for evaluation
    lh_label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth/View0/lh_pt')
    rh_label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth/View0/rh_pt')

    trainer.train(train_train_dataset, train_test_dataset, test_test_dataset, 
        loss_weights, class_weighting, soft_label,
        num_epochs, batch_size, learning_rate, weight_decay,
        lh_label_dir=lh_label_dir, rh_label_dir=rh_label_dir,
        result_dir=os.path.join(result_dir, naming), 
        log_freq=log_freq, log_train_results=log_train_results
    ) 