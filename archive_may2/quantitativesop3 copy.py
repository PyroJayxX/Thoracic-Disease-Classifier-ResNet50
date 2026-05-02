"""
Compare Attention Maps: Custom Spatial Layer vs Baseline ResNet50
Calculates average IoU score on test set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import timm
import cv2

# ============================================================================
# SPATIAL ATTENTION MODULE (Your Custom Layer)
# ============================================================================

class SpatialAttentionModule(nn.Module):
    """Learnable spatial attention module"""
    def __init__(self, in_channels=2048, reduction=8):
        super(SpatialAttentionModule, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels // reduction, 
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels // reduction)
        
        self.conv_out = nn.Conv2d(in_channels // reduction, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        att = self.conv1(x)
        att = self.bn1(att)
        att = self.relu(att)
        
        att = self.conv2(att)
        att = self.bn2(att)
        att = self.relu(att)
        
        att = self.conv_out(att)
        attention = self.sigmoid(att)
        
        weighted_features = x * attention
        return attention, weighted_features

# ============================================================================
# CUSTOM MODEL WITH SPATIAL ATTENTION
# ============================================================================

class WSODModelWithAttention(nn.Module):
    def __init__(self, base_model, num_classes=2, freeze_backbone=True):
        super(WSODModelWithAttention, self).__init__()
        
        if freeze_backbone:
            for param in base_model.parameters():
                param.requires_grad = False
        
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = list(base_model.children())[-2]
        self.spatial_attention = SpatialAttentionModule(in_channels=2048)
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x, return_attention=False):
        features = self.features(x)
        attention_map, weighted_features = self.spatial_attention(features)
        pooled = self.avgpool(weighted_features)
        pooled = pooled.flatten(1)
        output = self.fc(pooled)
        
        if return_attention:
            return output, attention_map
        return output
    
    def get_attention_map(self, x):
        """Get spatial attention map for visualization"""
        with torch.no_grad():
            features = self.features(x)
            attention_map, _ = self.spatial_attention(features)
        return attention_map

# ============================================================================
# BASELINE RESNET50 MODEL
# ============================================================================

class BaselineResNet50(nn.Module):
    """Standard ResNet50 with GradCAM-style attention extraction"""
    def __init__(self, num_classes=2):
        super(BaselineResNet50, self).__init__()
        
        # Load pretrained ResNet50
        base_model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
        
        # Extract components
        self.features = nn.Sequential(*list(base_model.children())[:-2])  # Up to layer4
        self.avgpool = list(base_model.children())[-2]
        self.fc = list(base_model.children())[-1]
        
        # For storing gradients and activations
        self.gradients = None
        self.activations = None
        
    def forward(self, x, return_attention=False):
        # Extract features
        features = self.features(x)  # [B, 2048, 7, 7]
        
        if return_attention and self.training == False:
            # Register hook to capture gradients
            features.register_hook(self.save_gradient)
            self.activations = features
        
        # Pool and classify
        pooled = self.avgpool(features)
        pooled = pooled.flatten(1)
        output = self.fc(pooled)
        
        if return_attention:
            # Generate attention map using class activation
            attention_map = self.get_attention_map_from_features(features, output)
            return output, attention_map
        
        return output
    
    def save_gradient(self, grad):
        """Hook to save gradients"""
        self.gradients = grad
    
    def get_attention_map_from_features(self, features, output):
        """
        Generate attention map from features using class weights
        Similar to CAM (Class Activation Mapping)
        """
        with torch.no_grad():
            # Get weights for the predicted class
            batch_size = features.size(0)
            attention_maps = []
            
            for i in range(batch_size):
                # Get predicted class
                pred_class = output[i].argmax().item()
                
                # Get FC weights for this class
                weights = self.fc.weight[pred_class]  # [2048]
                
                # Compute weighted sum of feature maps
                feat = features[i]  # [2048, 7, 7]
                cam = torch.sum(weights.view(-1, 1, 1) * feat, dim=0)  # [7, 7]
                
                # Apply ReLU and normalize
                cam = F.relu(cam)
                cam = cam - cam.min()
                cam_max = cam.max()
                if cam_max > 0:
                    cam = cam / cam_max
                
                attention_maps.append(cam)
            
            # Stack and add channel dimension
            attention_map = torch.stack(attention_maps).unsqueeze(1)  # [B, 1, 7, 7]
            
        return attention_map
    
    def get_attention_map(self, x):
        """Get attention map for a given input"""
        with torch.no_grad():
            output, attention_map = self.forward(x, return_attention=True)
        return attention_map

# ============================================================================
# IoU CALCULATION UTILITIES (Updated for BBox Ground Truth)
# ============================================================================

def bbox_to_mask(bboxes, image_size=(224, 224), normalized=True):
    """
    Convert bounding boxes to binary masks
    
    Args:
        bboxes: list of dicts/tensors with bbox info OR single tensor [N, 4] with format [x, y, w, h]
        image_size: (H, W) tuple for output mask size
        normalized: if True, bbox coordinates are in [0, 1] range and need to be scaled
    
    Returns:
        mask: [H, W] binary mask with 1s inside any bbox, 0s outside
    """
    mask = torch.zeros(image_size, dtype=torch.float32)
    
    # Handle different bbox formats
    if isinstance(bboxes, torch.Tensor):
        # Tensor format: [N, 4] or [4] with [x, y, width, height]
        if bboxes.dim() == 1:
            bboxes = bboxes.unsqueeze(0)  # Make it [1, 4]
        
        for i in range(bboxes.size(0)):
            bbox = bboxes[i].cpu() if bboxes[i].is_cuda else bboxes[i]
            x, y, w, h = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
            
            # Scale from normalized [0, 1] to pixel coordinates if needed
            if normalized:
                x = x * image_size[1]
                y = y * image_size[0]
                w = w * image_size[1]
                h = h * image_size[0]
            
            if w > 0 and h > 0:  # Valid bbox
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w)
                y2 = int(y + h)
                
                # Clip to image bounds
                x1 = max(0, min(x1, image_size[1]))
                y1 = max(0, min(y1, image_size[0]))
                x2 = max(0, min(x2, image_size[1]))
                y2 = max(0, min(y2, image_size[0]))
                
                mask[y1:y2, x1:x2] = 1.0
    
    elif isinstance(bboxes, list):
        for bbox in bboxes:
            if isinstance(bbox, dict):
                # Dictionary format
                x = bbox.get('x', 0)
                y = bbox.get('y', 0)
                w = bbox.get('width', 0)
                h = bbox.get('height', 0)
                
                if normalized:
                    x = x * image_size[1]
                    y = y * image_size[0]
                    w = w * image_size[1]
                    h = h * image_size[0]
                
                if w > 0 and h > 0:
                    x1 = int(x)
                    y1 = int(y)
                    x2 = int(x + w)
                    y2 = int(y + h)
                    
                    # Clip to image bounds
                    x1 = max(0, min(x1, image_size[1]))
                    y1 = max(0, min(y1, image_size[0]))
                    x2 = max(0, min(x2, image_size[1]))
                    y2 = max(0, min(y2, image_size[0]))
                    
                    mask[y1:y2, x1:x2] = 1.0
            elif isinstance(bbox, torch.Tensor):
                # Tensor in list
                bbox = bbox.cpu() if bbox.is_cuda else bbox
                x, y, w, h = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
                
                if normalized:
                    x = x * image_size[1]
                    y = y * image_size[0]
                    w = w * image_size[1]
                    h = h * image_size[0]
                
                if w > 0 and h > 0:
                    x1 = int(x)
                    y1 = int(y)
                    x2 = int(x + w)
                    y2 = int(y + h)
                    
                    # Clip to image bounds
                    x1 = max(0, min(x1, image_size[1]))
                    y1 = max(0, min(y1, image_size[0]))
                    x2 = max(0, min(x2, image_size[1]))
                    y2 = max(0, min(y2, image_size[0]))
                    
                    mask[y1:y2, x1:x2] = 1.0
    
    return mask


def calculate_iou_with_bbox(attention_map, bboxes, image_size=(224, 224), threshold=0.5, normalized=True):
    """
    Calculate IoU between attention map and ground truth bboxes
    
    Args:
        attention_map: [B, 1, H, W] or [B, H, W] attention maps
        bboxes: tensor [B, 4] or list of tensors/dicts with bbox info [x, y, w, h]
        image_size: original image size for bbox coordinates
        threshold: threshold for binarizing attention map
        normalized: if True, bbox coordinates are normalized to [0, 1]
    
    Returns:
        iou_scores: [B] IoU score for each sample
        has_nodule: [B] boolean array indicating if sample has nodules
    """
    # Ensure correct shape
    if attention_map.dim() == 4:
        attention_map = attention_map.squeeze(1)  # [B, H, W]
    
    batch_size = attention_map.size(0)
    att_h, att_w = attention_map.shape[-2:]
    
    # Upscale attention to image size if needed
    if (att_h, att_w) != image_size:
        attention_map = F.interpolate(
            attention_map.unsqueeze(1),
            size=image_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
    
    # Binarize attention maps
    binary_attention = (attention_map > threshold).float()
    
    iou_scores = []
    has_nodule = []
    
    for i in range(batch_size):
        # Get bbox for this sample
        if isinstance(bboxes, torch.Tensor):
            # Tensor format [B, 4]
            if bboxes.dim() == 2:
                bbox = bboxes[i]  # [4]
            else:
                bbox = bboxes  # Single bbox [4]
        elif isinstance(bboxes, (list, tuple)):
            # List of bboxes
            bbox = bboxes[i]
        else:
            bbox = bboxes
        
        # Create ground truth mask from bbox
        gt_mask = bbox_to_mask(bbox, image_size, normalized=normalized).to(attention_map.device)
        
        # Check if this sample has nodules
        has_bbox = gt_mask.sum() > 0
        has_nodule.append(bool(has_bbox.item()))  # Convert to Python bool
        
        if not has_bbox:
            # No nodule: IoU is N/A, we'll use -1 to indicate this
            iou_scores.append(-1.0)
            continue
        
        # Calculate IoU
        att_mask = binary_attention[i]
        intersection = (att_mask * gt_mask).sum()
        union = (att_mask + gt_mask).clamp(0, 1).sum()
        
        if union > 0:
            iou = (intersection / union).item()
        else:
            iou = 0.0
        
        iou_scores.append(iou)
    
    return np.array(iou_scores), np.array(has_nodule)

def calculate_iou(attention_map1, attention_map2, threshold=0.5):
    """
    Calculate IoU between two attention maps (kept for backward compatibility)
    
    Args:
        attention_map1: [B, 1, H, W] or [B, H, W]
        attention_map2: [B, 1, H, W] or [B, H, W]
        threshold: threshold for binarization
    
    Returns:
        iou_scores: [B] IoU score for each sample in batch
    """
    # Ensure same shape
    if attention_map1.dim() == 4:
        attention_map1 = attention_map1.squeeze(1)
    if attention_map2.dim() == 4:
        attention_map2 = attention_map2.squeeze(1)
    
    # Resize to same size if needed
    if attention_map1.shape != attention_map2.shape:
        attention_map2 = F.interpolate(
            attention_map2.unsqueeze(1), 
            size=attention_map1.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
    
    # Binarize attention maps
    binary_map1 = (attention_map1 > threshold).float()
    binary_map2 = (attention_map2 > threshold).float()
    
    # Calculate IoU for each sample in batch
    batch_size = binary_map1.size(0)
    iou_scores = []
    
    for i in range(batch_size):
        map1 = binary_map1[i]
        map2 = binary_map2[i]
        
        # Intersection and Union
        intersection = (map1 * map2).sum()
        union = (map1 + map2).clamp(0, 1).sum()
        
        # Avoid division by zero
        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(0.0)
        
        iou_scores.append(iou.item())
    
    return np.array(iou_scores)

def upscale_attention_map(attention_map, target_size=(224, 224)):
    """
    Upscale attention map to target size for visualization
    
    Args:
        attention_map: [B, 1, H, W] tensor
        target_size: (H, W) tuple
    
    Returns:
        upscaled_map: [B, 1, H, W] tensor at target size
    """
    return F.interpolate(
        attention_map,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )

# ============================================================================
# MAIN COMPARISON FUNCTION
# ============================================================================

def compare_attention_maps(custom_model_path, baseline_model_path, 
                          test_loader, device, threshold=0.5, use_bbox=True):
    """
    Compare attention maps between custom and baseline models against ground truth bboxes
    
    Args:
        custom_model_path: path to custom model checkpoint
        baseline_model_path: path to baseline model checkpoint
        test_loader: DataLoader for test set
        device: torch device
        threshold: threshold for binarization
        use_bbox: if True, compare against ground truth bboxes; if False, compare models to each other
    
    Returns:
        results: dict with IoU scores and statistics
    """
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    # Load custom model checkpoint first to inspect structure
    custom_checkpoint = torch.load(custom_model_path, map_location=device)
    state_dict = custom_checkpoint['model_state_dict']
    
    # Detect the architecture from checkpoint
    if 'base_model.conv1.weight' in state_dict and 'target_layer.conv1.weight' in state_dict:
        print("  Detected checkpoint with 'base_model' and 'target_layer' structure")
        
        # Inspect the actual architecture from checkpoint
        conv1_shape = state_dict['target_layer.conv1.weight'].shape
        conv2_shape = state_dict['target_layer.conv2.weight'].shape
        has_conv3 = 'target_layer.conv3.weight' in state_dict
        
        if has_conv3:
            conv3_shape = state_dict['target_layer.conv3.weight'].shape
            print(f"  Target layer architecture:")
            print(f"    conv1: {conv1_shape}")
            print(f"    conv2: {conv2_shape}")
            print(f"    conv3: {conv3_shape}")
        
        # Create custom target layer architecture matching checkpoint exactly
        class CustomTargetLayer(nn.Module):
            def __init__(self):
                super(CustomTargetLayer, self).__init__()
                
                # Match the SpatialAttentionModule architecture from guidedv3
                in_channels = 2048
                reduction = 8
                
                self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
                self.bn1 = nn.BatchNorm2d(in_channels // reduction)
                self.relu = nn.ReLU(inplace=True)
                
                self.conv2 = nn.Conv2d(in_channels // reduction, in_channels // reduction, 
                                    kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(in_channels // reduction)
                
                # This is the KEY layer that outputs 1-channel attention
                self.conv_out = nn.Conv2d(in_channels // reduction, 1, kernel_size=1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                # Generate spatial attention (exact same as SpatialAttentionModule)
                att = self.conv1(x)
                att = self.bn1(att)
                att = self.relu(att)
                
                att = self.conv2(att)
                att = self.bn2(att)
                att = self.relu(att)
                
                # Output attention map directly (1 channel)
                att = self.conv_out(att)
                attention = self.sigmoid(att)  # [B, 1, 7, 7]
                
                # Apply attention to original features
                weighted_features = x * attention
                return attention, weighted_features
        
        # Create wrapper with correct architecture
        class CustomModelWrapper(nn.Module):
            def __init__(self):
                super(CustomModelWrapper, self).__init__()
                self.base_model = timm.create_model('resnet50', pretrained=False, num_classes=2)
                self.target_layer = CustomTargetLayer()
                
            def forward(self, x, return_attention=False):
                # Extract features
                features = nn.Sequential(*list(self.base_model.children())[:-2])(x)
                attention_map, weighted_features = self.target_layer(features)
                
                # Pool and classify
                avgpool = list(self.base_model.children())[-2]
                pooled = avgpool(weighted_features)
                pooled = pooled.flatten(1)
                output = self.base_model.fc(pooled)
                
                if return_attention:
                    return output, attention_map
                return output
            
            def get_attention_map(self, x):
                with torch.no_grad():
                    features = nn.Sequential(*list(self.base_model.children())[:-2])(x)
                    attention_map, _ = self.target_layer(features)
                return attention_map
        
        custom_model = CustomModelWrapper().to(device)
        custom_model.load_state_dict(state_dict)
        
    else:
        # Standard structure with 'features' and 'spatial_attention'
        base_model = timm.create_model('resnet50', pretrained=True, num_classes=2)
        custom_model = WSODModelWithAttention(base_model, num_classes=2, freeze_backbone=True).to(device)
        custom_model.load_state_dict(state_dict)
    
    custom_model.eval()
    print(f"✓ Loaded custom model from {custom_model_path}")
    print(f"  Epoch: {custom_checkpoint.get('epoch', 'unknown')}")
    
    # Load baseline model
    baseline_checkpoint = torch.load(baseline_model_path, map_location=device)
    baseline_state_dict = baseline_checkpoint['model_state_dict']
    
    # Check baseline model structure
    if 'conv1.weight' in baseline_state_dict:
        # Standard ResNet50 structure, load directly into timm model
        print("  Detected standard ResNet50 structure for baseline")
        
        # Create wrapper for standard ResNet50
        class BaselineModelWrapper(nn.Module):
            def __init__(self):
                super(BaselineModelWrapper, self).__init__()
                self.model = timm.create_model('resnet50', pretrained=False, num_classes=2)
                
            def forward(self, x, return_attention=False):
                if return_attention:
                    # Get features before final pooling
                    features = None
                    def hook_fn(module, input, output):
                        nonlocal features
                        features = output
                    
                    # Register hook on layer4 (last conv layer)
                    handle = self.model.layer4.register_forward_hook(hook_fn)
                    output = self.model(x)
                    handle.remove()
                    
                    # Generate CAM-style attention
                    attention_map = self.get_cam_attention(features, output)
                    return output, attention_map
                else:
                    return self.model(x)
            
            def get_cam_attention(self, features, output):
                """Generate Class Activation Map style attention"""
                with torch.no_grad():
                    batch_size = features.size(0)
                    attention_maps = []
                    
                    for i in range(batch_size):
                        # Get predicted class
                        pred_class = output[i].argmax().item()
                        
                        # Get FC weights for this class
                        weights = self.model.fc.weight[pred_class]  # [2048]
                        
                        # Compute weighted sum of feature maps
                        feat = features[i]  # [2048, 7, 7]
                        cam = torch.sum(weights.view(-1, 1, 1) * feat, dim=0)  # [7, 7]
                        
                        # Apply ReLU and normalize
                        cam = F.relu(cam)
                        cam = cam - cam.min()
                        cam_max = cam.max()
                        if cam_max > 0:
                            cam = cam / cam_max
                        
                        attention_maps.append(cam)
                    
                    attention_map = torch.stack(attention_maps).unsqueeze(1)  # [B, 1, 7, 7]
                
                return attention_map
            
            def get_attention_map(self, x):
                """Get attention map for visualization"""
                with torch.no_grad():
                    output, attention_map = self.forward(x, return_attention=True)
                return attention_map
        
        baseline_model = BaselineModelWrapper().to(device)
        baseline_model.model.load_state_dict(baseline_state_dict)
    else:
        # Custom structure with features wrapper
        baseline_model = BaselineResNet50(num_classes=2).to(device)
        baseline_model.load_state_dict(baseline_state_dict)
    
    baseline_model.eval()
    print(f"✓ Loaded baseline model from {baseline_model_path}")
    print(f"  Epoch: {baseline_checkpoint.get('epoch', 'unknown')}")
    
    # Storage for IoU scores
    custom_iou_scores = []
    baseline_iou_scores = []
    all_labels = []
    all_has_nodule = []
    
    print("\n" + "="*80)
    if use_bbox:
        print("COMPUTING IoU SCORES AGAINST GROUND TRUTH BBOXES")
    else:
        print("COMPUTING IoU SCORES BETWEEN MODELS")
    print("="*80)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
            # Unpack batch
            if len(batch) == 3:
                images, labels, bboxes = batch  # With bboxes
            else:
                images, labels = batch
                bboxes = None
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Get attention maps from both models
            custom_attention = custom_model.get_attention_map(images)  # [B, 1, 7, 7]
            baseline_attention = baseline_model.get_attention_map(images)  # [B, 1, 7, 7]
            
            if use_bbox and bboxes is not None:
                # Calculate IoU against ground truth bboxes (normalized coordinates)
                custom_iou, has_nodule = calculate_iou_with_bbox(
                    custom_attention, bboxes, image_size=(224, 224), 
                    threshold=threshold, normalized=True
                )
                baseline_iou, _ = calculate_iou_with_bbox(
                    baseline_attention, bboxes, image_size=(224, 224), 
                    threshold=threshold, normalized=True
                )
                
                custom_iou_scores.extend(custom_iou)
                baseline_iou_scores.extend(baseline_iou)
                all_has_nodule.extend(has_nodule)
            else:
                # Calculate IoU between the two models
                batch_iou = calculate_iou(custom_attention, baseline_attention, threshold)
                custom_iou_scores.extend(batch_iou)
            
            # Store labels
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    custom_iou_scores = np.array(custom_iou_scores)
    baseline_iou_scores = np.array(baseline_iou_scores) if use_bbox else None
    all_labels = np.array(all_labels)
    all_has_nodule = np.array(all_has_nodule) if use_bbox else None
    
    # Calculate statistics
    if use_bbox:
        # Filter out samples without nodules (IoU = -1)
        nodule_mask = custom_iou_scores >= 0
        custom_iou_nodules = custom_iou_scores[nodule_mask]
        baseline_iou_nodules = baseline_iou_scores[nodule_mask]
        
        results = {
            'custom_iou_all': custom_iou_scores,
            'baseline_iou_all': baseline_iou_scores,
            'custom_iou_nodules': custom_iou_nodules,
            'baseline_iou_nodules': baseline_iou_nodules,
            'all_labels': all_labels,
            'has_nodule': all_has_nodule,
            'threshold': threshold,
            'num_nodule_samples': nodule_mask.sum(),
            'num_normal_samples': (~nodule_mask).sum(),
            # Custom model stats (only nodules)
            'custom_mean_iou': np.mean(custom_iou_nodules),
            'custom_std_iou': np.std(custom_iou_nodules),
            'custom_median_iou': np.median(custom_iou_nodules),
            'custom_min_iou': np.min(custom_iou_nodules),
            'custom_max_iou': np.max(custom_iou_nodules),
            # Baseline model stats (only nodules)
            'baseline_mean_iou': np.mean(baseline_iou_nodules),
            'baseline_std_iou': np.std(baseline_iou_nodules),
            'baseline_median_iou': np.median(baseline_iou_nodules),
            'baseline_min_iou': np.min(baseline_iou_nodules),
            'baseline_max_iou': np.max(baseline_iou_nodules),
        }
    else:
        results = {
            'all_iou_scores': custom_iou_scores,
            'all_labels': all_labels,
            'mean_iou': np.mean(custom_iou_scores),
            'std_iou': np.std(custom_iou_scores),
            'median_iou': np.median(custom_iou_scores),
            'min_iou': np.min(custom_iou_scores),
            'max_iou': np.max(custom_iou_scores),
            'threshold': threshold
        }
        
        # Calculate per-class IoU
        for label in np.unique(all_labels):
            mask = all_labels == label
            class_iou = custom_iou_scores[mask]
            label_name = "Nodule" if label == 1 else "Normal"
            results[f'mean_iou_class_{label_name}'] = np.mean(class_iou)
            results[f'std_iou_class_{label_name}'] = np.std(class_iou)
            results[f'count_class_{label_name}'] = len(class_iou)
    
    return results, custom_model, baseline_model

# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================

def visualize_attention_comparison(custom_model, baseline_model, test_loader, 
                                   device, num_samples=5, save_path='attention_comparison.png'):
    """
    Visualize attention maps from both models side by side
    """
    import matplotlib.pyplot as plt
    
    custom_model.eval()
    baseline_model.eval()
    
    # Get a batch
    batch = next(iter(test_loader))
    if len(batch) == 3:
        images, labels, _ = batch
    else:
        images, labels = batch
    
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    with torch.no_grad():
        custom_attention = custom_model.get_attention_map(images)
        baseline_attention = baseline_model.get_attention_map(images)
        
        # Upscale to image size
        custom_attention = upscale_attention_map(custom_attention, (224, 224))
        baseline_attention = upscale_attention_map(baseline_attention, (224, 224))
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i in range(num_samples):
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        custom_att = custom_attention[i, 0].cpu().numpy()
        baseline_att = baseline_attention[i, 0].cpu().numpy()
        
        # Calculate IoU for this sample
        iou = calculate_iou(
            custom_attention[i:i+1],
            baseline_attention[i:i+1]
        )[0]
        
        label_name = "Nodule" if labels[i] == 1 else "Normal"
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original\n{label_name}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(custom_att, cmap='jet')
        axes[i, 1].set_title('Custom Attention')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(baseline_att, cmap='jet')
        axes[i, 2].set_title('Baseline Attention')
        axes[i, 2].axis('off')
        
        # Overlay comparison
        axes[i, 3].imshow(img)
        axes[i, 3].imshow(custom_att, cmap='jet', alpha=0.5)
        axes[i, 3].set_title(f'Custom Overlay\nIoU: {iou:.3f}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {save_path}")
    plt.close()

# ============================================================================
# HELPER FUNCTION TO PRINT RESULTS
# ============================================================================

def print_iou_results(results):
    """Pretty print IoU comparison results"""
    print("\n" + "="*80)
    print("IoU COMPARISON RESULTS (vs Ground Truth BBoxes)")
    print("="*80)
    print(f"Threshold: {results['threshold']}")
    print(f"Total samples: {results.get('num_nodule_samples', 0) + results.get('num_normal_samples', 0)}")
    print(f"Samples with nodules: {results.get('num_nodule_samples', 'N/A')}")
    print(f"Samples without nodules: {results.get('num_normal_samples', 'N/A')}")
    
    if 'custom_mean_iou' in results:
        print(f"\n{'='*80}")
        print("CUSTOM MODEL (Spatial Attention Layer)")
        print(f"{'='*80}")
        print(f"  Mean IoU:   {results['custom_mean_iou']:.4f}")
        print(f"  Std IoU:    {results['custom_std_iou']:.4f}")
        print(f"  Median IoU: {results['custom_median_iou']:.4f}")
        print(f"  Min IoU:    {results['custom_min_iou']:.4f}")
        print(f"  Max IoU:    {results['custom_max_iou']:.4f}")
        
        print(f"\n{'='*80}")
        print("BASELINE MODEL (ResNet50 CAM)")
        print(f"{'='*80}")
        print(f"  Mean IoU:   {results['baseline_mean_iou']:.4f}")
        print(f"  Std IoU:    {results['baseline_std_iou']:.4f}")
        print(f"  Median IoU: {results['baseline_median_iou']:.4f}")
        print(f"  Min IoU:    {results['baseline_min_iou']:.4f}")
        print(f"  Max IoU:    {results['baseline_max_iou']:.4f}")
        
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        improvement = results['custom_mean_iou'] - results['baseline_mean_iou']
        print(f"  Custom vs Baseline (Mean IoU): {improvement:+.4f}")
        if improvement > 0:
            print(f"  ✓ Custom model attention is BETTER by {abs(improvement)*100:.2f}%")
        else:
            print(f"  ✗ Baseline model attention is BETTER by {abs(improvement)*100:.2f}%")
    else:
        # Old format (model comparison, not bbox)
        print(f"\nOverall Statistics:")
        print(f"  Mean IoU:   {results['mean_iou']:.4f}")
        print(f"  Std IoU:    {results['std_iou']:.4f}")
        print(f"  Median IoU: {results['median_iou']:.4f}")
        print(f"  Min IoU:    {results['min_iou']:.4f}")
        print(f"  Max IoU:    {results['max_iou']:.4f}")
    
    print("="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    BATCH_SIZE = 16
    IMAGE_SIZE = 224
    NUM_WORKERS = 4
    THRESHOLD = 0.5  # Threshold for binarizing attention maps
    
    # Paths
    CUSTOM_MODEL_PATH = './best_wsod_resnet50v2.pth'
    BASELINE_MODEL_PATH = './resnet-50_binary.pth'
    
    split_base_path = "./dataset_nodule21/cxr_images/proccessed_data/split_data"
    test_images_path = f"{split_base_path}/test/images"
    test_csv_path = f"{split_base_path}/test/metadata_test.csv"
    
    # Load test data
    print("\nLoading test dataset...")
    test_df = pd.read_csv(test_csv_path)
    print(f"Test samples: {len(test_df)}")
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    from NoduleDS import NoduleDataset
    test_dataset = NoduleDataset(test_df, test_images_path, 
                                 transform=val_transform, return_bbox=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=NUM_WORKERS, 
                            pin_memory=True)
    
    # Run comparison against ground truth bboxes
    results, custom_model, baseline_model = compare_attention_maps(
        CUSTOM_MODEL_PATH,
        BASELINE_MODEL_PATH,
        test_loader,
        device,
        threshold=THRESHOLD,
        use_bbox=True  # Compare against ground truth bboxes
    )
    
    # Print results
    print_iou_results(results)
    
    # Create visualizations
    print("\nGenerating comparison visualizations...")
    visualize_attention_comparison(
        custom_model, 
        baseline_model, 
        test_loader, 
        device,
        num_samples=5,
        save_path='attention_comparison.png'
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)