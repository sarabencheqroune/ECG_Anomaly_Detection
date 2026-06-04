"""
1D-CNN classifier for ECG beat classification
Architecture based on the screenshot: Conv blocks with pooling and dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class ECG1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for ECG beat classification
    
    Architecture:
    - Conv block 1: Conv1d(1→32, k=5) → BN → ReLU → MaxPool(2) → Dropout(0.2)
    - Conv block 2: Conv1d(32→64, k=5) → BN → ReLU → MaxPool(2) → Dropout(0.2)
    - Conv block 3: Conv1d(64→128, k=3) → BN → ReLU → MaxPool(2) → Dropout(0.2)
    - Global avg pooling → Dense(128→64) → ReLU → Softmax(5)
    
    Input shape: (batch, channels=1, sequence_length=187)
    Output shape: (batch, num_classes=5)
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 input_length: int = 187,
                 num_classes: int = 5,
                 conv_channels: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [5, 5, 3],
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True,
                 use_dropout: bool = True):
        """
        Initialize 1D-CNN classifier
        
        Args:
            input_channels: Number of input channels (1 for single lead)
            input_length: Length of input sequence
            num_classes: Number of output classes
            conv_channels: List of channels for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout_rate: Dropout probability
            use_batch_norm: Use batch normalization
            use_dropout: Use dropout layers
        """
        super(ECG1DCNN, self).__init__()
        
        self.input_channels = input_channels
        self.input_length = input_length
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        
        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        
        in_channels = input_channels
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            block = self._make_conv_block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                use_batch_norm=use_batch_norm,
                use_dropout=use_dropout,
                dropout_rate=dropout_rate
            )
            self.conv_blocks.append(block)
            in_channels = out_channels
            
        # Calculate output dimension after convolutions
        self.feature_dim = self._calculate_output_dim(input_length, conv_channels, kernel_sizes)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(64, num_classes)
        )
        
        # Weight initialization
        self._initialize_weights()
        
    def _make_conv_block(self, 
                        in_channels: int, 
                        out_channels: int, 
                        kernel_size: int,
                        use_batch_norm: bool = True,
                        use_dropout: bool = True,
                        dropout_rate: float = 0.2) -> nn.Sequential:
        """Create a single convolutional block"""
        padding = kernel_size // 2  # Same padding
        
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
            
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool1d(kernel_size=2))
        
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
            
        return nn.Sequential(*layers)
    
    def _calculate_output_dim(self, 
                             input_length: int, 
                             conv_channels: List[int],
                             kernel_sizes: List[int]) -> int:
        """Calculate output dimension after convolutions and pooling"""
        length = input_length
        
        for i, (_, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            # After convolution (padding maintains length)
            # After maxpool (stride=2)
            length = length // 2
            
        return length
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, channels, sequence_length)
            
        Returns:
            Dictionary with 'logits' and 'features' keys
        """
        # Convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            
        # Global average pooling
        features = self.global_avg_pool(x)
        features = features.view(features.size(0), -1)
        
        # Classification
        logits = self.fc(features)
        
        return {
            'logits': logits,
            'features': features
        }
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class probabilities
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (probabilities, predicted_classes)
        """
        output = self.forward(x)
        probs = F.softmax(output['logits'], dim=1)
        preds = torch.argmax(probs, dim=1)
        
        return probs, preds
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get intermediate feature maps for visualization
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature maps from each conv block
        """
        attention_maps = []
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            attention_maps.append(x)
            
        return attention_maps


class ECG1DCNNWithFeatures(nn.Module):
    """
    1D-CNN that also accepts pre-extracted features (RR intervals, etc.)
    
    Combines:
    - CNN features from raw ECG waveform
    - Handcrafted features (RR intervals, morphological features)
    """
    
    def __init__(self,
                 input_channels: int = 1,
                 input_length: int = 187,
                 num_classes: int = 5,
                 num_handcrafted_features: int = 50,
                 conv_channels: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [5, 5, 3],
                 dropout_rate: float = 0.2):
        """
        Initialize CNN with handcrafted features
        
        Args:
            input_channels: Number of input channels
            input_length: Length of input sequence
            num_classes: Number of output classes
            num_handcrafted_features: Number of additional features
            conv_channels: List of channels for conv layers
            kernel_sizes: List of kernel sizes
            dropout_rate: Dropout probability
        """
        super(ECG1DCNNWithFeatures, self).__init__()
        
        # CNN backbone
        self.cnn = ECG1DCNN(
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            dropout_rate=dropout_rate
        )
        
        # Handcrafted feature branch
        self.feature_branch = nn.Sequential(
            nn.Linear(num_handcrafted_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion layer
        fusion_dim = conv_channels[-1] + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, 
                x_ecg: torch.Tensor, 
                x_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x_ecg: Raw ECG waveform (batch, 1, length)
            x_features: Handcrafted features (batch, num_features)
            
        Returns:
            Dictionary with 'logits' and 'features' keys
        """
        # CNN features
        cnn_output = self.cnn.forward(x_ecg)
        cnn_features = cnn_output['features']
        
        # Handcrafted features
        handcrafted_features = self.feature_branch(x_features)
        
        # Concatenate features
        combined_features = torch.cat([cnn_features, handcrafted_features], dim=1)
        
        # Classification
        logits = self.fusion(combined_features)
        
        return {
            'logits': logits,
            'features': combined_features,
            'cnn_features': cnn_features,
            'handcrafted_features': handcrafted_features
        }


if __name__ == "__main__":
    # Test CNN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ECG1DCNN(input_channels=1, input_length=187, num_classes=5)
    model.to(device)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 1, 187).to(device)
    
    with torch.no_grad():
        output = model(x)
        
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Output features shape: {output['features'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")