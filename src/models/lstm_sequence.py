"""
Sequence models for multi-beat context in ECG analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class ECGSequenceModel(nn.Module):
    """
    LSTM-based model for processing sequences of ECG beats
    
    Captures long-range dependencies between consecutive beats
    for rhythm analysis (e.g., AFib detection, bradycardia)
    """
    
    def __init__(self,
                 input_channels: int = 1,
                 input_length: int = 187,  # Length of each beat
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 5,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        """
        Initialize sequence model
        
        Args:
            input_channels: Number of input channels
            input_length: Length of each beat (number of samples)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super(ECGSequenceModel, self).__init__()
        
        self.input_channels = input_channels
        self.input_length = input_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Feature extraction from each beat (optional CNN before LSTM)
        self.beat_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # LSTM for sequence modeling
        lstm_input_size = 32  # Output from beat_encoder
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Classifier
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Attention mechanism for variable-length sequences
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, sequence_length, input_size)
            lengths: Optional lengths of sequences for packing
            
        Returns:
            Dictionary with logits, features, and attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape for CNN: (batch * seq_len, 1, input_size)
        x_reshaped = x.view(batch_size * seq_len, 1, self.input_size)
        
        # Encode each beat
        beat_features = self.beat_encoder(x_reshaped)
        beat_features = beat_features.view(batch_size, seq_len, -1)
        
        # Pack sequence if lengths provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                beat_features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(beat_features)
            
        # Apply attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use final hidden state or attention-pooled
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            final_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            final_hidden = hidden[-1, :, :]
            
        # Classification
        logits = self.classifier(final_hidden)
        
        return {
            'logits': logits,
            'features': final_hidden,
            'attention_weights': attn_weights,
            'lstm_out': lstm_out
        }
    
    def predict(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple:
        """Predict class probabilities"""
        output = self.forward(x, lengths)
        probs = F.softmax(output['logits'], dim=1)
        preds = torch.argmax(probs, dim=1)
        return probs, preds


class BiLSTMWithAttention(nn.Module):
    """
    Bidirectional LSTM with multi-head attention for ECG rhythm analysis
    """
    
    def __init__(self,
                 input_dim: int = 187,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 num_classes: int = 5,
                 dropout: float = 0.2):
        """
        Initialize BiLSTM with attention
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(BiLSTMWithAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Dictionary with predictions and attention weights
        """
        # Project input
        x = self.input_proj(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=mask)
        
        # Residual connection with layer norm
        lstm_out = self.layer_norm(lstm_out + attn_out)
        
        # Global average pooling over sequence
        pooled = torch.mean(lstm_out, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return {
            'logits': logits,
            'attention_weights': attn_weights,
            'features': pooled
        }


class TransformerECG(nn.Module):
    """
    Transformer-based model for ECG sequence analysis
    
    Uses self-attention to capture long-range dependencies
    """
    
    def __init__(self,
                 input_dim: int = 187,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 num_classes: int = 5,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_seq_length: int = 100):
        """
        Initialize Transformer model
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            max_seq_length: Maximum sequence length for positional encoding
        """
        super(TransformerECG, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Dictionary with logits and attention weights
        """
        # Embed input
        x = self.input_embedding(x) * (self.d_model ** 0.5)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global average pooling
        pooled = torch.mean(x, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return {
            'logits': logits,
            'features': pooled,
            'encoded_sequence': x
        }
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention maps from all layers"""
        # This requires modifying the transformer to return attention weights
        # Simplified version here
        return []


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":
    # Test sequence models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 16
    seq_len = 10  # 10 consecutive beats
    input_size = 187  # Each beat has 187 samples
    
    # Create input (batch, seq_len, input_size)
    x = torch.randn(batch_size, seq_len, input_size).to(device)
    
    # LSTM model
    lstm_model = ECGSequenceModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=5
    ).to(device)
    
    with torch.no_grad():
        output = lstm_model(x)
        
    print(f"LSTM output logits shape: {output['logits'].shape}")
    print(f"LSTM attention shape: {output['attention_weights'].shape}")
    
    # Transformer model
    transformer = TransformerECG(
        input_dim=input_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        num_classes=5
    ).to(device)
    
    with torch.no_grad():
        output = transformer(x)
        
    print(f"\nTransformer output logits shape: {output['logits'].shape}")