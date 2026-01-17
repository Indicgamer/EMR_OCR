import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import CTCLoss
import numpy as np

class CRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) for OCR
    Architecture: CNN Feature Extraction + RNN Sequence Modeling + CTC Loss
    
    Suitable for recognizing text in images, especially for document OCR
    """
    
    def __init__(self, num_channels=3, num_classes=95, hidden_size=256, num_layers=2):
        """
        Initialize CRNN model
        
        Args:
            num_channels: Number of input channels (3 for RGB)
            num_classes: Number of character classes
            hidden_size: Hidden size for LSTM
            num_layers: Number of LSTM layers
        """
        super(CRNN, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN Feature Extraction (ResNet-34 backbone)
        resnet = models.resnet34(pretrained=True)
        # Remove the classification head
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 256))
        
        # RNN for sequence modeling
        self.rnn = nn.Sequential(
            nn.LSTM(
                input_size=512,  # ResNet-34 output channels
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True
            )
        )
        
        # Linear layer for character classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, images):
        """
        Forward pass
        
        Args:
            images: Input images of shape (batch_size, 3, height, width)
            
        Returns:
            output: Logits of shape (batch_size, sequence_length, num_classes)
        """
        # CNN feature extraction
        conv_features = self.cnn(images)  # (batch_size, 512, h, w)
        
        # Adaptive pooling
        features = self.adaptive_pool(conv_features)  # (batch_size, 512, 1, 256)
        features = features.squeeze(2)  # (batch_size, 512, 256)
        features = features.permute(0, 2, 1)  # (batch_size, 256, 512)
        
        # RNN sequence modeling
        rnn_output, _ = self.rnn(features)  # (batch_size, 256, 512)
        
        # Character classification
        output = self.fc(rnn_output)  # (batch_size, 256, num_classes)
        
        return output


class OCRModel(nn.Module):
    """
    Optimized OCR Model for Medical Documents
    Uses MobileNetV2 for faster inference
    """
    
    def __init__(self, num_classes=95, hidden_size=256):
        """
        Initialize optimized OCR model
        
        Args:
            num_classes: Number of character classes
            hidden_size: Hidden size for LSTM
        """
        super(OCRModel, self).__init__()
        
        # Lightweight CNN backbone (MobileNetV2)
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.cnn = mobilenet.features
        
        # Feature adaptation
        self.feature_adapt = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 256))
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, images):
        """Forward pass"""
        # CNN feature extraction
        features = self.cnn(images)
        
        # Feature adaptation
        features = self.feature_adapt(features)
        features = features.squeeze(2).permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        
        # Character prediction
        output = self.fc(lstm_out)
        
        return output


def create_model(model_type="CRNN", num_classes=95, pretrained=True, **kwargs):
    """
    Factory function to create OCR models
    
    Args:
        model_type: Type of model ('CRNN' or 'OCRModel')
        num_classes: Number of character classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for model
        
    Returns:
        model: Initialized model
    """
    if model_type == "CRNN":
        model = CRNN(num_classes=num_classes, **kwargs)
    elif model_type == "OCRModel":
        model = OCRModel(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model("CRNN", num_classes=95)
    model = model.to(device)
    
    # Dummy input
    dummy_input = torch.randn(2, 3, 64, 256).to(device)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
