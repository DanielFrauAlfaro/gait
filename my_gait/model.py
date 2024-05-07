import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


# Model with ResNet+LSTM
class ResNetLSTM(nn.Module):
    def __init__(self, heads, hidden_size=15, pretrained=True):
        super(ResNetLSTM, self).__init__()

        # Load pre-trained ResNet model
        resnet = models.resnet18(pretrained=pretrained)

        # Remove the classification layer (top layer) of ResNet
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze ResNet model
        for param in self.resnet_features.parameters():
            param.requires_grad = False

        # Define LSTM input size based on ResNet output
        resnet_output_size = resnet.fc.in_features
        self.hidden_size = hidden_size
        
        # LSTM layer
        self.lstm = nn.LSTM(resnet_output_size, hidden_size, batch_first=True)
        
        # Classifier layer
        self.fc_tasks = nn.ModuleList([
            nn.Linear(hidden_size, h) for h in heads
        ])

        # Max length of the heads for padding
        self.max_length = max(heads)



    def forward(self, x):
        # x is a batch of sequences of images
        # x shape: (batch_size, seq_length, channels, H, W)
        
        # Size values
        batch_size, seq_length, channels, H, W = x.size()

        # Reshape to (batch_size * seq_length, channels, H, W)
        x = x.view(batch_size * seq_length, channels, H, W)
        
        # Extract features from ResNet
        features = self.resnet_features(x)
        features = features.squeeze()
        features = features.view(batch_size, seq_length, -1)  # (batch_size, seq_length, resnet_output_size)
        
        # LSTM pass
        lstm_out, _ = self.lstm(features)
        
        # Take the output from the last time step (many-to-one architecture)
        lstm_last_output = lstm_out[:, -1, :]
        
        # Final classification
        output = [fc(lstm_last_output).squeeze() for fc in self.fc_tasks]

        # Add -inf padding to the tasks whose lengths are less than the maximum
        padded_tensors = []

        for idx, tensor in enumerate(output):
            padded_tensors.append([])
            for t in tensor:
                padded_tensors[idx].append(torch.nn.functional.pad(t, (0, self.max_length - len(t)), value=float("-inf")))
            padded_tensors[idx] = torch.stack(padded_tensors[idx])

        # Convert list to tensor
        padded_tensors = torch.stack(padded_tensors)

        # Apply softmax to the padded output tensor
        padded_tensors = F.softmax(padded_tensors, dim=-1)

        return padded_tensors