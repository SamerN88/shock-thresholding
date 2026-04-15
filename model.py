import torch
import torch.nn as nn

# Notation:
#   C = channels (C_i/C_o = in/out)
#   B = batch size
#   L = temporal length (for initial input, this is the ECG seg length, i.e. num samples)
#   r = reduction


# Component
class SEBlock(nn.Module):
    """
    "Squeeze-and-Excitation Block"
    Follows Hu et al. (2018); see Figure 3.
    """
    def __init__(self, channels, reduction):
        super(SEBlock, self).__init__()
        reduced = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),   # (B, C/r)
            nn.ReLU(),                      # (B, C/r)
            nn.Linear(reduced, channels),   # (B, C)
            nn.Sigmoid()                    # (B, C)
        )

    def forward(self, x):       # x: (B, C, L)
        x = x.mean(dim=-1)      # x: (B, C)     global average pooling over time
        x = self.fc(x)          # x: (B, C)     FC -> ReLU -> FC -> Sigmoid
        return x.unsqueeze(-1)  # (B, C, 1)     must unsqueeze to recover the time dim


# Component
class PreActResBranch(nn.Module):
    """
    "Pre-Activation Residual Unit"
    Follows He et al. (2016); see Figure 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(PreActResBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),  # (B, C_i, L)
            nn.ReLU(),  # (B, C_i, L)
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=samepad(kernel_size), bias=False),  # (B, C_o, L/stride)
            nn.BatchNorm1d(out_channels),  # (B, C_o, L/stride)
            nn.ReLU(),  # (B, C_o, L/stride)
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=samepad(kernel_size), bias=False)  # (B, C_o, L/stride)
        )

    def forward(self, x):       # x: (B, C_i, L)
        return self.conv(x)     # (B, C_o, L/stride)    temporal dimension compressed, and typically channels expanded


# Component (combines SEBlock and PreActResUnit)
class SEResNet(nn.Module):
    """
    "SE-ResNet Module"
    Follows Hu et al. (2018); see Figure 3.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, reduction):
        super(SEResNet, self).__init__()

        # Combined SE+Res arch:    input x0  ->  x1=Residual(x0)  ->  x2=SE(x1)  ->  x3=x1*x2  ->  output x3+x0

        self.preActResUnit = PreActResBranch(in_channels, out_channels, kernel_size=kernel_size, stride=stride)  # (B, C_o, L/stride)
        self.seBlock = SEBlock(out_channels, reduction)  # (B, C_o, 1)

        # Skip connection; may have to reshape original input x0 via convolution if its shape doesn't match
        # the output of the SE block after scaling (x3).
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if (stride != 1 or in_channels != out_channels) else nn.Identity()  # (B, C_o, L/stride)

    def forward(self, x):               # x:  (B, C_i, L)
        x0 = x                          # x0: (B, C_i, L)
        x1 = self.preActResUnit(x0)     # x1: (B, C_o, L/stride)    compress temporal dimension, use pre-activation residual
        x2 = self.seBlock(x1)           # x2: (B, C_o, 1)           "squeeze and excite"
        x3 = x1 * x2                    # x3: (B, C_o, L/stride)    scale the input feature map x1 by the per-channel weights x2
        return x3 + self.shortcut(x0)   # (B, C_o, L/stride)        skip connection; add original input x0 to output of SE block (may need reshaping, see defn of self.shortcut)


# Final CNN architecture for 1-lead ECG
class Ecg1LeadCNN(nn.Module):
    """
    Takes inspiration from Zhang et al. (2020); see Figure 2.
    """
    def __init__(self):
        super(Ecg1LeadCNN, self).__init__()

        # Here, L is the ECG seg length, i.e. num samples

        # Stem has one job: convert a raw signal into a spatial feature map at reduced resolution
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=25, stride=2, padding=samepad(25), bias=False),  # (B, 32, L/2)
            nn.Dropout(0.2)
        )
        self.blocks = nn.Sequential(
            SEResNet(in_channels=32, out_channels=64, kernel_size=15, stride=2, reduction=8),   # (B, 64, L/4)
            SEResNet(in_channels=64, out_channels=128, kernel_size=7, stride=2, reduction=8),   # (B, 128, L/8)
            SEResNet(in_channels=128, out_channels=256, kernel_size=5, stride=2, reduction=8)   # (B, 256, L/16)
        )
        self.final_activation = nn.Sequential(
            nn.BatchNorm1d(256),    # (B, 256, L/16)
            nn.ReLU()               # (B, 256, L/16)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # (B, 1)
        )

    def forward(self, x):               # x: (B, 1, L)
        x = self.stem(x)                # x: (B, 32, L/2)       convert a raw signal into a spatial feature map at reduced resolution
        x = self.blocks(x)              # x: (B, 256, L/16)     bulk of the CNN; multiple SE-ResNet modules
        x = self.final_activation(x)    # x: (B, 256, L/16)     ensures that the output of the last Conv1d is activated
        x = x.mean(dim=-1)              # x: (B, 256)           global average pooling
        return self.classifier(x)       # (B, 1)

    def predict(self, x, temperature=1.0, threshold=0.5):                # x: (B, 1, 500)
        self.eval()
        x = x.to(device=next(self.parameters()).device)
        with torch.no_grad():
            logits = self.forward(x)                        # logits: (B, 1)
            probs = torch.sigmoid(logits / temperature)     # probs:  (B, 1)
            preds = (probs >= threshold).long()             # preds:  (B, 1)
        return preds.squeeze(-1).cpu().numpy()              # (B,)  SHOCKABLE(1) / NON_SHOCKABLE(0)


def samepad(kernel_size):
    return int((kernel_size - 1) / 2)
