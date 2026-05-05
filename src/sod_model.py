import torch
import torch.nn as nn

class SODModel(nn.Module):
    def __init__(self):
        super(SODModel, self).__init__()

        # --- ENCODER ---
        # Input: 3 x 224 x 224
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        ) # Result: 64 x 112 x 112

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # Result: 128 x 56 x 56

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # Result: 256 x 28 x 28

        # --- BRIDGE ---
        self.bridge = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU()
        ) # Result: 512 x 28 x 28

        # --- DECODER ---
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU()
        ) # Result: 256 x 56 x 56

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU()
        ) # Result: 128 x 112 x 112

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU()
        ) # Result: 64 x 224 x 224

        # --- FINAL LAYER ---
        self.final = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid() # Squashes output between 0 and 1 (probability)
        ) # Result: 1 x 224 x 224

    def forward(self, x):
        # Forward pass through Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        
        # Bridge
        x = self.bridge(x)
        
        # Forward pass through Decoder
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        
        # Output
        return self.final(x)

# Quick test to check if the math is correct
if __name__ == "__main__":
    model = SODModel()
    dummy_input = torch.randn(1, 3, 224, 224) # One RGB image
    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}") # Should be [1, 1, 224, 224]