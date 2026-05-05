import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_dataloaders
from sod_model import SODModel

# 1. Loss Function: Binary Cross-Entropy + 0.5 * (1 - IoU)
def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    iou_loss = 1 - iou.mean()
    
    return bce + 0.5 * iou_loss

def train_model():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load Data & Model
    train_loader, val_loader, _ = get_dataloaders(batch_size=16)
    model = SODModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        # Progress bar for hands-on feedback
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            # Forward Pass
            outputs = model(images)
            loss = combined_loss(outputs, masks)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += combined_loss(outputs, masks).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Best Model (Checkpointing)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_state_dict(), "best_sod_model.pth")
            print("--> Model Saved!")

if __name__ == "__main__":
    train_model()