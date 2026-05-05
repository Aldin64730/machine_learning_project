import torch
import matplotlib.pyplot as plt
import numpy as np
from data_loader import get_dataloaders
from sod_model import SODModel

def calculate_metrics(pred, target):
    # Threshold predictions to get binary mask
    pred_bin = (pred > 0.5).float()
    
    # Intersection and Union for IoU
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Precision, Recall, F1
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return iou.item(), precision.item(), recall.item(), f1.item()

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data and model
    _, _, test_loader = get_dataloaders(batch_size=1)
    model = SODModel().to(device)
    model.load_state_dict(torch.load("best_sod_model.pth", map_location=device))
    model.eval()

    all_iou, all_p, all_r, all_f1 = [], [], [], []

    print("Evaluating on test set...")
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image, mask = image.to(device), mask.to(device)
            output = model(image)
            
            # Metric calculation
            iou, p, r, f1 = calculate_metrics(output, mask)
            all_iou.append(iou); all_p.append(p); all_r.append(r); all_f1.append(f1)

            # Visualize the first 5 results
            if i < 5:
                visualize_results(image[0], mask[0], output[0], i)

    print(f"\n--- Final Results ---")
    print(f"Mean IoU:       {np.mean(all_iou):.4f}")
    print(f"Mean Precision: {np.mean(all_p):.4f}")
    print(f"Mean Recall:    {np.mean(all_r):.4f}")
    print(f"Mean F1-Score:  {np.mean(all_f1):.4f}")

def visualize_results(image, gt, pred, index):
    # Convert tensors back to numpy for plotting
    image = image.cpu().permute(1, 2, 0).numpy()
    gt = gt.cpu().squeeze().numpy()
    pred = pred.cpu().squeeze().numpy()
    
    # Create overlay
    overlay = image.copy()
    overlay[:,:,1] = np.where(pred > 0.5, 1.0, overlay[:,:,1]) # Highlight salient area in green

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image)
    
    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(gt, cmap='gray')
    
    plt.subplot(1, 4, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred, cmap='gray')
    
    plt.subplot(1, 4, 4)
    plt.title("Overlay Result")
    plt.imshow(overlay)
    
    plt.savefig(f"result_{index}.png")
    plt.show()

if __name__ == "__main__":
    evaluate()