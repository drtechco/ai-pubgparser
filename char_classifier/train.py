import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import torchvision.utils as vutils
from typing import List, Tuple




def plot_losses(losses: List[float], save_path: str = "training_loss.png"):
    """Plot the training loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_highest_loss_images(images: torch.Tensor, labels: torch.Tensor, 
                           predictions: torch.Tensor, losses: torch.Tensor,
                           class_names: List[str], dataset,
                           image_indices: List[int],
                           num_images: int = 10,
                           save_path: str = "highest_losses.png"):
    """
    Plot images with the highest losses.
    
    Args:
        dataset: The ImageFolder dataset containing the samples attribute
        image_indices: Original indices of the images in the dataset
    """
    # Get indices of highest losses
    _, indices = torch.topk(losses, min(num_images, len(losses)))
    
    # Create figure
    fig = plt.figure(figsize=(20, 6))  # Made taller to accommodate filename
    for idx, i in enumerate(indices):
        ax = plt.subplot(1, len(indices), idx + 1)
        
        # Get the image and convert it for display
        img = images[i].cpu()
        # Denormalize the image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Get the original image path and extract filename
        original_idx = image_indices[i]
        img_path = dataset.samples[original_idx][0]
        filename = os.path.basename(img_path)
        
        plt.imshow(img.permute(1, 2, 0))
        true_label = class_names[labels[i]]
        pred_label = class_names[predictions[i]]
        
        # Create multi-line title with smaller font size
        # print("FNAME: ", filename[:10])
        title = f'True: {true_label}\nPred: {pred_label}\nLoss: {losses[i]:.2f}\n{filename[:10]}'
        plt.title(title, fontsize=8, pad=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)  # Increased DPI for better text readability
    plt.close()


def load_data(data_path: str, batch_size: int):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # ResNet standard input size
        transforms.ToTensor(),
        transforms.RandomInvert(0.8),
        # transforms.RandomAdjustSharpness(0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])
    train_dataset = datasets.ImageFolder(root=data_path, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), train_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int, help="epochs to train for")
    parser.add_argument("--batch", type=int, help="batch size")
    parser.add_argument("--plot-interval", default=5, type=int, help="epoch interval for plotting")
    if not os.path.exists("./graphs"):
        os.makedirs("./graphs")

    opts = parser.parse_args()
    print(f"running training for {opts.epochs} with {opts.batch} image batch")
    num_epochs = opts.epochs
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet34(pretrained=True)
    # model = models.resnet50(pretrained=True)
    # train_loader, train_dataset = load_data("/home/hbdesk/labelstudio_convert/bbox_char2_concat/", opts.batch)
    train_loader, train_dataset = load_data("/home/hbdesk/labelstudio_convert/char3_padded/", opts.batch)
    with open('./char_list.txt', 'w') as file:
        file.write('\n'.join(train_dataset.classes))
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')  # Changed to 'none' to get per-sample losses
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store losses
    epoch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_batch_losses = []
        epoch_batch_images = []
        epoch_batch_labels = []
        epoch_batch_predictions = []
        epoch_batch_indices = []  # To store original dataset indices
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)

            # Calculate the original indices for this batch
            start_idx = batch_idx * train_loader.batch_size
            end_idx = start_idx + labels.size(0)
            batch_indices = list(range(start_idx, end_idx))
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)  # This is now a tensor with batch_size elements
            
            # Store batch information for plotting
            epoch_batch_losses.append(loss.cpu().detach())
            epoch_batch_images.append(images.cpu())
            epoch_batch_labels.append(labels.cpu())
            epoch_batch_predictions.append(outputs.argmax(dim=1).cpu())
            epoch_batch_indices.extend(batch_indices)  # Store indices

            # Calculate mean loss for backward pass
            loss_mean = loss.mean()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()
            running_loss += loss_mean.item() * images.size(0)

        # Calculate and print average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Plot every plot_interval epochs and on the last epoch
        if (epoch + 1) % opts.plot_interval == 0 or epoch == num_epochs - 1:
            # Plot loss curve
            plot_losses(epoch_losses, "./graphs/training_loss.png")
            
            try:
                # Concatenate batch information
                all_losses = torch.cat(epoch_batch_losses)
                all_images = torch.cat(epoch_batch_images)
                all_labels = torch.cat(epoch_batch_labels)
                all_predictions = torch.cat(epoch_batch_predictions)
                
                # Plot highest loss images
                plot_highest_loss_images(
                    all_images, all_labels, all_predictions, all_losses,
                    train_dataset.classes, train_dataset,
                    epoch_batch_indices,
                    num_images=20,
                    save_path=f"./graphs/highest_losses_epoch_{epoch+1}.png"
                )
            except RuntimeError as e:
                print(f"Warning: Could not generate highest loss plot for epoch {epoch+1}: {str(e)}")

    # Save the model checkpoint
    torch.save(model.state_dict(), "char_classifierv1.6_112.pth")
    print("Training complete. Model saved.")
