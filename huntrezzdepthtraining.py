# PyTorch and neural network imports
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Hugging Face transformers for depth estimation
from transformers import DPTForDepthEstimation

# Utilities for logging, progress bars and system
import logging
import sys
from tqdm import tqdm

# Data handling and processing
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import cv2
import os

# Visualization
import matplotlib.pyplot as plt

# Image transformations and augmentation
import torchvision.transforms as transforms

# Mixed precision training
from torch.amp import autocast, GradScaler 

# Dataset utilities
from torch.utils.data import Subset
import random

# Set up logging configuration to track training progress and errors
# Format includes timestamp, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Enable CUDA optimization by allowing the cuDNN auto-tuner to find the best algorithm
torch.backends.cudnn.benchmark = True  

# Custom Dataset class for loading high-resolution WSI (Whole Slide Image) data
class HRWSIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Initialize dataset with directory path and optional transforms
        self.data_dir = data_dir
        self.transform = transform
        # Get list of all jpg images in the imgs subdirectory
        self.image_files = [f for f in os.listdir(os.path.join(data_dir, 'imgs')) if f.endswith('.jpg')]

    def __len__(self):
        # Return total number of samples in dataset
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get paths for both image and its corresponding depth map
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, 'imgs', img_name)
        depth_path = os.path.join(self.data_dir, 'gts', img_name.replace('.jpg', '.png'))

        # Load and process the image and depth map
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB
        depth = Image.open(depth_path)  # Load depth map

        # Apply any transforms if specified
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)

        # Normalize depth values from [0,255] to [0,1] range
        depth = depth.float() / 255.0  

        # Return image-depth pair as tensors for training
        # Using float32 for compatibility with Automatic Mixed Precision (AMP)
        return image, depth

# Function to calculate Root Mean Square Error between predicted and target depth maps
# Takes two tensors as input and returns their RMSE using PyTorch's MSE loss followed by square root
def calculate_rmse(predictions, targets): #
    return torch.sqrt(F.mse_loss(predictions, targets))

# Compressed student model architecture for depth estimation
# Uses an encoder-decoder structure with convolutional and transposed convolutional layers
class CompressedStudentModel(nn.Module):
    def __init__(self):
        # Initialize the parent nn.Module class
        super(CompressedStudentModel, self).__init__()
        
        # Encoder network that progressively reduces spatial dimensions while increasing channels
        self.encoder = nn.Sequential(
            # First conv block: RGB input (3 channels) -> 64 channels, 3x3 kernel with padding to maintain size
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),  # Activation function to introduce non-linearity
            # Second conv block: 64 -> 64 channels, same spatial dimensions
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # First pooling layer: reduces spatial dimensions by 2
            nn.MaxPool2d(2),
            # Third conv block: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Fourth conv block: 128 -> 128 channels
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Second pooling layer: further reduces spatial dimensions
            nn.MaxPool2d(2),
            # Fifth conv block: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # Sixth conv block: 256 -> 256 channels
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Decoder network that upsamples back to original resolution
        self.decoder = nn.Sequential(
            # First upsampling: 256 -> 128 channels, doubles spatial dimensions
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Second upsampling: 128 -> 64 channels, doubles spatial dimensions again
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Final conv layer: 64 -> 1 channel for depth map output
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Pass input tensor x through encoder network to extract hierarchical features
        features = self.encoder(x)
        # Pass encoded features through decoder network to generate single-channel depth map
        depth = self.decoder(features)
        # Return the predicted depth map tensor
        return depth

def train_step(teacher, student, optimizer, inputs, targets, device, scaler):
    # Enable automatic mixed precision training using cuda device and float16 precision
    with autocast(device_type='cuda', dtype=torch.float16):  # Updated autocast usage
        # Check if input is single channel, if so repeat to make 3 channels for teacher model
        if inputs.shape[1] == 1:
            teacher_inputs = inputs.repeat(1, 3, 1, 1)
        else:
            teacher_inputs = inputs
        
        # Get teacher model predictions without computing gradients since we don't update teacher
        with torch.no_grad():
            teacher_outputs = teacher(teacher_inputs).predicted_depth
        
        # Get student model predictions by passing inputs through student network
        student_outputs = student(inputs)
        
        # Remove singleton dimensions from outputs to match shapes
        student_outputs = student_outputs.squeeze(1)
        teacher_outputs = teacher_outputs.squeeze(1)
        
        # Calculate MSE loss between student and teacher predictions, ensuring float32 precision
        loss = F.mse_loss(student_outputs.float(), teacher_outputs.float())  # Ensure float32 for loss calculation
    
    # Scale loss and backpropagate gradients using gradient scaler
    scaler.scale(loss).backward()
    # Update model parameters with scaled gradients
    scaler.step(optimizer)
    # Update gradient scaler statistics
    scaler.update()
    # Zero out gradients for next iteration, using memory efficient option
    optimizer.zero_grad(set_to_none=True)
    
    # Return scalar loss value for logging
    return loss.item()

def train(teacher, student, train_dataset, val_loader, epochs, learning_rate, device):
    # Set teacher model to eval mode since we don't train it
    teacher.eval()
    # Set student model to training mode
    student.train()
    # Initialize Adam optimizer for student model parameters
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    # Initialize learning rate scheduler that reduces LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Initialize lists to track metrics over epochs
    train_losses = []
    val_losses = []
    train_rmses = []
    val_rmses = []
    
    # Main training loop over epochs
    for epoch in range(epochs):
        # Randomly sample a portion of training data for each epoch to reduce computation
        train_subset_indices = random.sample(range(len(train_dataset)), len(train_dataset) // 20)
        train_subset = Subset(train_dataset, train_subset_indices)
        # Create data loader for training subset with parallel loading and pinned memory
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        
        # Initialize running totals for epoch metrics
        total_loss = 0
        total_rmse = 0
        # Create progress bar for training batches
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        # Iterate over batches in training data
        for inputs, targets in progress_bar:
            # Move batch data to appropriate device (CPU/GPU)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Perform single training step and get loss
            loss = train_step(teacher, student, optimizer, inputs, targets, device, scaler)
            total_loss += loss
            
            # Calculate RMSE metric without computing gradients
            with torch.no_grad():
                student_outputs = student(inputs).squeeze(1)
                rmse = calculate_rmse(student_outputs, targets.squeeze(1))
                total_rmse += rmse.item()
            
            # Update progress bar with current metrics
            progress_bar.set_postfix({'loss': f'{loss:.4f}', 'rmse': f'{rmse.item():.4f}'})
        
        # Calculate average metrics for training epoch
        avg_train_loss = total_loss / len(train_loader)
        avg_train_rmse = total_rmse / len(train_loader)
        train_losses.append(avg_train_loss)
        train_rmses.append(avg_train_rmse)
        # Log training metrics
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train RMSE: {avg_train_rmse:.4f}")
        
        # Switch to evaluation mode for validation
        student.eval()
        total_val_loss = 0
        total_val_rmse = 0
        # Validate without computing gradients
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move validation batch to device
                inputs, targets = inputs.to(device), targets.to(device)
                # Get model predictions
                outputs = student(inputs).squeeze(1)
                # Calculate validation metrics
                val_loss = F.mse_loss(outputs, targets.squeeze(1))
                val_rmse = calculate_rmse(outputs, targets.squeeze(1))
                total_val_loss += val_loss.item()
                total_val_rmse += val_rmse.item()
        
        # Calculate average validation metrics
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_rmse = total_val_rmse / len(val_loader)
        val_losses.append(avg_val_loss)
        val_rmses.append(avg_val_rmse)
        # Log validation metrics
        logger.info(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}, Validation RMSE: {avg_val_rmse:.4f}")
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        # Switch back to training mode
        student.train()
    
    # Save final trained model weights
    torch.save(student.state_dict(), "huntrezz_depth_v5.pt")
    # Return all tracked metrics
    return train_losses, val_losses, train_rmses, val_rmses

if __name__ == "__main__":
    try:
        # Check if CUDA GPU is available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        # Log which device (GPU/CPU) is being used for training
        logger.info(f"Using device: {device}")
        
        # Define image preprocessing pipeline - resize images to 256x256 and convert to PyTorch tensors
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Load training dataset from HR-WSI train directory with defined transforms
        train_dataset = HRWSIDataset(r"C:\Users\hunte\OneDrive\Desktop\HR-WSI\HR-WSI\train", transform=transform)
        # Load validation dataset from HR-WSI val directory with defined transforms
        val_dataset = HRWSIDataset(r"C:\Users\hunte\OneDrive\Desktop\HR-WSI\HR-WSI\val", transform=transform)
        
        # Create validation data loader with batch size 64, no shuffling, 8 worker processes, pinned memory for faster GPU transfer
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
        
        # Load pre-trained DPT teacher model from Intel's Hugging Face repository
        teacher = DPTForDepthEstimation.from_pretrained("Intel/dpt-swinv2-tiny-256")
        # Move teacher model to selected device (GPU/CPU)
        teacher.to(device)
        # Set teacher model to evaluation mode
        teacher.eval()
        
        # Initialize student model with compressed architecture and move to device
        student = CompressedStudentModel().to(device)
        # Set initial learning rate for optimizer
        learning_rate = 2.06e-4
        # Log the learning rate being used
        logger.info(f"Using learning rate: {learning_rate}")
        
        # Train the student model using knowledge distillation from teacher, store training metrics
        train_losses, val_losses, train_rmses, val_rmses = train(teacher, student, train_dataset, val_loader, epochs=20, learning_rate=learning_rate, device=device)

        # Create figure with two subplots side by side, size 10x5
        plt.figure(figsize=(10, 5))
        # First subplot for loss curves
        plt.subplot(1, 2, 1)
        # Plot training loss history
        plt.plot(train_losses, label='Train Loss')
        # Plot validation loss history
        plt.plot(val_losses, label='Validation Loss')
        # Label x-axis
        plt.xlabel('Epoch')
        # Label y-axis
        plt.ylabel('Loss')
        # Add legend to distinguish curves
        plt.legend()
        # Add title to loss plot
        plt.title('Training and Validation Loss')
        
        # Second subplot for RMSE curves
        plt.subplot(1, 2, 2)
        # Plot training RMSE history
        plt.plot(train_rmses, label='Train RMSE')
        # Plot validation RMSE history
        plt.plot(val_rmses, label='Validation RMSE')
        # Label x-axis
        plt.xlabel('Epoch')
        # Label y-axis
        plt.ylabel('RMSE')
        # Add legend to distinguish curves
        plt.legend()
        # Add title to RMSE plot
        plt.title('Training and Validation RMSE')
        
        # Adjust spacing between subplots
        plt.tight_layout()
        # Save plot to file
        plt.savefig('training_metrics.png')
        # Display plot
        plt.show()
    
    except Exception as e:
        # Log any errors that occur during execution
        logger.error(f"An error occurred: {e}")
        # Exit program with error code 1
        sys.exit(1)