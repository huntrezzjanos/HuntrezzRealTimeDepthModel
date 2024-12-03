# Import required libraries
# torch - Deep learning framework for model definition and inference
# torch.nn - Neural network modules and layers
# torchvision.transforms - Image preprocessing utilities
# cv2 - OpenCV library for image processing and visualization
# numpy - Numerical computing library for array operations
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np

# Define the compressed student model architecture that will estimate depth from RGB images
class CompressedStudentModel(nn.Module):
    def __init__(self):
        super(CompressedStudentModel, self).__init__()
        # Encoder network that extracts features from input image
        # Uses a series of convolutional layers with ReLU activation and max pooling
        # Progressively increases channels (3->64->128->256) while reducing spatial dimensions
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),    # First conv layer: RGB(3) -> 64 channels
            nn.ReLU(),                                      # ReLU activation after conv
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # Second conv: preserve 64 channels
            nn.ReLU(),
            nn.MaxPool2d(2),                               # Reduce spatial size by 2x
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Increase channels to 128
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # Preserve 128 channels
            nn.ReLU(),
            nn.MaxPool2d(2),                               # Further reduce spatial size
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Increase to 256 channels
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Final encoder layer
            nn.ReLU(),
        )
        # Decoder network that upsamples features back to original resolution
        # Uses transposed convolutions for upsampling and reduces channels
        # Final output is single-channel depth map
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample and reduce to 128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # Upsample and reduce to 64
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),    # Final conv to produce single-channel depth
        )

    # Forward pass through the network
    def forward(self, x):
        features = self.encoder(x)      # Extract features through encoder
        depth = self.decoder(features)  # Decode features to depth map
        return depth

# Preprocess input image for model inference
# Converts image to PyTorch tensor and resizes to 256x256
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),              # Convert to PIL image format
        transforms.Resize((256, 256)),        # Resize to model's expected input size
        transforms.ToTensor(),                # Convert to tensor and normalize to [0,1]
    ])
    return transform(image).unsqueeze(0)      # Add batch dimension

# Convert predicted depth map to visualizable colormap
def visualize_depth(image, depth):
    depth = depth.squeeze().cpu().numpy()     # Remove batch dimension and convert to numpy
    depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to [0,1] range
    depth_colormap = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)  # Apply colormap for visualization
    
    # Resize image to match the depth map dimensions
    image_resized = cv2.resize(image, (depth_colormap.shape[1], depth_colormap.shape[0]))
    
    combined = np.hstack((image_resized, depth_colormap))
    cv2.imshow('Webcam and Depth Estimation', combined)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CompressedStudentModel().to(device)
model.load_state_dict(torch.load("huntrezz_depth_v2.pt", map_location=device))

# Set model to evaluation mode to disable dropout and batch normalization layers for inference
model.eval()

# Initialize video capture from the default webcam (device 0)
# Returns a VideoCapture object that will be used to read frames
cap = cv2.VideoCapture(0)

# Use torch.no_grad() context manager to disable gradient calculations
# This reduces memory usage and speeds up inference since we don't need gradients
with torch.no_grad():
    # Infinite loop to continuously process frames from webcam
    while True:
        # Read a frame from the webcam
        # ret is a boolean indicating if frame was successfully captured
        # frame contains the actual image data as a numpy array
        ret, frame = cap.read()
        # Break the loop if frame capture failed
        if not ret:
            break

        # Preprocess the captured frame for model input:
        # 1. Convert to PIL image and resize to 256x256
        # 2. Convert to tensor and add batch dimension
        # 3. Move tensor to GPU/CPU device
        input_tensor = preprocess_image(frame).to(device)
        # Pass preprocessed frame through model to get depth prediction
        # depth_output will be a tensor containing the predicted depth values
        depth_output = model(input_tensor)
        
        # Create visualization by:
        # 1. Converting depth prediction to colormap
        # 2. Resizing original frame to match depth map
        # 3. Combining original and depth images side by side
        # 4. Displaying the combined image in a window
        visualize_depth(frame, depth_output)

        # Check if 'q' key was pressed
        # cv2.waitKey(1) returns the code of pressed key
        # 0xFF mask ensures i get last 8 bits only
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam device so other applications can use it
cap.release()
# Close all OpenCV windows that were created
cv2.destroyAllWindows()
