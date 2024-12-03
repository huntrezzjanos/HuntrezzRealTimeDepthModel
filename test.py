import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np

class CompressedStudentModel(nn.Module):
    def __init__(self):
        super(CompressedStudentModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def visualize_depth(image, depth):
    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_colormap = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    
    # Resize image to match the depth map dimensions
    image_resized = cv2.resize(image, (depth_colormap.shape[1], depth_colormap.shape[0]))
    
    combined = np.hstack((image_resized, depth_colormap))
    cv2.imshow('Webcam and Depth Estimation', combined)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CompressedStudentModel().to(device)
model.load_state_dict(torch.load("huntrezz_depth_v2.pt", map_location=device))

model.eval()

# Open webcam
cap = cv2.VideoCapture(0)

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_image(frame).to(device)
        depth_output = model(input_tensor)
        
        visualize_depth(frame, depth_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()