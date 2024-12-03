# HuntrezzRealTimeDepthModel
Monocular Depth estimation for Live Video

# Compressed Depth Estimation Model using Knowledge Distillation
Overview
This project implements a compressed depth estimation model using knowledge distillation techniques. The goal is to create a lightweight, efficient model capable of predicting depth maps from single RGB images, at such a fast rate on standard hardware that it can be used in real-time video applications. The model is trained on the HR-WSI (High-Resolution Whole Slide Image) dataset and utilizes a pre-trained DPT (Dense Prediction Transformer) model as a teacher for knowledge distillation.

# Introduction
Depth estimation from single RGB images is a challenging computer vision task with applications in robotics, augmented reality, and autonomous driving. This project aims to create a compressed model that can perform this task as efficiently as a digital camera captures images while maintaining accuracy. We employ knowledge distillation, where a larger, more complex teacher model (DPT) guides the training of a smaller, more efficient student model.

# Installation
To set up the project environment, follow these steps:
Clone the repository:
text
git clone https://github.com/yourusername/compressed-depth-estimation.git
cd compressed-depth-estimation

Install the required dependencies:
text
pip install torch torchvision tqdm pillow opencv-python matplotlib transformers

# Dataset
The project uses the HR-WSI dataset, which consists of high-resolution images paired with corresponding depth maps. The dataset should be organized as follows:
text
HR-WSI/
├── train/
│   ├── imgs/
│   │   └── *.jpg
│   └── gts/
│       └── *.png
└── val/
    ├── imgs/
    │   └── *.jpg
    └── gts/
        └── *.png

It can be downloaded at  https://drive.google.com/file/d/1OVOx6x-B0Cs-m2z_-7ZxSgRFHz_VBvDd/view?pli=1

The HRWSIDataset class in the code handles the loading and preprocessing of this dataset.

# Model Architecture
Teacher Model
We use the DPT (Dense Prediction Transformer) model pre-trained on depth estimation tasks as our teacher model. This model is loaded from the Hugging Face model hub.
Student Model
The student model (CompressedStudentModel) is a custom-designed CNN with an encoder-decoder architecture:
Encoder: Consists of convolutional layers and max pooling operations to extract hierarchical features.
Decoder: Uses transposed convolutions to upsample the features and generate the final depth map.
The student model is significantly smaller than the teacher, making it more suitable for deployment in resource-constrained environments.

# Training Process
The training process involves the following key components:
Knowledge Distillation: The student model learns from both the ground truth depth maps and the predictions of the teacher model.
Mixed Precision Training: We use automatic mixed precision to speed up training and reduce memory usage.
Learning Rate Scheduling: A ReduceLROnPlateau scheduler is employed to adjust the learning rate based on validation performance.
Data Sampling: To reduce computation, we randomly sample a subset of the training data for each epoch.
Metrics: We track both Mean Squared Error (MSE) loss and Root Mean Square Error (RMSE) during training and validation.
Usage
To train the model, run the script with the appropriate dataset paths:
text
python train_depth_model.py

The script will automatically use GPU if available, otherwise it will fall back to CPU.
Results
The training process generates a plot (training_metrics.png) showing the progression of training and validation loss and RMSE over epochs. The final trained model is saved as huntrezz_depth_v5.pt.
