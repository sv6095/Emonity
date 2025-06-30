#!/usr/bin/env python
# coding: utf-8
# Speech Emotion Recognition with Enhanced Features and Models
# This script demonstrates speech emotion recognition using audio features and deep learning.
# The implementation has been enhanced to achieve higher training accuracy.

## Key Enhancements:
1. **Advanced Model Architectures**
   - Self-attention mechanisms
   - Residual connections
   - Squeeze-excitation blocks
   - BiLSTM with attention
   - ResNet-based 2D CNN

2. **Improved Feature Extraction**
   - Multiple feature types (MFCCs, spectral features, chroma, etc.)
   - Delta and delta-delta features
   - Enhanced mel spectrogram extraction

3. **Advanced Data Augmentation**
   - Multiple augmentation techniques (noise, pitch shift, time stretch)
   - Reverb effects and filter augmentation
   - Mixup and CutMix for on-the-fly augmentation
   - Class balancing

4. **Training Optimizations**
   - AdamW optimizer with weight decay
   - Cosine annealing learning rate scheduling
   - Mixed precision training
   - Gradient clipping
   - Early stopping with increased patience

5. **Ensemble Learning**
   - Weighted ensemble of multiple architectures
   - Model weights based on validation accuracy
   - Combined predictions for improved robustness

# In[1]:


pip install pandas numpy librosa seaborn matplotlib scikit-learn ipython torch torchaudio xgboost lightgbm scikit-image


# In[2]:


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# In[3]:


# Import necessary libraries
import pandas as pd
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from skimage.transform import resize

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.cuda.amp import autocast, GradScaler

import warnings
warnings.filterwarnings("ignore")


# In[4]:


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

# Set seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# In[ ]:


# Define the file paths to different datasets containing emotional speech audio.

# RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
# This dataset contains audio recordings of speech performed with different emotions.
Ravdess = "dataset/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"

# CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
# This dataset contains audio-visual recordings of actors performing with various emotions.
Crema = "dataset/cremad/AudioWAV/"

# TESS (Toronto Emotional Speech Set)
# This dataset includes emotional speech data recorded by female speakers with different emotional expressions.
Tess = "dataset/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"

# SAVEE (Surrey Audio-Visual Expressed Emotion)
# This dataset includes audio recordings of speech expressing different emotions, recorded by male speakers.
Savee = "dataset/surrey-audiovisual-expressed-emotion-savee/ALL/"


# In[6]:


# List all actor directories in the RAVDESS dataset
ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []  # To store emotion labels
file_path = []  # To store file paths

# Iterate through each actor's directory and process audio files
for dir in ravdess_directory_list:
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0].split('-')  # Parse metadata from file name
        file_emotion.append(int(part[2]))  # Extract emotion label
        file_path.append(Ravdess + dir + '/' + file)  # Store full file path

# Create DataFrames for emotions and file paths
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])

# Combine emotion and path DataFrames
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# Map numeric emotion labels to descriptive names
Ravdess_df.Emotions.replace({
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 
    5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
}, inplace=True)

# Display the first few rows of the DataFrame
Ravdess_df.head()


# In[7]:


# List all files in the CREMA-D dataset
crema_directory_list = os.listdir(Crema)

file_emotion = []  # To store emotion labels
file_path = []  # To store file paths

# Iterate through files and extract file paths and emotions
for file in crema_directory_list:
    file_path.append(Crema + file)  # Store file path
    part = file.split('_')  # Parse metadata from file name
    # Map file metadata to emotion labels
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')

# Create DataFrames for emotions and file paths
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])

# Combine emotion and path DataFrames
Crema_df = pd.concat([emotion_df, path_df], axis=1)

# Display the first few rows of the DataFrame
Crema_df.head()


# In[8]:


# List all directories in the TESS dataset
tess_directory_list = os.listdir(Tess)

file_emotion = []  # To store emotion labels
file_path = []  # To store file paths

# Iterate through each directory and process files
for dir in tess_directory_list:
    directories = os.listdir(Tess + dir)  # List all files in the current directory
    for file in directories:
        part = file.split('.')[0].split('_')[2]  # Extract emotion from the file name
        # Map 'ps' to 'surprise', otherwise use the extracted part
        if part == 'ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)  # Store full file path

# Create DataFrames for emotions and file paths
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])

# Combine emotion and path DataFrames
Tess_df = pd.concat([emotion_df, path_df], axis=1)

# Display the first few rows of the DataFrame
Tess_df.head()


# In[9]:


# List all files in the SAVEE dataset
savee_directory_list = os.listdir(Savee)

file_emotion = []  # To store emotion labels
file_path = []  # To store file paths

# Iterate through each file and extract file paths and emotions
for file in savee_directory_list:
    file_path.append(Savee + file)  # Store full file path
    part = file.split('_')[1]  # Extract the emotion code from the file name
    ele = part[:-6]  # Remove the last 6 characters to isolate the emotion code
    # Map emotion codes to corresponding emotion labels
    if ele == 'a':
        file_emotion.append('angry')
    elif ele == 'd':
        file_emotion.append('disgust')
    elif ele == 'f':
        file_emotion.append('fear')
    elif ele == 'h':
        file_emotion.append('happy')
    elif ele == 'n':
        file_emotion.append('neutral')
    elif ele == 'sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')

# Create DataFrames for emotions and file paths
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])

# Combine emotion and path DataFrames
Savee_df = pd.concat([emotion_df, path_df], axis=1)

# Display the first few rows of the DataFrame
Savee_df.head()


# In[10]:


# Combine all the individual DataFrames (RAVDESS, CREMA-D, TESS, SAVEE) into a single DataFrame
data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis=0)

# Save the combined DataFrame to a CSV file for future use
data_path.to_csv("data_path.csv", index=False)

# Display the first few rows of the combined DataFrame
data_path.head()


# In[11]:


# Set the title of the plot with a larger font and bold styling
plt.title('Count of Emotions', size=16, weight='bold')

# Create a count plot for the 'Emotions' column with a custom color palette
sns.countplot(data_path.Emotions, palette='Set2', edgecolor='black')

# Label the axes with a larger font size and bold styling
plt.ylabel('Count', size=12, weight='bold')
plt.xlabel('Emotions', size=12, weight='bold')

# Customize the ticks for better readability
plt.xticks(rotation=45, ha='right', size=10)

# Remove the top and right spines for a cleaner look
sns.despine(top=True, right=True, left=False, bottom=False)

# Add gridlines to the background for better visibility of the counts
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()  # Adjust the plot to ensure everything fits
plt.show()


# In[12]:


# Function to create a waveplot for the audio data
def create_waveplot(data, sr, e):
    # Set the figure size for the waveplot
    plt.figure(figsize=(10, 3))
    # Set the title for the plot, displaying the emotion associated with the audio
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    # Display the waveplot using librosa's waveshow function
    librosa.display.waveshow(data, sr=sr)
    # Show the plot
    plt.show()

# Function to create a spectrogram for the audio data
def create_spectrogram(data, sr, e):
    # Convert the audio data into a Short-Time Fourier Transform (STFT)
    X = librosa.stft(data)
    # Convert the amplitude of the STFT to decibels (dB)
    Xdb = librosa.amplitude_to_db(abs(X))
    
    # Set the figure size for the spectrogram
    plt.figure(figsize=(12, 3))
    # Set the title for the plot, displaying the emotion associated with the audio
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    
    # Display the spectrogram using librosa's specshow function
    # The y-axis is set to 'hz' (frequency in Hz)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')  
    
    # Optionally, you can use 'log' for a logarithmic y-axis instead of 'hz'
    # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    
    # Display the colorbar for the spectrogram to represent dB values
    plt.colorbar()
    # Show the plot
    plt.show()


# In[15]:


# Set the emotion to visualize (e.g., 'fear')
emotion = 'fear'

# Get the file path for the specified emotion
path = np.array(data_path.Path[data_path.Emotions == emotion])[1]

# Load the audio file and get the data and sample rate
data, sampling_rate = librosa.load(path)

# Create a waveplot for the audio
create_waveplot(data, sampling_rate, emotion)

# Create a spectrogram for the audio
create_spectrogram(data, sampling_rate, emotion)


# In[17]:


# Set the emotion to visualize (e.g., 'angry')
emotion = 'angry'

# Get the file path for the specified emotion
path = np.array(data_path.Path[data_path.Emotions == emotion])[1]

# Load the audio file and get the data and sample rate
data, sampling_rate = librosa.load(path)

# Create a waveplot for the audio
create_waveplot(data, sampling_rate, emotion)

# Create a spectrogram for the audio
create_spectrogram(data, sampling_rate, emotion)



# In[19]:


# Set the emotion to visualize (e.g., 'sad')
emotion = 'sad'

# Get the file path for the specified emotion
path = np.array(data_path.Path[data_path.Emotions == emotion])[1]

# Load the audio file and get the data and sample rate
data, sampling_rate = librosa.load(path)

# Create a waveplot for the audio
create_waveplot(data, sampling_rate, emotion)

# Create a spectrogram for the audio
create_spectrogram(data, sampling_rate, emotion)



# In[20]:


# Set the emotion to visualize (e.g., 'happy')
emotion = 'happy'

# Get the file path for the specified emotion
path = np.array(data_path.Path[data_path.Emotions == emotion])[1]

# Load the audio file and get the data and sample rate
data, sampling_rate = librosa.load(path)

# Create a waveplot for the audio
create_waveplot(data, sampling_rate, emotion)

# Create a spectrogram for the audio
create_spectrogram(data, sampling_rate, emotion)

# In[ ]:


# Function to implement more aggressive regularization to reduce overfitting
def create_high_regularization_model(input_shape, num_classes):
    """
    Creates a CNN1D model with more aggressive regularization techniques
    to reduce overfitting and improve validation accuracy.
    """
    model = nn.Sequential(
        # First block with stronger regularization
        nn.Conv1d(input_shape[0], 32, kernel_size=5, padding=2),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.3),  # Higher dropout
        
        # Second block
        nn.Conv1d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(0.1),
        nn.MaxPool1d(2),
        nn.Dropout(0.4),  # Higher dropout
        
        # Third block
        nn.Conv1d(64, 128, kernel_size=5, padding=2),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(0.1),
        nn.MaxPool1d(2),
        nn.Dropout(0.5),  # Higher dropout
        
        # Fourth block
        nn.Conv1d(128, 128, kernel_size=5, padding=2),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(0.1),
        nn.MaxPool1d(2),
        nn.Dropout(0.5),  # Higher dropout
        
        # Global pooling
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        
        # Dense layers with strong regularization
        nn.Linear(128, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.6),  # Very high dropout
        
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.6),  # Very high dropout
        
        nn.Linear(128, num_classes)
    )
    
    # Apply weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    
    return model


# In[ ]:


# Enhanced data augmentation functions
def augment_audio_features(features, augmentation_factor=2):
    """
    Apply multiple augmentation techniques to audio features to increase dataset diversity
    and improve model generalization.
    
    Args:
        features: Audio features (MFCC or other)
        augmentation_factor: Number of augmented samples to create
    
    Returns:
        List of augmented features
    """
    augmented_features = [features]  # Start with original
    
    for _ in range(augmentation_factor):
        # Choose random augmentation technique
        aug_type = random.choice(['freq_mask', 'time_mask', 'noise', 'shift'])
        
        if aug_type == 'freq_mask':
            # Frequency masking (mask random frequency bands)
            aug_feature = features.clone()
            num_bands = random.randint(1, 3)
            for _ in range(num_bands):
                f_min = random.randint(0, features.shape[0] - 1)
                f_max = min(f_min + random.randint(1, 5), features.shape[0])
                aug_feature[f_min:f_max, :] = 0
            augmented_features.append(aug_feature)
            
        elif aug_type == 'time_mask':
            # Time masking (mask random time segments)
            aug_feature = features.clone()
            num_segments = random.randint(1, 3)
            for _ in range(num_segments):
                t_min = random.randint(0, features.shape[1] - 1)
                t_max = min(t_min + random.randint(1, 10), features.shape[1])
                aug_feature[:, t_min:t_max] = 0
            augmented_features.append(aug_feature)
            
        elif aug_type == 'noise':
            # Add random noise
            aug_feature = features.clone()
            noise_level = random.uniform(0.001, 0.02)
            noise = torch.randn_like(features) * noise_level
            aug_feature = aug_feature + noise
            augmented_features.append(aug_feature)
            
        else:  # shift
            # Time shift (roll the features in time dimension)
            aug_feature = features.clone()
            shift_amount = random.randint(-10, 10)
            aug_feature = torch.roll(aug_feature, shifts=shift_amount, dims=1)
            augmented_features.append(aug_feature)
    
    return augmented_features

# Function to apply augmentation to a batch of data
def augment_batch(features_batch, labels_batch, augmentation_factor=1):
    """
    Apply augmentation to a batch of features and duplicate corresponding labels
    
    Args:
        features_batch: Batch of features (B, C, L)
        labels_batch: Batch of labels (B)
        augmentation_factor: Number of augmented samples to create per original sample
        
    Returns:
        Augmented features and labels
    """
    augmented_features = []
    augmented_labels = []
    
    for i in range(len(features_batch)):
        # Get original sample and label
        features = features_batch[i]
        label = labels_batch[i]
        
        # Apply augmentation
        aug_features = augment_audio_features(features, augmentation_factor)
        
        # Add original and augmented features/labels
        augmented_features.extend(aug_features)
        augmented_labels.extend([label] * len(aug_features))
    
    # Convert lists to tensors
    augmented_features = torch.stack(augmented_features)
    augmented_labels = torch.tensor(augmented_labels, dtype=torch.long)
    
    return augmented_features, augmented_labels


# In[ ]:


# Calculate class weights to handle class imbalance
def calculate_class_weights(train_dataset, num_classes):
    """
    Calculate class weights inversely proportional to class frequencies
    to handle class imbalance.
    
    Args:
        train_dataset: Training dataset
        num_classes: Number of classes
        
    Returns:
        Tensor of class weights
    """
    # Count class frequencies
    class_counts = torch.zeros(num_classes)
    for _, label in train_dataset:
        class_counts[label] += 1
    
    # Calculate weights inversely proportional to class frequencies
    total_samples = len(train_dataset)
    class_weights = total_samples / (num_classes * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return class_weights


# In[ ]:


# Advanced CNN1D model with residual connections and squeeze-excitation blocks
class ResidualBlock(nn.Module):
    """
    Residual block with squeeze-excitation mechanism
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
        # Squeeze-Excitation block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Squeeze-Excitation attention
        se_weight = self.se(out)
        out = out * se_weight
        
        # Add residual connection
        out += residual
        out = self.relu(out)
        
        return out

class AdvancedCNN1D(nn.Module):
    """
    Advanced CNN1D model with residual connections and attention mechanisms
    """
    def __init__(self, input_shape, num_classes):
        super(AdvancedCNN1D, self).__init__()
        
        # Initial convolution block
        self.initial_block = nn.Sequential(
            nn.Conv1d(input_shape[0], 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Residual blocks
        self.res_block1 = ResidualBlock(64)
        self.transition1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.res_block2 = ResidualBlock(128)
        self.transition2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.res_block3 = ResidualBlock(256)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Initial convolution
        x = self.initial_block(x)
        
        # Residual blocks with transitions
        x = self.res_block1(x)
        x = self.transition1(x)
        
        x = self.res_block2(x)
        x = self.transition2(x)
        
        x = self.res_block3(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x


# In[ ]:


# Train with advanced techniques
# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Calculate class weights to handle imbalance
class_weights = calculate_class_weights(train_dataset, num_classes)
print("Class weights:", class_weights)

# Create advanced model
advanced_model = AdvancedCNN1D(input_shape, num_classes)
print("Advanced model created with residual connections and squeeze-excitation blocks")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Train with advanced techniques
print("Starting training with advanced techniques...")
advanced_model, advanced_history = train_with_advanced_techniques(
    advanced_model, train_loader, val_loader, criterion,
    num_epochs=150,  # Increase max epochs
    device=device,
    patience=25,     # Increase patience for early stopping
    class_weights=class_weights  # Use class weights
)

# Save the model
torch.save(advanced_model.state_dict(), 'best_advanced_emotion_recognition_model.pth')
print("Advanced model saved to 'best_advanced_emotion_recognition_model.pth'")

# Plot training and validation metrics
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(advanced_history['train_loss'], label='Train Loss')
plt.plot(advanced_history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(advanced_history['train_acc'], label='Train Acc')
plt.plot(advanced_history['val_acc'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on test set
advanced_model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = advanced_model(inputs)
        loss = criterion(outputs, targets)
        
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

test_loss = test_loss / len(test_loader)
test_acc = correct / total

print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')
# In[ ]:


# Enhanced CNN1D model with self-attention mechanism
class AttentionCNN1D(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(AttentionCNN1D, self).__init__()
        
        # Self-attention mechanism
        class SelfAttention(nn.Module):
            def __init__(self, in_dim):
                super(SelfAttention, self).__init__()
                self.query = nn.Conv1d(in_dim, in_dim//8, kernel_size=1)
                self.key = nn.Conv1d(in_dim, in_dim//8, kernel_size=1)
                self.value = nn.Conv1d(in_dim, in_dim, kernel_size=1)
                self.gamma = nn.Parameter(torch.zeros(1))
                self.softmax = nn.Softmax(dim=-1)
                
            def forward(self, x):
                batch_size, C, width = x.size()
                
                # Compute query, key, value projections
                proj_query = self.query(x).view(batch_size, -1, width).permute(0, 2, 1)  # B x W x C'
                proj_key = self.key(x).view(batch_size, -1, width)  # B x C' x W
                energy = torch.bmm(proj_query, proj_key)  # B x W x W
                attention = self.softmax(energy)  # B x W x W
                
                proj_value = self.value(x).view(batch_size, -1, width)  # B x C x W
                out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x W
                out = out.view(batch_size, C, width)
                
                out = self.gamma * out + x
                return out
        
        # Initial convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_shape[0], 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        
        # Self-attention layer
        self.attention = SelfAttention(256)
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Add noise for robustness during training
        if self.training:
            x = x + 0.01 * torch.randn_like(x)
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Apply self-attention
        x = self.attention(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Fully connected layers
        x = self.fc(x)
        
        return x


# In[ ]:


# Enhanced data augmentation for better generalization
def enhanced_augment_audio(audio, sr, augment_types=None, augment_prob=0.7):
    """
    Apply multiple advanced augmentation techniques to audio data
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        augment_types: List of augmentation types to apply (if None, will use all)
        augment_prob: Probability of applying each augmentation
        
    Returns:
        Augmented audio
    """
    if augment_types is None:
        augment_types = ['noise', 'pitch', 'stretch', 'shift', 'filter', 'speed', 'reverb']
    
    # Make a copy of the original audio
    augmented = audio.copy()
    
    # Apply noise augmentation
    if 'noise' in augment_types and np.random.random() < augment_prob:
        noise_level = np.random.uniform(0.001, 0.015)
        noise = np.random.normal(0, noise_level, len(augmented))
        augmented = augmented + noise
    
    # Apply pitch shift augmentation
    if 'pitch' in augment_types and np.random.random() < augment_prob:
        n_steps = np.random.uniform(-3, 3)
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)
    
    # Apply time stretch augmentation
    if 'stretch' in augment_types and np.random.random() < augment_prob:
        rate = np.random.uniform(0.85, 1.15)
        augmented = librosa.effects.time_stretch(augmented, rate=rate)
        
        # Ensure fixed length after stretching
        if len(augmented) < len(audio):
            augmented = np.pad(augmented, (0, len(audio) - len(augmented)))
        else:
            augmented = augmented[:len(audio)]
    
    # Apply time shift augmentation
    if 'shift' in augment_types and np.random.random() < augment_prob:
        shift_factor = np.random.uniform(-0.2, 0.2)
        shift_amount = int(len(augmented) * shift_factor)
        
        if shift_amount > 0:
            augmented = np.pad(augmented, (shift_amount, 0))[:len(augmented)]
        else:
            augmented = np.pad(augmented, (0, -shift_amount))[:-shift_amount]
    
    # Apply filter augmentation
    if 'filter' in augment_types and np.random.random() < augment_prob:
        filter_type = np.random.choice(['lowpass', 'highpass'])
        
        if filter_type == 'lowpass':
            # Simple lowpass filter
            augmented = librosa.effects.preemphasis(augmented, coef=0.97)
        else:
            # Simple highpass filter
            augmented = augmented - librosa.effects.preemphasis(augmented, coef=0.97)
    
    # Apply speed tuning
    if 'speed' in augment_types and np.random.random() < augment_prob:
        speed_factor = np.random.uniform(0.9, 1.1)
        
        if speed_factor == 1.0:  # No change
            pass
        else:
            # Resample to change speed
            new_length = int(len(augmented) / speed_factor)
            augmented = librosa.resample(augmented, orig_sr=new_length, target_sr=len(augmented))
            
            # Ensure the output is the same length as input
            if len(augmented) > len(audio):
                augmented = augmented[:len(audio)]
            else:
                augmented = np.pad(augmented, (0, max(0, len(audio) - len(augmented))), mode='constant')
    
    # Apply reverb effect (simple approximation)
    if 'reverb' in augment_types and np.random.random() < augment_prob:
        reverb_delay = np.random.randint(1000, 3000)
        decay = np.random.uniform(0.1, 0.5)
        
        # Create simple impulse response
        impulse_response = np.zeros(reverb_delay)
        impulse_response[0] = 1
        impulse_response[reverb_delay // 2] = decay / 2
        impulse_response[reverb_delay - 1] = decay / 4
        
        # Apply convolution for reverb effect
        reverb_audio = np.convolve(augmented, impulse_response, mode='full')[:len(augmented)]
        
        # Mix with original
        mix_ratio = np.random.uniform(0.5, 0.9)
        augmented = mix_ratio * augmented + (1 - mix_ratio) * reverb_audio
    
    return augmented


# In[ ]:


# Advanced feature extraction with multiple feature types
def extract_advanced_features(audio, sr):
    """
    Extract multiple types of audio features and combine them
    for better emotion recognition performance
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # 1. Extract MFCCs with improved parameters
    n_mfcc = 40
    mfccs = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512,
        lifter=22  # Liftering parameter to emphasize higher MFCCs
    )
    
    # Add delta and delta-delta features
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    # Combine MFCC features
    features['mfcc'] = np.concatenate([mfccs, mfcc_delta, mfcc_delta2])
    
    # 2. Extract enhanced mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=128,
        n_fft=2048, 
        hop_length=512,
        fmax=sr/2,  # Include full frequency range
        power=2.0    # Use power spectrogram
    )
    
    # Convert to log scale
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
    
    # Normalize
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
    
    features['mel'] = log_mel
    
    # 3. Extract chroma features (useful for tonal content)
    chroma = librosa.feature.chroma_stft(
        y=audio, 
        sr=sr,
        n_fft=2048,
        hop_length=512
    )
    features['chroma'] = chroma
    
    # 4. Extract spectral contrast (useful for voice/music discrimination)
    contrast = librosa.feature.spectral_contrast(
        y=audio, 
        sr=sr,
        n_fft=2048,
        hop_length=512
    )
    features['contrast'] = contrast
    
    # 5. Extract tonnetz features (harmonic content)
    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(audio), 
        sr=sr
    )
    features['tonnetz'] = tonnetz
    
    # 6. Extract spectral features
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio, 
        sr=sr,
        n_fft=2048,
        hop_length=512
    )
    
    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, 
        sr=sr,
        n_fft=2048,
        hop_length=512
    )
    
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, 
        sr=sr,
        n_fft=2048,
        hop_length=512
    )
    
    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=2048,
        hop_length=512
    )
    
    # Combine spectral features
    spectral_features = np.concatenate([centroid, bandwidth, rolloff, flatness])
    features['spectral'] = spectral_features
    
    # 7. Zero crossing rate (useful for voice/unvoiced discrimination)
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        hop_length=512
    )
    features['zcr'] = zcr
    
    # 8. RMS energy
    rms = librosa.feature.rms(
        y=audio,
        hop_length=512
    )
    features['rms'] = rms
    
    return features


# In[ ]:


# Enhanced feature extraction pipeline with augmentation
def extract_enhanced_features(data_path, augment=True, max_samples_per_class=None, augment_factor=2):
    """
    Extract enhanced features from audio files with augmentation
    
    Args:
        data_path: DataFrame with file paths and emotion labels
        augment: Whether to apply augmentation
        max_samples_per_class: Maximum number of samples per class (for balancing)
        augment_factor: Number of augmented samples to create per original sample
        
    Returns:
        X_1d: Features for 1D models (MFCCs, spectral features, etc.)
        X_2d: Features for 2D models (Mel spectrograms)
        y: Labels
    """
    # Lists to store features and labels
    X_1d_list = []
    X_2d_list = []
    y_list = []
    
    # Group by emotion for balanced sampling
    grouped = data_path.groupby('Emotions')
    
    # Process each emotion group
    for emotion, group in grouped:
        print(f"Processing {len(group)} samples for emotion: {emotion}")
        
        # Sample if max_samples_per_class is specified
        if max_samples_per_class and len(group) > max_samples_per_class:
            group = group.sample(max_samples_per_class, random_state=42)
        
        # Process each audio file
        for idx, row in group.iterrows():
            try:
                # Load audio
                audio, sr = librosa.load(row['Path'], duration=3.0)
                
                # Ensure fixed length
                if len(audio) < sr * 3.0:
                    audio = np.pad(audio, (0, int(sr * 3.0) - len(audio)))
                else:
                    audio = audio[:int(sr * 3.0)]
                
                # Extract features for original audio
                features = extract_advanced_features(audio, sr)
                
                # Prepare 1D features (concatenate all 1D features)
                mfcc_features = features['mfcc']  # Shape: (120, time_steps)
                spectral_features = features['spectral']  # Shape: (4, time_steps)
                chroma_features = features['chroma']  # Shape: (12, time_steps)
                zcr_features = features['zcr']  # Shape: (1, time_steps)
                rms_features = features['rms']  # Shape: (1, time_steps)
                
                # Standardize time dimension to 128 frames
                target_length = 128
                
                # Resize all features to have 128 time steps
                mfcc_resized = resize(mfcc_features, (mfcc_features.shape[0], target_length))
                spectral_resized = resize(spectral_features, (spectral_features.shape[0], target_length))
                chroma_resized = resize(chroma_features, (chroma_features.shape[0], target_length))
                zcr_resized = resize(zcr_features, (zcr_features.shape[0], target_length))
                rms_resized = resize(rms_features, (rms_features.shape[0], target_length))
                
                # Concatenate all 1D features
                X_1d = np.concatenate([
                    mfcc_resized,
                    spectral_resized,
                    chroma_resized,
                    zcr_resized,
                    rms_resized
                ])  # Shape: (features, time_steps)
                
                # Prepare 2D features (mel spectrogram)
                X_2d = resize(features['mel'], (128, 128))  # Resize to 128x128
                X_2d = np.stack([X_2d] * 3, axis=-1)  # Add 3 channels (like RGB)
                
                # Add to lists
                X_1d_list.append(X_1d)
                X_2d_list.append(X_2d)
                y_list.append(emotion)
                
                # Apply augmentation if enabled
                if augment:
                    for _ in range(augment_factor):
                        # Apply augmentation to audio
                        aug_audio = enhanced_augment_audio(audio, sr)
                        
                        # Extract features for augmented audio
                        aug_features = extract_advanced_features(aug_audio, sr)
                        
                        # Prepare 1D features for augmented audio
                        aug_mfcc = resize(aug_features['mfcc'], (aug_features['mfcc'].shape[0], target_length))
                        aug_spectral = resize(aug_features['spectral'], (aug_features['spectral'].shape[0], target_length))
                        aug_chroma = resize(aug_features['chroma'], (aug_features['chroma'].shape[0], target_length))
                        aug_zcr = resize(aug_features['zcr'], (aug_features['zcr'].shape[0], target_length))
                        aug_rms = resize(aug_features['rms'], (aug_features['rms'].shape[0], target_length))
                        
                        # Concatenate all 1D features for augmented audio
                        aug_X_1d = np.concatenate([
                            aug_mfcc,
                            aug_spectral,
                            aug_chroma,
                            aug_zcr,
                            aug_rms
                        ])
                        
                        # Prepare 2D features for augmented audio
                        aug_X_2d = resize(aug_features['mel'], (128, 128))
                        aug_X_2d = np.stack([aug_X_2d] * 3, axis=-1)
                        
                        # Add augmented features to lists
                        X_1d_list.append(aug_X_1d)
                        X_2d_list.append(aug_X_2d)
                        y_list.append(emotion)
                
            except Exception as e:
                print(f"Error processing {row['Path']}: {e}")
    
    # Convert lists to numpy arrays
    X_1d = np.array(X_1d_list)
    X_2d = np.array(X_2d_list)
    y = np.array(y_list)
    
    print(f"Extracted features: 1D shape={X_1d.shape}, 2D shape={X_2d.shape}")
    
    return X_1d, X_2d, y


# In[ ]:


# Advanced training function with mixup, cutmix, and learning rate scheduling
def train_advanced_model(model, train_loader, val_loader, criterion, device, 
                        num_epochs=100, learning_rate=0.001, weight_decay=0.01):
    """
    Train model with advanced techniques:
    - Mixup and CutMix augmentation
    - Cosine annealing with warm restarts
    - Gradient clipping
    - Mixed precision training
    - Early stopping with patience
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to train on
        num_epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        
    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler - cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # For storing metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # For early stopping
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None
    
    # Mixup function
    def mixup(x, y, alpha=0.2):
        """Applies mixup augmentation to a batch"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    # CutMix function
    def cutmix(x, y, alpha=1.0):
        """Applies cutmix augmentation to a batch"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)
        
        # For 1D data (B, C, L)
        if len(x.shape) == 3:
            bbx1 = np.random.randint(0, x.size(2))
            bbx2 = np.random.randint(bbx1, x.size(2))
            
            # Create mixed batch
            mixed_x = x.clone()
            mixed_x[:, :, bbx1:bbx2] = x[index, :, bbx1:bbx2]
            
            # Adjust lambda to reflect the proportion of the image that was mixed
            lam = 1 - ((bbx2 - bbx1) / x.size(2))
            
        # For 2D data (B, C, H, W)
        elif len(x.shape) == 4:
            bbx1 = np.random.randint(0, x.size(2))
            bby1 = np.random.randint(0, x.size(3))
            bbx2 = np.random.randint(bbx1, x.size(2))
            bby2 = np.random.randint(bby1, x.size(3))
            
            # Create mixed batch
            mixed_x = x.clone()
            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda to reflect the proportion of the image that was mixed
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))
            
        else:
            return x, y, y, 1.0  # No CutMix for other dimensions
            
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply augmentation (mixup or cutmix) with 70% probability
            aug_type = np.random.choice(['none', 'mixup', 'cutmix'], p=[0.3, 0.35, 0.35])
            
            if aug_type == 'mixup':
                inputs, targets_a, targets_b, lam = mixup(inputs, targets)
                aug_applied = True
            elif aug_type == 'cutmix':
                inputs, targets_a, targets_b, lam = cutmix(inputs, targets)
                aug_applied = True
            else:
                aug_applied = False
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                
                if aug_applied:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Update scheduler
            scheduler.step(epoch + batch_idx / len(train_loader))
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            
            if aug_applied:
                # For augmented data, use weighted accuracy
                train_correct += (lam * predicted.eq(targets_a).sum().float() + 
                                 (1 - lam) * predicted.eq(targets_b).sum().float())
            else:
                train_correct += predicted.eq(targets).sum().item()
        
        # Calculate average loss and accuracy
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate average loss and accuracy
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f'Validation loss decreased. Saving model...')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history


# In[ ]:


# Main execution with enhanced models and techniques
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Starting enhanced speech emotion recognition training...")
    
    try:
        # Load the dataset
        data_path = pd.read_csv("data_path.csv")
        print(f"Loaded dataset with {len(data_path)} samples")
        
        # Extract enhanced features with augmentation and class balancing
        print("\nExtracting enhanced features...")
        X_1d, X_2d, y = extract_enhanced_features(
            data_path, 
            augment=True,
            max_samples_per_class=1000,  # Balance classes
            augment_factor=2  # Generate 2 augmented samples per original
        )
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        num_classes = len(le.classes_)
        print(f"Classes: {le.classes_}")
        
        # Split data with stratification
        X_1d_train, X_1d_test, X_2d_train, X_2d_test, y_train, y_test = train_test_split(
            X_1d, X_2d, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Create PyTorch datasets
        train_dataset_1d = EmotionDataset(X_1d_train, y_train)
        test_dataset_1d = EmotionDataset(X_1d_test, y_test)
        train_dataset_2d = EmotionDataset(X_2d_train, y_train)
        test_dataset_2d = EmotionDataset(X_2d_test, y_test)
        
        # Create data loaders
        batch_size = 32  # Smaller batch size for better generalization
        train_loader_1d = DataLoader(train_dataset_1d, batch_size=batch_size, shuffle=True)
        val_loader_1d = DataLoader(test_dataset_1d, batch_size=batch_size)
        train_loader_2d = DataLoader(train_dataset_2d, batch_size=batch_size, shuffle=True)
        val_loader_2d = DataLoader(test_dataset_2d, batch_size=batch_size)
        
        # Calculate class weights to handle imbalance
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = torch.tensor(total_samples / (num_classes * class_counts), dtype=torch.float32).to(device)
        
        # Create weighted loss function
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Create enhanced models
        print("\nCreating enhanced models...")
        
        # 1. Attention-based CNN1D model
        attention_model = AttentionCNN1D(
            input_shape=(X_1d_train.shape[1], X_1d_train.shape[2]),
            num_classes=num_classes
        ).to(device)
        print("Created attention-based CNN1D model")
        
        # 2. CNN2D model with ResNet architecture for 2D features
        class ResNet2D(nn.Module):
            def __init__(self, input_shape, num_classes):
                super(ResNet2D, self).__init__()
                
                # Use a pretrained ResNet but modify for our input shape
                self.model = torchvision.models.resnet18(pretrained=True)
                
                # Modify first layer to accept our input shape
                self.model.conv1 = nn.Conv2d(
                    input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                
                # Modify final layer for our number of classes
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
                
            def forward(self, x):
                return self.model(x)
        
        # Create ResNet2D model
        resnet_model = ResNet2D(
            input_shape=(X_2d_train.shape[1], X_2d_train.shape[2], X_2d_train.shape[3]),
            num_classes=num_classes
        ).to(device)
        print("Created ResNet2D model")
        
        # 3. Enhanced CNN-BiLSTM model with attention
        class EnhancedCNNBiLSTM(nn.Module):
            def __init__(self, input_shape, num_classes):
                super(EnhancedCNNBiLSTM, self).__init__()
                
                # CNN layers
                self.conv = nn.Sequential(
                    nn.Conv1d(input_shape[0], 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(4),
                    nn.Dropout(0.3),
                    
                    nn.Conv1d(64, 128, kernel_size=5, padding=2),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(4),
                    nn.Dropout(0.3)
                )
                
                # BiLSTM layers
                self.lstm = nn.LSTM(
                    input_size=128,
                    hidden_size=128,
                    num_layers=2,
                    bidirectional=True,
                    dropout=0.3,
                    batch_first=True
                )
                
                # Attention mechanism
                self.attention = nn.Sequential(
                    nn.Linear(256, 64),  # 256 = 128*2 (bidirectional)
                    nn.Tanh(),
                    nn.Linear(64, 1),
                    nn.Softmax(dim=1)
                )
                
                # Fully connected layers
                self.fc = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, num_classes)
                )
                
            def forward(self, x):
                # CNN feature extraction
                x = self.conv(x)
                
                # Reshape for LSTM: (batch, channels, seq_len) -> (batch, seq_len, channels)
                x = x.transpose(1, 2)
                
                # BiLSTM
                x, _ = self.lstm(x)  # x shape: (batch, seq_len, 2*hidden_size)
                
                # Attention mechanism
                attn_weights = self.attention(x)  # Shape: (batch, seq_len, 1)
                context = torch.sum(attn_weights * x, dim=1)  # Shape: (batch, 2*hidden_size)
                
                # Classification
                output = self.fc(context)
                
                return output
        
        # Create enhanced CNN-BiLSTM model
        bilstm_model = EnhancedCNNBiLSTM(
            input_shape=(X_1d_train.shape[1], X_1d_train.shape[2]),
            num_classes=num_classes
        ).to(device)
        print("Created enhanced CNN-BiLSTM model")
        
        # Train models with advanced techniques
        print("\nTraining attention-based CNN1D model...")
        attention_model, history_attention = train_advanced_model(
            attention_model, train_loader_1d, val_loader_1d, criterion, device,
            num_epochs=150, learning_rate=0.0005, weight_decay=0.01
        )
        
        print("\nTraining ResNet2D model...")
        resnet_model, history_resnet = train_advanced_model(
            resnet_model, train_loader_2d, val_loader_2d, criterion, device,
            num_epochs=150, learning_rate=0.0005, weight_decay=0.01
        )
        
        print("\nTraining enhanced CNN-BiLSTM model...")
        bilstm_model, history_bilstm = train_advanced_model(
            bilstm_model, train_loader_1d, val_loader_1d, criterion, device,
            num_epochs=150, learning_rate=0.0005, weight_decay=0.01
        )
        
        # Plot training history
        plt.figure(figsize=(15, 10))
        
        # Plot training accuracy
        plt.subplot(2, 2, 1)
        plt.plot(history_attention['train_acc'], label='Attention CNN')
        plt.plot(history_resnet['train_acc'], label='ResNet')
        plt.plot(history_bilstm['train_acc'], label='CNN-BiLSTM')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot validation accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history_attention['val_acc'], label='Attention CNN')
        plt.plot(history_resnet['val_acc'], label='ResNet')
        plt.plot(history_bilstm['val_acc'], label='CNN-BiLSTM')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot training loss
        plt.subplot(2, 2, 3)
        plt.plot(history_attention['train_loss'], label='Attention CNN')
        plt.plot(history_resnet['train_loss'], label='ResNet')
        plt.plot(history_bilstm['train_loss'], label='CNN-BiLSTM')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot validation loss
        plt.subplot(2, 2, 4)
        plt.plot(history_attention['val_loss'], label='Attention CNN')
        plt.plot(history_resnet['val_loss'], label='ResNet')
        plt.plot(history_bilstm['val_loss'], label='CNN-BiLSTM')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Create and evaluate ensemble model
        print("\nCreating ensemble model...")
        
        # Define ensemble model
        class EnhancedEnsemble:
            def __init__(self, models, model_types, weights=None):
                self.models = models
                self.model_types = model_types
                
                if weights is None:
                    self.weights = [1/len(models)] * len(models)
                else:
                    # Normalize weights
                    total = sum(weights)
                    self.weights = [w/total for w in weights]
            
            def predict(self, inputs_1d, inputs_2d=None):
                """Make prediction using the ensemble"""
                all_probs = []
                
                # Get predictions from each model
                for i, (model, model_type) in enumerate(zip(self.models, self.model_types)):
                    model.eval()
                    with torch.no_grad():
                        # Select appropriate input based on model type
                        if model_type == '1d':
                            outputs = model(inputs_1d)
                        else:  # '2d'
                            outputs = model(inputs_2d)
                        
                        # Get probabilities
                        probs = F.softmax(outputs, dim=1)
                        
                        # Weight probabilities
                        weighted_probs = probs * self.weights[i]
                        all_probs.append(weighted_probs)
                
                # Sum weighted probabilities
                ensemble_probs = sum(all_probs)
                
                # Get predicted class
                _, predicted = ensemble_probs.max(1)
                
                return predicted, ensemble_probs
        
        # Calculate validation accuracy for each model
        models = [attention_model, resnet_model, bilstm_model]
        model_types = ['1d', '2d', '1d']
        val_accuracies = []
        
        for model, model_type in zip(models, model_types):
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                # Select appropriate loader based on model type
                val_loader = val_loader_1d if model_type == '1d' else val_loader_2d
                
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_acc = correct / total
            val_accuracies.append(val_acc)
            print(f"{model_type.upper()} model validation accuracy: {val_acc:.4f}")
        
        # Create ensemble with weights based on validation accuracy
        ensemble = EnhancedEnsemble(models, model_types, weights=val_accuracies)
        
        # Evaluate ensemble on validation set
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        # Process validation set in batches
        for (inputs_1d, targets_1d), (inputs_2d, _) in zip(val_loader_1d, val_loader_2d):
            inputs_1d, inputs_2d = inputs_1d.to(device), inputs_2d.to(device)
            targets = targets_1d.to(device)
            
            # Get ensemble predictions
            predicted, _ = ensemble.predict(inputs_1d, inputs_2d)
            
            # Update statistics
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Calculate ensemble accuracy
        ensemble_acc = correct / total
        print(f"Ensemble validation accuracy: {ensemble_acc:.4f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=le.classes_,
                   yticklabels=le.classes_)
        plt.title('Ensemble Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds, target_names=le.classes_))
        
        # Save models
        print("\nSaving models...")
        torch.save(attention_model.state_dict(), 'best_attention_cnn_model.pth')
        torch.save(resnet_model.state_dict(), 'best_resnet_model.pth')
        torch.save(bilstm_model.state_dict(), 'best_bilstm_model.pth')
        
        # Save ensemble model components
        torch.save({
            'attention_model': attention_model.state_dict(),
            'resnet_model': resnet_model.state_dict(),
            'bilstm_model': bilstm_model.state_dict(),
            'weights': ensemble.weights,
            'model_types': ensemble.model_types,
            'classes': le.classes_
        }, 'best_ensemble_emotion_recognition_model.pth')
        
        print("All models saved successfully!")
        
    except Exception as e:
        print(f"Error in training process: {str(e)}")
        import traceback
        traceback.print_exc()


# In[ ]:


# Prediction function for the enhanced ensemble model
def predict_emotion_with_ensemble(audio_path, model_path='best_ensemble_emotion_recognition_model.pth'):
    """
    Predict emotion from audio using the enhanced ensemble model
    
    Args:
        audio_path: Path to audio file
        model_path: Path to saved ensemble model
        
    Returns:
        predicted_emotion: Predicted emotion label
        probabilities: Probability distribution over emotions
    """
    try:
        # Load the ensemble model
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get class labels
        classes = checkpoint['classes']
        
        # Create models
        attention_model = AttentionCNN1D(
            input_shape=(138, 128),  # Expected input shape for 1D models
            num_classes=len(classes)
        ).to(device)
        
        resnet_model = ResNet2D(
            input_shape=(3, 128, 128),  # Expected input shape for 2D models
            num_classes=len(classes)
        ).to(device)
        
        bilstm_model = EnhancedCNNBiLSTM(
            input_shape=(138, 128),  # Expected input shape for 1D models
            num_classes=len(classes)
        ).to(device)
        
        # Load model weights
        attention_model.load_state_dict(checkpoint['attention_model'])
        resnet_model.load_state_dict(checkpoint['resnet_model'])
        bilstm_model.load_state_dict(checkpoint['bilstm_model'])
        
        # Get model weights and types
        weights = checkpoint['weights']
        model_types = checkpoint['model_types']
        
        # Create ensemble
        models = [attention_model, resnet_model, bilstm_model]
        ensemble = EnhancedEnsemble(models, model_types, weights)
        
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, duration=3.0)
        
        # Ensure fixed length
        if len(audio) < sr * 3.0:
            audio = np.pad(audio, (0, int(sr * 3.0) - len(audio)))
        else:
            audio = audio[:int(sr * 3.0)]
        
        # Extract features
        features = extract_advanced_features(audio, sr)
        
        # Prepare 1D features
        mfcc_features = features['mfcc']
        spectral_features = features['spectral']
        chroma_features = features['chroma']
        zcr_features = features['zcr']
        rms_features = features['rms']
        
        # Standardize time dimension
        target_length = 128
        
        # Resize features
        mfcc_resized = resize(mfcc_features, (mfcc_features.shape[0], target_length))
        spectral_resized = resize(spectral_features, (spectral_features.shape[0], target_length))
        chroma_resized = resize(chroma_features, (chroma_features.shape[0], target_length))
        zcr_resized = resize(zcr_features, (zcr_features.shape[0], target_length))
        rms_resized = resize(rms_features, (rms_features.shape[0], target_length))
        
        # Concatenate 1D features
        X_1d = np.concatenate([
            mfcc_resized,
            spectral_resized,
            chroma_resized,
            zcr_resized,
            rms_resized
        ])
        
        # Prepare 2D features
        X_2d = resize(features['mel'], (128, 128))
        X_2d = np.stack([X_2d] * 3, axis=-1)
        
        # Convert to PyTorch tensors
        X_1d_tensor = torch.FloatTensor(X_1d).unsqueeze(0).to(device)  # Add batch dimension
        X_2d_tensor = torch.FloatTensor(X_2d).permute(2, 0, 1).unsqueeze(0).to(device)  # (H,W,C) -> (1,C,H,W)
        
        # Get ensemble prediction
        predicted, probs = ensemble.predict(X_1d_tensor, X_2d_tensor)
        
        # Convert to numpy
        predicted_idx = predicted.item()
        probabilities = probs.cpu().numpy()[0]
        
        # Get predicted emotion
        predicted_emotion = classes[predicted_idx]
        
        return predicted_emotion, probabilities, classes
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Example usage
def test_prediction():
    # Get a random sample from the dataset
    data_path = pd.read_csv("data_path.csv")
    sample = data_path.sample(1).iloc[0]
    
    # Get the audio path and true emotion
    audio_path = sample['Path']
    true_emotion = sample['Emotions']
    
    # Make prediction
    predicted_emotion, probabilities, classes = predict_emotion_with_ensemble(audio_path)
    
    # Print results
    print(f"Audio: {audio_path}")
    print(f"True emotion: {true_emotion}")
    print(f"Predicted emotion: {predicted_emotion}")
    
    # Print probabilities
    print("\nProbabilities:")
    for i, emotion in enumerate(classes):
        print(f"{emotion}: {probabilities[i]:.4f}")
    
    # Plot probabilities
    plt.figure(figsize=(10, 5))
    plt.bar(classes, probabilities)
    plt.title('Emotion Prediction Probabilities')
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Load and play audio if IPython is available
    try:
        from IPython.display import Audio, display
        print("\nPlaying audio sample...")
        display(Audio(audio_path, autoplay=True))
    except ImportError:
        print("\nIPython not available. Cannot play audio.")

# Run test prediction if the model exists
if os.path.exists('best_ensemble_emotion_recognition_model.pth'):
    test_prediction()
else:
    print("Ensemble model not found. Train the model first.")


# In[ ]:


# Create and evaluate ensemble of models
# Load all models for ensemble
try:
    # Try to load the models if they exist
    # Load advanced CNN1D model
    advanced_model = AdvancedCNN1D(input_shape, num_classes)
    advanced_model.load_state_dict(torch.load('best_advanced_emotion_recognition_model.pth'))
    advanced_model = advanced_model.to(device)
    
    # Load original CNN1D model if it exists
    cnn1d_model = CNN1D(input_shape, num_classes)
    cnn1d_model.load_state_dict(torch.load('best_emotion_recognition_model.pth'))
    cnn1d_model = cnn1d_model.to(device)
    
    # Load CNN2D model if it exists
    cnn2d_model = CNN2D(input_shape_2d, num_classes)
    cnn2d_model.load_state_dict(torch.load('best_emotion_recognition_model_2d.pth'))
    cnn2d_model = cnn2d_model.to(device)
    
    # Load BiLSTM model if it exists
    bilstm_model = CNNBiLSTM(input_shape, num_classes)
    bilstm_model.load_state_dict(torch.load('best_emotion_recognition_model_bilstm.pth'))
    bilstm_model = bilstm_model.to(device)
    
    # Create list of available models
    models = []
    if 'advanced_model' in locals():
        models.append(advanced_model)
    if 'cnn1d_model' in locals():
        models.append(cnn1d_model)
    if 'cnn2d_model' in locals():
        models.append(cnn2d_model)
    if 'bilstm_model' in locals():
        models.append(bilstm_model)
    
    # Create and evaluate ensemble if we have at least 2 models
    if len(models) >= 2:
        print(f"Creating ensemble with {len(models)} models")
        ensemble, ensemble_acc = create_and_evaluate_ensemble(models, val_loader, test_loader, device)
        print(f"Final ensemble test accuracy: {ensemble_acc:.4f}")
    else:
        print("Not enough models available for ensemble. Train more models first.")
        
except Exception as e:
    print(f"Error creating ensemble: {str(e)}")
    print("Make sure to train all models first before creating ensemble.")

