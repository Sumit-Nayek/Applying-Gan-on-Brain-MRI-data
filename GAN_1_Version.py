"""
## **1. Setup and Installation**
"""

# Install required packages
!pip install kaggle tensorflow matplotlib numpy pandas opencv-python scikit-image

# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import zipfile
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")

"""
## **2. Kaggle Dataset Setup**
"""

# Upload Kaggle API credentials
from google.colab import files
print("Please upload your kaggle.json file:")
files.upload()

# Configure Kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download Brain MRI dataset
print("Downloading Brain MRI dataset...")
!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset

# Extract the dataset
print("Extracting dataset...")
with zipfile.ZipFile('brain-tumor-mri-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('brain_mri_data')

# Alternative dataset if the above fails
!kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri

print("Dataset extracted successfully!")

"""
## **3. Data Loading and Preprocessing**
"""

def load_and_preprocess_images(data_path, img_size=(128, 128), max_images=None):
    """
    Load and preprocess MRI images from the dataset
    
    Args:
        data_path: Path to the dataset directory
        img_size: Target image size (height, width)
        max_images: Maximum number of images to load
    
    Returns:
        Array of preprocessed images
    """
    images = []
    
    # Walk through all directories
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                try:
                    # Read and preprocess image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize
                        img = cv2.resize(img, img_size)
                        # Normalize to [-1, 1] (better for GAN training)
                        img = (img.astype(np.float32) - 127.5) / 127.5
                        # Add channel dimension
                        img = np.expand_dims(img, axis=-1)
                        images.append(img)
                        
                        if max_images and len(images) >= max_images:
                            return np.array(images)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
    
    return np.array(images)

# Load the dataset
print("Loading and preprocessing MRI images...")
data_path = 'brain_mri_data'
images = load_and_preprocess_images(data_path, img_size=(128, 128), max_images=5000)

print(f"Loaded {len(images)} images")
print(f"Image shape: {images[0].shape if len(images) > 0 else 'No images found'}")
print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")

# Visualize sample images
def plot_sample_images(images, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        idx = np.random.randint(0, len(images))
        axes[i].imshow(images[idx].squeeze(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
    plt.tight_layout()
    plt.show()

if len(images) > 0:
    plot_sample_images(images)

# Create data pipeline
BUFFER_SIZE = len(images)
BATCH_SIZE = 32

# Convert to tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print(f"Dataset created with batch size: {BATCH_SIZE}")

"""
## **4. GAN Architecture Definition**
"""

class BrainMRIGAN:
    def __init__(self, img_shape=(128, 128, 1), latent_dim=100):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
        
    def build_generator(self):
        """Build generator model"""
        model = keras.Sequential([
            # Start with dense layer
            layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(self.latent_dim,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # Reshape to start convolutional process
            layers.Reshape((8, 8, 256)),
            
            # Upsample to 16x16
            layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # Upsample to 32x32
            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # Upsample to 64x64
            layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # Upsample to 128x128
            layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
                                  activation='tanh')
        ])
        
        return model
    
    def build_discriminator(self):
        """Build discriminator model"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.img_shape),
            
            # First convolutional block
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            # Second convolutional block
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            # Third convolutional block
            layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            # Fourth convolutional block
            layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            # Flatten and output
            layers.Flatten(),
            layers.Dense(1)
        ])
        
        return model
    
    def build_gan(self):
        """Build combined GAN model"""
        # Freeze discriminator when training generator
        self.discriminator.trainable = False
        
        # GAN takes noise as input and outputs generated images
        gan_input = layers.Input(shape=(self.latent_dim,))
        generated_img = self.generator(gan_input)
        gan_output = self.discriminator(generated_img)
        
        gan = keras.Model(gan_input, gan_output)
        return gan
    
    def compile_models(self, g_optimizer, d_optimizer):
        """Compile generator, discriminator, and GAN"""
        self.discriminator.compile(optimizer=d_optimizer, 
                                  loss=keras.losses.BinaryCrossentropy(from_logits=True))
        self.gan.compile(optimizer=g_optimizer, 
                        loss=keras.losses.BinaryCrossentropy(from_logits=True))
    
    def summary(self):
        """Print model summaries"""
        print("\n=== Generator Summary ===")
        self.generator.summary()
        print("\n=== Discriminator Summary ===")
        self.discriminator.summary()

# Initialize GAN
print("Initializing GAN models...")
gan_model = BrainMRIGAN(img_shape=(128, 128, 1), latent_dim=100)
gan_model.summary()

"""
## **5. Training Configuration and Functions**
"""

# Compile models
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

gan_model.compile_models(generator_optimizer, discriminator_optimizer)

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
LATENT_DIM = 100

# Create directory for generated images
os.makedirs('generated_images', exist_ok=True)

def train_gan(gan, dataset, epochs, latent_dim, save_interval=10):
    """
    Train the GAN
    
    Args:
        gan: BrainMRIGAN instance
        dataset: Training dataset
        epochs: Number of epochs
        latent_dim: Latent dimension size
        save_interval: Interval for saving generated images
    """
    # Training history
    g_losses = []
    d_losses = []
    
    # Labels for real and fake images
    real_labels = tf.ones((BATCH_SIZE, 1))
    fake_labels = tf.zeros((BATCH_SIZE, 1))
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_g_loss = []
        epoch_d_loss = []
        
        for batch_idx, real_images in enumerate(dataset):
            batch_size = real_images.shape[0]
            
            # Train Discriminator
            # Generate fake images
            noise = tf.random.normal([batch_size, latent_dim])
            fake_images = gan.generator(noise, training=True)
            
            # Train on real images
            with tf.GradientTape() as tape:
                real_output = gan.discriminator(real_images, training=True)
                d_loss_real = keras.losses.binary_crossentropy(real_labels[:batch_size], 
                                                              real_output, 
                                                              from_logits=True)
            
            # Train on fake images
            with tf.GradientTape() as tape:
                fake_output = gan.discriminator(fake_images, training=True)
                d_loss_fake = keras.losses.binary_crossentropy(fake_labels[:batch_size], 
                                                              fake_output, 
                                                              from_logits=True)
            
            # Calculate total discriminator loss
            d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
            
            # Update discriminator
            grads = tape.gradient(d_loss, gan.discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(
                zip(grads, gan.discriminator.trainable_variables))
            
            # Train Generator
            noise = tf.random.normal([batch_size, latent_dim])
            with tf.GradientTape() as tape:
                fake_images = gan.generator(noise, training=True)
                fake_output = gan.discriminator(fake_images, training=True)
                g_loss = keras.losses.binary_crossentropy(real_labels[:batch_size], 
                                                          fake_output, 
                                                          from_logits=True)
            
            # Update generator
            grads = tape.gradient(g_loss, gan.generator.trainable_variables)
            generator_optimizer.apply_gradients(
                zip(grads, gan.generator.trainable_variables))
            
            epoch_g_loss.append(g_loss)
            epoch_d_loss.append(d_loss)
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")
        
        # Calculate average losses
        avg_g_loss = np.mean(epoch_g_loss)
        avg_d_loss = np.mean(epoch_d_loss)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"Epoch {epoch+1}: Avg D_loss={avg_d_loss:.4f}, Avg G_loss={avg_g_loss:.4f}")
        
        # Save generated images
        if (epoch + 1) % save_interval == 0:
            generate_and_save_images(gan.generator, epoch + 1, latent_dim)
    
    return g_losses, d_losses

def generate_and_save_images(generator, epoch, latent_dim, num_examples=16):
    """Generate and save sample images"""
    noise = tf.random.normal([num_examples, latent_dim])
    generated_images = generator(noise, training=False)
    
    # Denormalize images
    generated_images = (generated_images.numpy() + 1) / 2.0
    
    fig = plt.figure(figsize=(8, 8))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    
    plt.suptitle(f'Generated Brain MRI Images - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'generated_images/epoch_{epoch:03d}.png')
    plt.show()
    plt.close()

"""
## **6. Train the GAN**
"""

# Train the GAN
print("Starting GAN training...")
g_losses, d_losses = train_gan(gan_model, dataset, EPOCHS, LATENT_DIM, save_interval=5)

"""
## **7. Training Results Visualization**
"""

def plot_training_history(g_losses, d_losses):
    """Plot training losses"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(d_losses, label='Discriminator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(g_losses, d_losses)

"""
## **8. Generate Final Images**
"""

def generate_and_compare(generator, num_samples=16):
    """Generate final images and compare with real ones"""
    noise = tf.random.normal([num_samples, LATENT_DIM])
    generated_images = generator(noise, training=False)
    generated_images = (generated_images.numpy() + 1) / 2.0
    
    # Get random real images for comparison
    random_indices = np.random.randint(0, len(images), num_samples)
    real_images = (images[random_indices] + 1) / 2.0
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
    
    for i in range(num_samples):
        # Real images (top row)
        axes[0, i].imshow(real_images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == num_samples // 2:
            axes[0, i].set_title('Real MRI Images')
        
        # Generated images (bottom row)
        axes[1, i].imshow(generated_images[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == num_samples // 2:
            axes[1, i].set_title('Generated MRI Images')
    
    plt.tight_layout()
    plt.savefig('real_vs_generated.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate and compare final images
print("Generating and comparing final images...")
generate_and_compare(gan_model.generator)

"""
## **9. Save the Model**
"""

# Save the generator model
gan_model.generator.save('brain_mri_generator.h5')
print("Generator model saved as 'brain_mri_generator.h5'")

# Save the discriminator model
gan_model.discriminator.save('brain_mri_discriminator.h5')
print("Discriminator model saved as 'brain_mri_discriminator.h5'")

# Create a zip file with all generated images
!zip -r generated_images.zip generated_images/
print("Generated images archived as 'generated_images.zip'")


def load_and_use_generator(model_path, num_images=5):
    """Load a saved generator and create new images"""
    # Load the model
    loaded_generator = keras.models.load_model(model_path)
    
    # Generate images
    noise = tf.random.normal([num_images, LATENT_DIM])
    generated_images = loaded_generator(noise, training=False)
    generated_images = (generated_images.numpy() + 1) / 2.0
    
    # Display images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(generated_images[i, :, :, 0], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Generated {i+1}')
    
    plt.suptitle('Newly Generated Brain MRI Images')
    plt.tight_layout()
    plt.show()
    
    return generated_images

# Test the loaded generator
print("Testing loaded generator...")
test_images = load_and_use_generator('brain_mri_generator.h5', num_images=5)

"""
## **11. Image Quality Assessment**
"""

def assess_image_quality(generated_images, real_images):
    """
    Simple quality assessment of generated images
    """
    from skimage.metrics import structural_similarity as ssim
    
    # Ensure same number of images
    n = min(len(generated_images), len(real_images))
    generated = generated_images[:n]
    real = real_images[:n]
    
    # Calculate metrics
    ssim_scores = []
    for i in range(n):
        # Calculate SSIM
        score = ssim(real[i].squeeze(), generated[i].squeeze(), data_range=1.0)
        ssim_scores.append(score)
    
    print("Image Quality Assessment Results:")
    print(f"Average SSIM Score: {np.mean(ssim_scores):.4f}")
    print(f"SSIM Std Deviation: {np.std(ssim_scores):.4f}")
    
    # Plot histogram of SSIM scores
    plt.figure(figsize=(8, 5))
    plt.hist(ssim_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('SSIM Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of SSIM Scores')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return ssim_scores

# Assess quality
print("Assessing generated image quality...")
real_samples = (images[:16] + 1) / 2.0
quality_scores = assess_image_quality(test_images, real_samples)

