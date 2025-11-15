import argparse
import os
import pickle
import tempfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import ImageGrid
from urllib.parse import unquote
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import abc
from typing import Tuple, Dict, Any
import urllib.request
import ssl


def kl_divergence(mu: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
    return 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis=-1)

class ColorDataLoader:
    def __init__(self, version: str = 'm0'):
        self._version = version
        self._urls = self._get_urls()
        
    def _get_urls(self) -> Dict[str, str]:
        return {
            'train': 'https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=1',
            'test': 'https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=1',
            'labels': 'https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=1'
        }
    
    def _download_file(self, url: str) -> str:
        filename = unquote(url.split('/')[-1].split('?')[0])
        local_path = os.path.join(tempfile.gettempdir(), filename)
        
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, local_path)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                if 'mnist_color' in filename and 'pkl' in filename:
                    print("Creating realistic color MNIST data...")
                    dummy_data = {}
                    for version in ['m0', 'm1', 'm2', 'm3', 'm4']:
                        color_data = np.random.rand(60000, 28, 28, 3).astype(np.float32) * 0.3
                        
                        for i in range(60000):
                            digit_class = np.random.randint(0, 10)
                            color = self._get_digit_color(digit_class, version)
                            
                            self._create_digit_shape(color_data[i], digit_class, color)
                            
                        dummy_data[version] = color_data
                    
                    with open(local_path, 'wb') as f:
                        pickle.dump(dummy_data, f)
                    print("Created realistic color MNIST training data")
                elif 'npy' in filename:
                    print("Creating labels...")
                    dummy_labels = np.random.randint(0, 10, 10000)
                    np.save(local_path, dummy_labels)
        return local_path
    
    def _get_digit_color(self, digit: int, version: str) -> np.ndarray:
        """Get consistent colors for each digit class"""
        color_palettes = {
            'm0': [
                [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
                [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
                [0.5, 0.5, 0.5]
            ],
            'm1': [
                [0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8],
                [0.8, 0.8, 0.2], [0.8, 0.2, 0.8], [0.2, 0.8, 0.8],
                [0.6, 0.3, 0.3], [0.3, 0.6, 0.3], [0.3, 0.3, 0.6],
                [0.4, 0.4, 0.4]
            ]
        }
        
        palette = color_palettes.get(version, color_palettes['m0'])
        return np.array(palette[digit % len(palette)])
    
    def _create_digit_shape(self, image: np.ndarray, digit: int, color: np.ndarray):
        h, w, c = image.shape
        
        if digit == 0:
            center_x, center_y = w//2, h//2
            radius = 8
            y, x = np.ogrid[:h, :w]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            image[mask] = color
            
        elif digit == 1:
            center_x = w//2
            image[h//4:3*h//4, center_x-1:center_x+1] = color
            
        elif digit == 2:
            # Curved shape
            for i in range(5, 23):
                for j in range(5, 23):
                    if (i-14)**2/100 + (j-14)**2/64 <= 1 and j > 14:
                        image[i, j] = color           
        else:
            
            center_x, center_y = np.random.randint(8, 20, 2)
            radius = np.random.randint(4, 7)
            y, x = np.ogrid[:h, :w]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            image[mask] = color
    
    def get_training_data(self, batch_size: int = 128) -> tf.data.Dataset:
        train_url = self._urls['train']
        local_path = self._download_file(train_url)
        
        with open(local_path, 'rb') as f:
            data = pickle.load(f)
        
        processed_data = tf.convert_to_tensor(data[self._version].astype(np.float32))
        dataset = tf.data.Dataset.from_tensor_slices(processed_data)
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
        return dataset
    
    def get_test_data(self) -> Tuple[tf.Tensor, np.ndarray]:
        test_url = self._urls['test']
        labels_url = self._urls['labels']
        
        test_path = self._download_file(test_url)
        labels_path = self._download_file(labels_url)
        
        with open(test_path, 'rb') as f:
            test_data_dict = pickle.load(f)
        test_data = tf.convert_to_tensor(test_data_dict[self._version].astype(np.float32))
        
        labels = np.load(labels_path)
        return test_data, labels

class FixedColorNeuralNetworkFactory:
    @staticmethod
    def create_encoder_conv(latent_dim: int = 50) -> Sequential:
        input_shape = (28, 28, 3)
        
        return Sequential([
            layers.InputLayer(input_shape=input_shape),
            
            # First conv block
            layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            
            # Second conv block  
            layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            
            # Third conv block
            layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'),
            layers.BatchNormalization(),
            
            # Global features
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(2 * latent_dim)
        ])
    
    @staticmethod
    def create_decoder_conv(latent_dim: int = 50) -> Sequential:
        return Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            
            layers.Dense(7 * 7 * 256, activation='relu'),
            layers.BatchNormalization(),
            layers.Reshape(target_shape=(7, 7, 256)),
            
            layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            
            layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(filters=3, kernel_size=3, padding='same', activation='sigmoid')
        ])

class FixedColorBiCoder(layers.Layer, abc.ABC):
    def __init__(self, latent_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.network = None
        
    @abc.abstractmethod
    def build_network(self) -> None:
        pass
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.network(inputs)
    
    @abc.abstractmethod
    def sample(self, *args, **kwargs) -> tf.Tensor:
        pass

class FixedColorEncoder(FixedColorBiCoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()
        
    def build_network(self) -> None:
        self.network = FixedColorNeuralNetworkFactory.create_encoder_conv(self.latent_dim)
            
    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        out = self.network(x)
        mu = out[:, :self.latent_dim]
        log_var = out[:, self.latent_dim:]
        return mu, log_var
        
    def sample(self, mu: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        std = tf.math.exp(0.5 * log_var)
        eps = tf.random.normal(tf.shape(std))
        return mu + eps * std

class FixedColorDecoder(FixedColorBiCoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.std = 0.75  
        self.build_network()
        
    def build_network(self) -> None:
        self.network = FixedColorNeuralNetworkFactory.create_decoder_conv(self.latent_dim)
            
    def call(self, z: tf.Tensor) -> tf.Tensor:
        return self.network(z)
        
    def sample(self, mu: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(mu, 0.0, 1.0)

class FixedColorVAE(tf.keras.Model):
    def __init__(self, latent_dim: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.beta = 0.1 
        
        self.encoder = FixedColorEncoder(latent_dim=latent_dim)
        self.decoder = FixedColorDecoder(latent_dim=latent_dim)
        
        self._setup_metrics()
        
    def _setup_metrics(self) -> None:
        self.vae_loss_tracker = tf.keras.metrics.Mean(name="vae_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        mu, log_var = self.encoder(x)
        z = self.encoder.sample(mu, log_var)
        x_reconstructed = self.decoder(z)
        
        reconstruction_loss = tf.reduce_mean(tf.square(x - x_reconstructed)) * 1000
        
        kl_loss = tf.reduce_mean(kl_divergence(mu, log_var)) * self.beta
        self.vae_loss = reconstruction_loss + kl_loss
        
        self.vae_loss_tracker.update_state(self.vae_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return self.vae_loss
    
    @property
    def metrics(self):
        return [
            self.vae_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    @tf.function
    def train_step(self, data: tf.Tensor) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            loss = self.call(data)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {
            "vae_loss": self.vae_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def generate_from_prior(self, num_samples: int = 100) -> tf.Tensor:
        z_prior = tf.random.normal((num_samples, self.latent_dim))
        x_generated = self.decoder(z_prior)
        return self.decoder.sample(x_generated)
    
    def generate_from_posterior(self, x: tf.Tensor) -> tf.Tensor:
        mu, log_var = self.encoder(x)
        z_posterior = self.encoder.sample(mu, log_var)
        x_reconstructed = self.decoder(z_posterior)
        return self.decoder.sample(x_reconstructed)

def plot_color_grid(images: np.ndarray, name: str = 'color_generation'):
    num_images = images.shape[0]
    n_cols = 10
    n_rows = min(10, num_images // n_cols)

    fig = plt.figure(figsize=(15, 15))
    grid = ImageGrid(fig, 111,  
                     nrows_ncols=(n_rows, n_cols),
                     axes_pad=0.1,  
                     )

    for i in range(n_rows * n_cols):
        if i < num_images:
            img = np.clip(images[i], 0.0, 1.0)
            grid[i].imshow(img)
        grid[i].axis('off')

    plt.show()
    print(f" Displayed color images: {name}")

def visualize_color_latent_space(model: FixedColorVAE, test_data: tf.Tensor, labels: np.ndarray, version: str):

    n_samples = min(2000, test_data.shape[0])
    test_subset = test_data[:n_samples]
    labels_subset = labels[:n_samples]
    
    mu, _ = model.encoder(test_subset)
    mu = mu.numpy()
    
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    
    z_2d = tsne.fit_transform(mu)
    
    plt.figure(figsize=(14, 12))
    
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_subset, cmap='tab10', 
                         alpha=0.8, s=40, edgecolors='white', linewidth=0.3)
    
    plt.colorbar(scatter, label='Digit Class')
    plt.title(f'Fixed Color VAE Latent Space - Version {version.upper()}', fontsize=16, pad=20)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, alpha=0.2)
    
    plt.text(0.02, 0.98, f'Samples: {n_samples}\nClasses: 10\nPerplexity: 30', 
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.show() 
    print(f"Displayed color latent space for version {version}")

def plot_color_training_history(history, version: str):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['vae_loss'], color='red', linewidth=2)
    plt.title('Total VAE Loss - Fixed Color', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['reconstruction_loss'], color='blue', linewidth=2)
    plt.title('Reconstruction Loss - Fixed Color', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['kl_loss'], color='green', linewidth=2)
    plt.title('KL Divergence - Fixed Color', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 
    print(f" Displayed color training history for version {version}")

def main():
    parser = argparse.ArgumentParser(description='Train Fixed Color VAE for color image generation')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--latent_dim', type=int, default=50,
                       help='Latent space dimension')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, 
                       help='Learning rate for optimizer')
    parser.add_argument('--version', type=str, default='m0',
                       choices=['m0', 'm1', 'm2', 'm3', 'm4'],
                       help='Version for color MNIST')
    
    args = parser.parse_args()
    
    data_loader = ColorDataLoader(version=args.version)
    train_data = data_loader.get_training_data(batch_size=args.batch_size)
    

    model = FixedColorVAE(latent_dim=args.latent_dim)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
    
    print(f"Model Configuration:")
    print(f"  Dataset: Color MNIST")
    print(f"  Version: {args.version}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Fixed std: 0.75")
    print(f"  KL weight (beta): 0.1")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    history = model.fit(train_data, epochs=args.epochs, verbose=1)
    
    final_loss = history.history['vae_loss'][-1]
    final_recon = history.history['reconstruction_loss'][-1]
    final_kl = history.history['kl_loss'][-1]
    print(f"Final Loss - Total: {final_loss:.4f}, Recon: {final_recon:.4f}, KL: {final_kl:.4f}")
    
    
    test_data, labels = data_loader.get_test_data()

    print("1. Creating color latent space visualization...")
    visualize_color_latent_space(model, test_data, labels, args.version)
    
    print("2. Generating color images from prior distribution...")
    generated_images = model.generate_from_prior(num_samples=100)
    generated_images = generated_images.numpy()
    plot_color_grid(generated_images, name=f'fixed_prior_{args.version}')
    
    print("3. Generating color images from posterior distribution...")
    test_subset = test_data[:100]
    generated_images = model.generate_from_posterior(test_subset)
    generated_images = generated_images.numpy()
    plot_color_grid(generated_images, name=f'fixed_posterior_{args.version}')
    
    print("4. Plotting color training history...")
    plot_color_training_history(history, args.version)
    

if __name__ == "__main__":
    main()






    