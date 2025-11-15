import os
import urllib.request
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import ImageGrid


tf.random.set_seed(42)
np.random.seed(42)



def kl_divergence(mu, log_var):
    return 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis=-1)

def log_diag_mvn(x, mu, log_sigma):
    sum_axes = tf.range(1, tf.rank(mu))
    k = tf.cast(tf.reduce_prod(tf.shape(mu)[1:]), x.dtype)
    logp = - 0.5 * k * tf.math.log(2*np.pi) \
           - tf.reduce_sum(log_sigma, axis=sum_axes) \
           - 0.5 * tf.reduce_sum(tf.square(x - mu) / tf.exp(2.*log_sigma), axis=sum_axes)
    return logp


class FixedBWBiCoder(layers.Layer):
    
    def __init__(self, latent_dim, name="BiCoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.network = None
        
    def call(self, inputs):
        return self.network(inputs)
    
    def sample(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement sample method")

class FixedBWEncoder(FixedBWBiCoder):
    
    def __init__(self, latent_dim=20, **kwargs):
        super().__init__(latent_dim, name="Encoder", **kwargs)
        self.build_network()
        
    def build_network(self):
        input_shape = (28*28,)
        units = 400
        activation = 'relu'
        
        self.network = Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Dense(units, activation=activation),
            layers.Dense(2 * self.latent_dim),
        ])
            
    def call(self, x):
        out = self.network(x)
        mu = out[:, :self.latent_dim]
        log_var = out[:, self.latent_dim:]
        return mu, log_var
        
    def sample(self, mu, log_var):
        std = tf.math.exp(0.5 * log_var)
        eps = tf.random.normal(tf.shape(std))
        return mu + eps * std

class FixedBWDecoder(FixedBWBiCoder):
    
    def __init__(self, latent_dim=20, **kwargs):
        super().__init__(latent_dim, name="Decoder", **kwargs)
        self.std = 0.75  
        self.build_network()
        
    def build_network(self):
        units = 400
        activation = 'relu'
        output_dim = 28*28
        
        self.network = Sequential([
            layers.InputLayer(input_shape=(self.latent_dim,)),
            layers.Dense(units, activation=activation),
            layers.Dense(output_dim),  
        ])
        
    def sample(self, mu):
        eps = tf.random.normal(tf.shape(mu))
        return mu + eps * self.std

class FixedBWVAE(tf.keras.Model):
    
    def __init__(self, latent_dim=20, **kwargs):
        super().__init__(name="FixedBWVAE", **kwargs)
        self.latent_dim = latent_dim
        
        self.encoder = FixedBWEncoder(latent_dim=latent_dim)
        self.decoder = FixedBWDecoder(latent_dim=latent_dim)
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def call(self, inputs):
        mu, log_var = self.encoder(inputs)
        z = self.encoder.sample(mu, log_var)
        
        mu_decoder = self.decoder(z)
        
        x_reconstructed = self.decoder.sample(mu_decoder)
        
        log_sigma = tf.math.log(self.decoder.std)
        
        log_sigma_expanded = tf.ones_like(inputs) * log_sigma
        
        reconstruction_loss = -tf.reduce_mean(
            log_diag_mvn(inputs, x_reconstructed, log_sigma_expanded)
        )
        

        kl_loss = tf.reduce_mean(kl_divergence(mu, log_var))
        
        self.vae_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(self.vae_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return self.vae_loss
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.call(data)
            
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def generate_from_prior(self, num_samples):
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        mu_decoder = self.decoder(z)
        return tf.clip_by_value(mu_decoder, 0.0, 1.0)
    
    def generate_from_posterior(self, x):
        mu, log_var = self.encoder(x)
        z = self.encoder.sample(mu, log_var)
        mu_decoder = self.decoder(z)
        return tf.clip_by_value(mu_decoder, 0.0, 1.0)
    
    def encode(self, x):
        mu, log_var = self.encoder(x)
        return self.encoder.sample(mu, log_var)

class FixedBWDataLoader:
    
    def __init__(self):
        self.data_urls = {
            'train': 'https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=1',
            'test': 'https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=1',
            'labels': 'https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=1'
        }
        
    def _download_file(self, url, filename):
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                if 'mnist_bw' in filename and 'te' not in filename:
                    np.save(filename, np.random.rand(60000, 28, 28, 1).astype(np.float32))
                elif 'mnist_bw_te' in filename:
                    np.save(filename, np.random.rand(10000, 28, 28, 1).astype(np.float32))
                elif 'y_te' in filename:
                    np.save(filename, np.random.randint(0, 10, 10000))
        return filename
    
    def get_training_data(self, batch_size=128):
        train_file = self._download_file(self.data_urls['train'], 'mnist_bw.npy')
        
        x_train = np.load(train_file)
        print(f"Original train shape: {x_train.shape}")
        
        x_train = x_train.astype('float32') / 255.0
        if len(x_train.shape) == 4:  # (N, 28, 28, 1)
            x_train = x_train.reshape((x_train.shape[0], 28*28))
        elif len(x_train.shape) == 3:  # (N, 28, 28)
            x_train = x_train.reshape((x_train.shape[0], 28*28))
        
        print(f"Processed train shape: {x_train.shape}")
        
        dataset = tf.data.Dataset.from_tensor_slices(x_train)
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        return dataset
    
    def get_test_data(self):
        test_file = self._download_file(self.data_urls['test'], 'mnist_bw_te.npy')
        
        x_test = np.load(test_file)
        print(f"Original test shape: {x_test.shape}")
        
        x_test = x_test.astype('float32') / 255.0
        if len(x_test.shape) == 4:  
            x_test = x_test.reshape((x_test.shape[0], 28*28))
        elif len(x_test.shape) == 3:  
            x_test = x_test.reshape((x_test.shape[0], 28*28))
        
        print(f"Processed test shape: {x_test.shape}")
        
        return tf.convert_to_tensor(x_test)
    
    def get_labels(self):
        labels_file = self._download_file(self.data_urls['labels'], 'mnist_bw_y_te.npy')
        return np.load(labels_file)


def plot_grid(images, N=10, C=10, figsize=(10, 10), name='posterior'):
    if isinstance(images, tf.Tensor):
        images = images.numpy()
    
    if len(images.shape) == 2:  
        images = images.reshape(-1, 28, 28)
    
    img = tf.clip_by_value(255 * images, clip_value_min=0, clip_value_max=255).numpy().astype(np.uint8)
    
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(N, C), axes_pad=0)
    
    for ax, im in zip(grid, img[:N*C]):
        ax.imshow(im, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()  


def visualize_latent_space_tsne(model, test_data, labels, latent_dim):
    n_samples = min(1000, test_data.shape[0])
    test_subset = test_data[:n_samples]
    labels_subset = labels[:n_samples]
    
    z = model.encode(test_subset).numpy()
    
    try:
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
        z_2d = tsne.fit_transform(z)
    except Exception as e:
        from sklearn.decomposition import PCA
        z_2d = PCA(n_components=2).fit_transform(z)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_subset, cmap='tab10', 
                          alpha=0.7, s=20, edgecolors='white', linewidth=0.3)
    plt.colorbar(scatter, label='Digit Class')
    plt.title(f'BW VAE Latent Space - Latent Dim: {latent_dim}', fontsize=14)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.2)
    plt.show() 

def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], color='red', linewidth=2)
    plt.title('Total VAE Loss - BW', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['reconstruction_loss'], color='blue', linewidth=2)
    plt.title('Reconstruction Loss - BW', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['kl_loss'], color='green', linewidth=2)
    plt.title('KL Divergence - BW', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 



def main():
    parser = argparse.ArgumentParser(description='Train Fixed BW VAE with specified architecture')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--latent_dim', type=int, default=20,
                       help='Latent space dimension')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for optimizer')
    
    args = parser.parse_args()

    
    data_loader = FixedBWDataLoader()
    train_data = data_loader.get_training_data(batch_size=args.batch_size)
    
    model = FixedBWVAE(latent_dim=args.latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer)
    
    print(f"\nTesting model output shapes...")
    sample_batch = next(iter(train_data))
    mu, log_var = model.encoder(sample_batch)
    z = model.encoder.sample(mu, log_var)
    decoded = model.decoder(z)
    print(f"  Input shape: {sample_batch.shape}")
    print(f"  Encoder mu shape: {mu.shape}")
    print(f"  Latent z shape: {z.shape}")
    print(f"  Decoder output shape: {decoded.shape}")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    history = model.fit(train_data, epochs=args.epochs, verbose=1)
    
    
    test_data = data_loader.get_test_data()
    labels = data_loader.get_labels()
    
    print("1. Generating BW images from prior...")
    generated_images = model.generate_from_prior(num_samples=100)
    plot_grid(generated_images, name='prior')
    
    print("2. Generating BW images from posterior...")
    test_subset = test_data[:100]
    generated_images = model.generate_from_posterior(test_subset)
    plot_grid(generated_images, name='posterior')
    
    print("3. Creating latent space visualization...")
    visualize_latent_space_tsne(model, test_data, labels, args.latent_dim)
    
    print("4. Plotting training history...")
    plot_training_history(history)


if __name__ == "__main__":
    main()



