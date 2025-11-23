"""
CS 599: Deep Learning - Assignment: Implementing Various Normalization Techniques
Author: Smirthya Somaskantha Iyer
"""

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow import keras
import gc  # For garbage collection

# Set random seeds for reproducibility
seed = 1234
tf.random.set_seed(seed)
np.random.seed(seed)

# Hyperparameters
batch_size = 64
hidden_size = 100
learning_rate = 0.01
output_size = 10
num_epochs = 3  # Reduced to avoid memory issues

# Choose dataset: 'fashion_mnist' or 'cifar10'
DATASET = 'fashion_mnist'

# Memory optimization: set to True if running out of memory
USE_SUBSET = False  # Set to True to use only 30,000 training samples
SUBSET_SIZE = 30000

# Load dataset
if DATASET == 'fashion_mnist':
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    input_shape = (28, 28, 1)
else:  # cifar10
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()
    input_shape = (32, 32, 3)

# Convert labels to one-hot
train_labels_onehot = tf.one_hot(train_labels, depth=10)
test_labels_onehot = tf.one_hot(test_labels, depth=10)

# Apply subset if needed for memory optimization
if USE_SUBSET:
    train_images = train_images[:SUBSET_SIZE]
    train_labels_onehot = train_labels_onehot[:SUBSET_SIZE]
    print(f"Using subset of {SUBSET_SIZE} training samples for memory optimization")

print(f"Dataset: {DATASET}")
print(f"Training samples: {len(train_images)}")
print(f"Test samples: {len(test_images)}")
print(f"Input shape: {input_shape}")


class CNN(object):
    def __init__(self, hidden_size, output_size, norm_type='none', use_tf_norm=False):
        """
        CNN Model with different normalization options

        Args:
            hidden_size: Number of hidden units in fully connected layer
            output_size: Number of output classes
            norm_type: Type of normalization ('none', 'batch', 'layer', 'weight')
            use_tf_norm: Whether to use TensorFlow's built-in normalization
        """
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.norm_type = norm_type
        self.use_tf_norm = use_tf_norm

        # Convolutional layer parameters
        if DATASET == 'fashion_mnist':
            filter_h, filter_w, filter_c, filter_n = 5, 5, 1, 30
        else:  # cifar10
            filter_h, filter_w, filter_c, filter_n = 5, 5, 3, 32

        self.W1 = tf.Variable(tf.random.normal([filter_h, filter_w, filter_c, filter_n], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([filter_n]), dtype=tf.float32)

        # Calculate size after conv and pooling
        if DATASET == 'fashion_mnist':
            conv_output_size = 14 * 14 * filter_n
        else:
            conv_output_size = 16 * 16 * filter_n

        # Fully connected layer 1
        self.W2 = tf.Variable(tf.random.normal([conv_output_size, hidden_size], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([hidden_size]), dtype=tf.float32)

        # For Weight Normalization on W2
        if norm_type == 'weight':
            self.v2 = tf.Variable(tf.random.normal([conv_output_size, hidden_size], stddev=0.1))
            self.g2 = tf.Variable(tf.ones([hidden_size]))

        # Fully connected layer 2
        self.W3 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([output_size]), dtype=tf.float32)

        # For Weight Normalization on W3
        if norm_type == 'weight':
            self.v3 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1))
            self.g3 = tf.Variable(tf.ones([output_size]))

        # Batch Normalization parameters
        if norm_type == 'batch' and not use_tf_norm:
            self.gamma_bn = tf.Variable(tf.ones([filter_n]))
            self.beta_bn = tf.Variable(tf.zeros([filter_n]))
            self.gamma_bn2 = tf.Variable(tf.ones([hidden_size]))
            self.beta_bn2 = tf.Variable(tf.zeros([hidden_size]))

        # Layer Normalization parameters
        if norm_type == 'layer' and not use_tf_norm:
            self.gamma_ln = tf.Variable(tf.ones([filter_n]))
            self.beta_ln = tf.Variable(tf.zeros([filter_n]))
            self.gamma_ln2 = tf.Variable(tf.ones([hidden_size]))
            self.beta_ln2 = tf.Variable(tf.zeros([hidden_size]))

        # TensorFlow normalization layers
        if use_tf_norm:
            if norm_type == 'batch':
                self.bn1 = keras.layers.BatchNormalization()
                self.bn2 = keras.layers.BatchNormalization()
            elif norm_type == 'layer':
                self.ln1 = keras.layers.LayerNormalization()
                self.ln2 = keras.layers.LayerNormalization()

        # Collect all trainable variables
        self.variables = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

        if norm_type == 'weight':
            self.variables.extend([self.v2, self.g2, self.v3, self.g3])

        if norm_type == 'batch' and not use_tf_norm:
            self.variables.extend([self.gamma_bn, self.beta_bn, self.gamma_bn2, self.beta_bn2])

        if norm_type == 'layer' and not use_tf_norm:
            self.variables.extend([self.gamma_ln, self.beta_ln, self.gamma_ln2, self.beta_ln2])

        if use_tf_norm:
            if norm_type == 'batch':
                self.variables.extend(self.bn1.trainable_variables)
                self.variables.extend(self.bn2.trainable_variables)
            elif norm_type == 'layer':
                self.variables.extend(self.ln1.trainable_variables)
                self.variables.extend(self.ln2.trainable_variables)

    def batch_normalization(self, x, gamma, beta, epsilon=1e-5):
        """
        Custom Batch Normalization implementation
        Normalizes across the batch dimension
        """
        # Calculate mean and variance across batch dimension
        batch_mean = tf.reduce_mean(x, axis=0, keepdims=True)
        batch_var = tf.reduce_mean(tf.square(x - batch_mean), axis=0, keepdims=True)

        # Normalize
        x_normalized = (x - batch_mean) / tf.sqrt(batch_var + epsilon)

        # Scale and shift
        return gamma * x_normalized + beta

    def layer_normalization(self, x, gamma, beta, epsilon=1e-5):
        """
        Custom Layer Normalization implementation
        Normalizes across the feature dimension
        """
        # Calculate mean and variance across feature dimensions
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)

        # Normalize
        x_normalized = (x - mean) / tf.sqrt(variance + epsilon)

        # Scale and shift
        return gamma * x_normalized + beta

    def weight_normalization(self, v, g):
        """
        Custom Weight Normalization implementation
        w = (g / ||v||) * v
        """
        # Calculate L2 norm of v along the input dimension
        v_norm = tf.norm(v, axis=0, keepdims=True)

        # Normalize and scale
        w = (g / v_norm) * v

        return w

    def forward(self, X, training=True):
        """
        Forward pass through the network
        """
        # Convolutional layer using tf.nn.conv2d
        conv_output = tf.nn.conv2d(X, self.W1, strides=[1, 1, 1, 1], padding='SAME')
        conv_output = tf.nn.bias_add(conv_output, self.b1)

        # Apply normalization after convolution
        if self.norm_type == 'batch':
            if self.use_tf_norm:
                conv_output = self.bn1(conv_output, training=training)
            else:
                # Custom batch norm - reshape for batch norm
                shape = conv_output.shape
                conv_output_reshaped = tf.reshape(conv_output, [-1, shape[-1]])
                conv_output_normalized = self.batch_normalization(
                    conv_output_reshaped, self.gamma_bn, self.beta_bn
                )
                conv_output = tf.reshape(conv_output_normalized, shape)

        elif self.norm_type == 'layer':
            if self.use_tf_norm:
                conv_output = self.ln1(conv_output, training=training)
            else:
                conv_output = self.layer_normalization(conv_output, self.gamma_ln, self.beta_ln)

        # ReLU activation
        conv_activation = tf.nn.relu(conv_output)

        # Max pooling using tf.nn.max_pool2d
        pool_output = tf.nn.max_pool2d(conv_activation, ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1], padding='VALID')

        # Flatten
        flattened = tf.reshape(pool_output, [pool_output.shape[0], -1])

        # Fully connected layer 1
        if self.norm_type == 'weight':
            W2_normalized = self.weight_normalization(self.v2, self.g2)
            fc1_output = tf.matmul(flattened, W2_normalized) + self.b2
        else:
            fc1_output = tf.matmul(flattened, self.W2) + self.b2

        # Apply normalization after first FC layer
        if self.norm_type == 'batch':
            if self.use_tf_norm:
                fc1_output = self.bn2(fc1_output, training=training)
            else:
                fc1_output = self.batch_normalization(fc1_output, self.gamma_bn2, self.beta_bn2)

        elif self.norm_type == 'layer':
            if self.use_tf_norm:
                fc1_output = self.ln2(fc1_output, training=training)
            else:
                fc1_output = self.layer_normalization(fc1_output, self.gamma_ln2, self.beta_ln2)

        # ReLU activation
        fc1_activation = tf.nn.relu(fc1_output)

        # Fully connected layer 2 (output)
        if self.norm_type == 'weight':
            W3_normalized = self.weight_normalization(self.v3, self.g3)
            output = tf.matmul(fc1_activation, W3_normalized) + self.b3
        else:
            output = tf.matmul(fc1_activation, self.W3) + self.b3

        return output

    def loss(self, y_pred, y_true):
        """
        Compute cross-entropy loss
        """
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        )

    def backward(self, X_train, y_train, optimizer):
        """
        Backward pass using GradientTape
        """
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train, training=True)
            current_loss = self.loss(predicted, y_train)

        gradients = tape.gradient(current_loss, self.variables)
        optimizer.apply_gradients(zip(gradients, self.variables))

        return current_loss


def accuracy_function(y_pred, y_true):
    """
    Calculate accuracy
    """
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def compute_accuracy_in_batches(model, images, labels, batch_size=256):
    """
    Compute accuracy in batches to avoid memory issues
    """
    num_samples = len(images)
    num_batches = (num_samples + batch_size - 1) // batch_size

    total_correct = 0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        batch_preds = model.forward(batch_images, training=False)
        correct = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(batch_preds, 1), tf.argmax(batch_labels, 1)), tf.float32)
        )
        total_correct += correct.numpy()

    return total_correct / num_samples


def train_model(model, train_images, train_labels, test_images, test_labels,
                num_epochs, batch_size, learning_rate, model_name="Model"):
    """
    Train the model and return training history
    """
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epoch_time': []
    }

    num_train = len(train_images)

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Create dataset
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_ds = train_ds.shuffle(buffer_size=10000).batch(batch_size)

        epoch_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_ds:
            loss = model.backward(batch_x, batch_y, optimizer)
            epoch_loss += loss.numpy()
            num_batches += 1

        # Calculate training accuracy (in batches to avoid OOM)
        train_acc = compute_accuracy_in_batches(model, train_images, train_labels, batch_size=512)

        # Calculate test accuracy (in batches to avoid OOM)
        test_acc = compute_accuracy_in_batches(model, test_images, test_labels, batch_size=512)

        epoch_time = time.time() - epoch_start

        # Store history
        history['train_loss'].append(epoch_loss / num_batches)
        history['train_acc'].append(train_acc * 100)
        history['test_acc'].append(test_acc * 100)
        history['epoch_time'].append(epoch_time)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Loss: {epoch_loss / num_batches:.4f} - "
              f"Train Acc: {train_acc * 100:.2f}% - "
              f"Test Acc: {test_acc * 100:.2f}% - "
              f"Time: {epoch_time:.2f}s")

    return history


def compare_normalizations(custom_model, tf_model, test_images, test_labels):
    """
    Compare custom normalization implementation with TensorFlow's built-in
    """
    print(f"\n{'='*60}")
    print("Comparing Custom vs TensorFlow Normalization")
    print(f"{'='*60}")

    # Use only 500 samples for comparison to avoid OOM
    sample_size = 500
    test_sample = test_images[:sample_size]
    labels_sample = test_labels[:sample_size]

    # Get predictions from both models in smaller batches
    batch_size = 100
    custom_preds_list = []
    tf_preds_list = []

    for i in range(0, sample_size, batch_size):
        batch = test_sample[i:i+batch_size]
        custom_preds_list.append(custom_model.forward(batch, training=False))
        tf_preds_list.append(tf_model.forward(batch, training=False))

    custom_preds = tf.concat(custom_preds_list, axis=0)
    tf_preds = tf.concat(tf_preds_list, axis=0)

    # Calculate difference
    diff = tf.abs(custom_preds - tf_preds)
    mean_diff = tf.reduce_mean(diff).numpy()
    max_diff = tf.reduce_max(diff).numpy()

    print(f"Mean absolute difference: {mean_diff:.8f}")
    print(f"Max absolute difference: {max_diff:.8f}")

    # Calculate accuracies
    custom_acc = accuracy_function(custom_preds, labels_sample)
    tf_acc = accuracy_function(tf_preds, labels_sample)

    print(f"Custom implementation accuracy: {custom_acc.numpy() * 100:.2f}%")
    print(f"TensorFlow implementation accuracy: {tf_acc.numpy() * 100:.2f}%")

    if mean_diff < 1e-4:
        print("Implementations match within floating point tolerance!")
    else:
        print("Significant difference detected - check backward pass")

    return mean_diff, max_diff


def plot_results(histories, names):
    """
    Plot training results
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot training loss
    for history, name in zip(histories, names):
        axes[0].plot(history['train_loss'], marker='o', label=name)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot training accuracy
    for history, name in zip(histories, names):
        axes[1].plot(history['train_acc'], marker='o', label=name)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Plot test accuracy
    for history, name in zip(histories, names):
        axes[2].plot(history['test_acc'], marker='o', label=name)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Test Accuracy')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nPlot saved as 'normalization_comparison.png'")


# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("CNN NORMALIZATION COMPARISON EXPERIMENT")
    print("="*60)

    # Train models with different normalization techniques
    models_to_train = [
        ('No Normalization', 'none', False),
        ('Batch Normalization (Custom)', 'batch', False),
        ('Layer Normalization (Custom)', 'layer', False),
        ('Weight Normalization', 'weight', False),
    ]

    histories = []
    model_names = []
    trained_models = {}

    for name, norm_type, use_tf_norm in models_to_train:
        model = CNN(hidden_size, output_size, norm_type=norm_type, use_tf_norm=use_tf_norm)
        history = train_model(
            model, train_images, train_labels_onehot,
            test_images, test_labels_onehot,
            num_epochs, batch_size, learning_rate,
            model_name=name
        )
        histories.append(history)
        model_names.append(name)
        trained_models[name] = model

        # Clear memory after each model
        gc.collect()
        tf.keras.backend.clear_session()

    # Plot results
    plot_results(histories, model_names)

    # Compare custom implementations with TensorFlow's built-in
    print("\n" + "="*60)
    print("VALIDATION: Comparing Custom vs TensorFlow Built-in")
    print("="*60)

    # Train models with TensorFlow's built-in normalization (using smaller subset)
    for norm_type, norm_name in [('batch', 'Batch Normalization'), ('layer', 'Layer Normalization')]:
        print(f"\n{norm_name}:")
        custom_model = trained_models[f'{norm_name} (Custom)']

        # Create and train TensorFlow model on smaller subset
        tf_model = CNN(hidden_size, output_size, norm_type=norm_type, use_tf_norm=True)
        _ = train_model(
            tf_model, train_images[:3000], train_labels_onehot[:3000],
            test_images[:1000], test_labels_onehot[:1000],
            2, batch_size, learning_rate,
            model_name=f'{norm_name} (TensorFlow - Validation)'
        )

        # Compare
        compare_normalizations(custom_model, tf_model, test_images, test_labels_onehot)

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for i, (name, history) in enumerate(zip(model_names, histories)):
        final_train_acc = history['train_acc'][-1]
        final_test_acc = history['test_acc'][-1]
        avg_epoch_time = np.mean(history['epoch_time'])

        print(f"\n{name}:")
        print(f"  Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"  Final Test Accuracy: {final_test_acc:.2f}%")
        print(f"  Average Epoch Time: {avg_epoch_time:.2f}s")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
