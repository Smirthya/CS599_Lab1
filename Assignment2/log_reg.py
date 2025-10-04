"""
CS 599: Lab 1
Problem 2
Student: Smirthya Somaskantha Iyer
"""

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("TensorFlow version:", tf.__version__)

# Set seed
SEED = 12345
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Load Fashion-MNIST
print("\n=== Loading Fashion-MNIST Dataset ===")

# Load using TensorFlow datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Flatten images from 28x28 to 784
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

# Convert labels to one-hot encoding
def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

y_train_onehot = one_hot(y_train)
y_test_onehot = one_hot(y_test)

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")
print(f"Image shape (flattened): {x_train.shape[1]}")

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Configuration
input_size = 784  # 28x28
num_classes = 10
learning_rate = 0.01
batch_size = 128
n_epochs = 20

# Create Train/Val Split
print("\n=== Creating Train/Val Split ===")

# Use 50k for training, 10k for validation
n_train = 50000
n_val = 10000

x_train_split = x_train[:n_train]
y_train_split = y_train_onehot[:n_train]

x_val = x_train[n_train:]
y_val = y_train_onehot[n_train:]

print(f"Training: {n_train}, Validation: {n_val}, Test: {len(x_test)}")

# Model Parameters
W = tf.Variable(tf.random.normal([input_size, num_classes], stddev=0.01))
b = tf.Variable(tf.zeros([num_classes]))

# Model Functions

def logits_fn(X):
    """Compute logits: X @ W + b"""
    return tf.matmul(X, W) + b

def softmax_cross_entropy(logits, labels):
    """Cross-entropy loss"""
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )

def accuracy_fn(logits, labels):
    """Calculate accuracy"""
    predictions = tf.argmax(logits, axis=1)
    true_labels = tf.argmax(labels, axis=1)
    correct = tf.equal(predictions, true_labels)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

# Training Function

def train_epoch(X, y, batch_size, optimizer):
    """Train for one epoch"""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    total_loss = 0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = tf.constant(X[batch_idx])
        y_batch = tf.constant(y[batch_idx])
        
        with tf.GradientTape() as tape:
            logits = logits_fn(X_batch)
            loss = softmax_cross_entropy(logits, y_batch)
        
        gradients = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(gradients, [W, b]))
        
        total_loss += loss.numpy()
        n_batches += 1
    
    return total_loss / n_batches

# EXPERIMENT 1: Basic Training
print("\n=== EXPERIMENT 1: Training with SGD ===")

optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

train_acc_history = []
val_acc_history = []
train_loss_history = []

start_time = time.time()

for epoch in range(n_epochs):
    # Train
    avg_loss = train_epoch(x_train_split, y_train_split, batch_size, optimizer)
    train_loss_history.append(avg_loss)
    
    # Evaluate on train and val
    train_logits = logits_fn(tf.constant(x_train_split))
    train_acc = accuracy_fn(train_logits, tf.constant(y_train_split)).numpy()
    train_acc_history.append(train_acc)
    
    val_logits = logits_fn(tf.constant(x_val))
    val_acc = accuracy_fn(val_logits, tf.constant(y_val)).numpy()
    val_acc_history.append(val_acc)
    
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

training_time = time.time() - start_time
print(f"\nTraining time: {training_time:.2f} seconds")

# Test accuracy
test_logits = logits_fn(tf.constant(x_test))
test_acc = accuracy_fn(test_logits, tf.constant(y_test_onehot)).numpy()
print(f"Final Test Accuracy: {test_acc:.4f}")

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Train')
plt.plot(val_acc_history, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train/Val Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Save SGD results
sgd_results = {
    'train_acc': train_acc_history[-1],
    'val_acc': val_acc_history[-1],
    'test_acc': test_acc,
    'time': training_time
}

# EXPERIMENT 2: Different Optimizers
print("\n=== EXPERIMENT 2: Comparing Optimizers ===")

optimizers_to_test = {
    'SGD': tf.optimizers.SGD(learning_rate=0.01),
    'Adam': tf.optimizers.Adam(learning_rate=0.001),
    'RMSprop': tf.optimizers.RMSprop(learning_rate=0.001),
    'Adagrad': tf.optimizers.Adagrad(learning_rate=0.01)
}

optimizer_results = {}

plt.figure(figsize=(12, 4))

for opt_name, optimizer in optimizers_to_test.items():
    print(f"\nTraining with {opt_name}...")
    
    # Reset weights
    W.assign(tf.random.normal([input_size, num_classes], stddev=0.01))
    b.assign(tf.zeros([num_classes]))
    
    val_acc_hist = []
    
    for epoch in range(n_epochs):
        avg_loss = train_epoch(x_train_split, y_train_split, batch_size, optimizer)
        
        val_logits = logits_fn(tf.constant(x_val))
        val_acc = accuracy_fn(val_logits, tf.constant(y_val)).numpy()
        val_acc_hist.append(val_acc)
    
    # Final test accuracy
    test_logits = logits_fn(tf.constant(x_test))
    test_acc = accuracy_fn(test_logits, tf.constant(y_test_onehot)).numpy()
    
    optimizer_results[opt_name] = {'val_acc': val_acc_hist[-1], 'test_acc': test_acc}
    print(f"{opt_name}: Val Acc={val_acc_hist[-1]:.4f}, Test Acc={test_acc:.4f}")
    
    plt.plot(val_acc_hist, label=opt_name)

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Optimizer Comparison')
plt.legend()
plt.grid()
plt.show()

# EXPERIMENT 3: Different Train/Val Splits
print("\n=== EXPERIMENT 3: Different Train/Val Splits ===")

splits = [(55000, 5000), (50000, 10000), (45000, 15000)]

for n_tr, n_v in splits:
    W.assign(tf.random.normal([input_size, num_classes], stddev=0.01))
    b.assign(tf.zeros([num_classes]))
    
    x_tr = x_train[:n_tr]
    y_tr = y_train_onehot[:n_tr]
    x_v = x_train[n_tr:n_tr+n_v]
    y_v = y_train_onehot[n_tr:n_tr+n_v]
    
    optimizer = tf.optimizers.SGD(learning_rate=0.01)
    
    for epoch in range(15):  # fewer epochs for speed
        train_epoch(x_tr, y_tr, batch_size, optimizer)
    
    val_logits = logits_fn(tf.constant(x_v))
    val_acc = accuracy_fn(val_logits, tf.constant(y_v)).numpy()
    
    print(f"Split {n_tr}/{n_v}: Val Acc={val_acc:.4f}")

# EXPERIMENT 4: Different Batch Sizes
print("\n=== EXPERIMENT 4: Effect of Batch Size ===")

batch_sizes = [32, 64, 128, 256, 512]

for bs in batch_sizes:
    W.assign(tf.random.normal([input_size, num_classes], stddev=0.01))
    b.assign(tf.zeros([num_classes]))
    
    optimizer = tf.optimizers.SGD(learning_rate=0.01)
    
    start = time.time()
    for epoch in range(10):
        train_epoch(x_train_split, y_train_split, bs, optimizer)
    epoch_time = (time.time() - start) / 10
    
    val_logits = logits_fn(tf.constant(x_val))
    val_acc = accuracy_fn(val_logits, tf.constant(y_val)).numpy()
    
    print(f"Batch size {bs}: Val Acc={val_acc:.4f}, Time/epoch={epoch_time:.2f}s")

# EXPERIMENT 5: Overfitting Check
print("\n=== EXPERIMENT 5: Overfitting Analysis ===")

# Train for more epochs to check overfitting
W.assign(tf.random.normal([input_size, num_classes], stddev=0.01))
b.assign(tf.zeros([num_classes]))

optimizer = tf.optimizers.SGD(learning_rate=0.01)

train_accs = []
val_accs = []

for epoch in range(50):
    train_epoch(x_train_split, y_train_split, batch_size, optimizer)
    
    train_logits = logits_fn(tf.constant(x_train_split))
    train_acc = accuracy_fn(train_logits, tf.constant(y_train_split)).numpy()
    train_accs.append(train_acc)
    
    val_logits = logits_fn(tf.constant(x_val))
    val_acc = accuracy_fn(val_logits, tf.constant(y_val)).numpy()
    val_accs.append(val_acc)

plt.figure(figsize=(8, 4))
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Overfitting Check (50 epochs)')
plt.legend()
plt.grid()
plt.show()

gap = train_accs[-1] - val_accs[-1]
print(f"Final train acc: {train_accs[-1]:.4f}")
print(f"Final val acc: {val_accs[-1]:.4f}")
print(f"Train-Val gap: {gap:.4f}")

if gap > 0.05:
    print("Model is overfitting (train-val gap > 0.05)")
else:
    print("No significant overfitting")

# EXPERIMENT 6: Compare with sklearn
print("\n=== EXPERIMENT 6: Comparison with Random Forest and SVM ===")

# Use subset for speed (sklearn is slow on 50k samples)
n_subset = 10000
x_subset = x_train[:n_subset]
y_subset = y_train[:n_subset]

print("\nTraining Random Forest...")
rf_start = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
rf.fit(x_subset, y_subset)
rf_time = time.time() - rf_start
rf_acc = accuracy_score(y_test, rf.predict(x_test))
print(f"Random Forest: Test Acc={rf_acc:.4f}, Time={rf_time:.2f}s")

print("\nTraining SVM...")
svm_start = time.time()
svm = SVC(kernel='rbf', random_state=SEED)
svm.fit(x_subset, y_subset)
svm_time = time.time() - svm_start
svm_acc = accuracy_score(y_test, svm.predict(x_test))
print(f"SVM: Test Acc={svm_acc:.4f}, Time={svm_time:.2f}s")

print(f"\nLogistic Regression (our model): Test Acc={sgd_results['test_acc']:.4f}, Time={sgd_results['time']:.2f}s")

# EXPERIMENT 7: Plot Sample Images
print("\n=== Visualizing Sample Images ===")

def plot_images(images, labels, predictions=None, n=9):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < n:
            img = images[i].reshape(28, 28)
            ax.imshow(img, cmap='binary')
            
            true_label = class_names[labels[i]]
            if predictions is not None:
                pred_label = class_names[predictions[i]]
                xlabel = f'True: {true_label}\nPred: {pred_label}'
                color = 'green' if labels[i] == predictions[i] else 'red'
            else:
                xlabel = f'True: {true_label}'
                color = 'black'
            
            ax.set_xlabel(xlabel, color=color)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.show()

# Get predictions for test set
test_logits = logits_fn(tf.constant(x_test[:9]))
test_preds = tf.argmax(test_logits, axis=1).numpy()

plot_images(x_test[:9], y_test[:9], test_preds)

# EXPERIMENT 8: Plot Weights
print("\n=== Visualizing Learned Weights ===")

def plot_weights(W_matrix):
    w = W_matrix.numpy()
    w_min = w.min()
    w_max = w.max()
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < 10:
            img = w[:, i].reshape(28, 28)
            ax.imshow(img, vmin=w_min, vmax=w_max, cmap='seismic')
            ax.set_xlabel(f'Class {i}: {class_names[i]}')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.show()

plot_weights(W)

# EXPERIMENT 9: CPU/GPU Timing
print("\n=== EXPERIMENT 9: CPU Timing ===")

# Reset and retrain on CPU
W.assign(tf.random.normal([input_size, num_classes], stddev=0.01))
b.assign(tf.zeros([num_classes]))

with tf.device('/CPU:0'):
    optimizer = tf.optimizers.SGD(learning_rate=0.01)
    start = time.time()
    
    for epoch in range(10):
        train_epoch(x_train_split, y_train_split, batch_size, optimizer)
    
    cpu_time = time.time() - start

print(f"CPU time (10 epochs): {cpu_time:.4f}s")
print(f"Per epoch: {cpu_time/10:.4f}s")

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\nGPU detected: {gpus}")
    # GPU timing would go here
else:
    print("\nNo GPU detected")

print("\n=== ALL EXPERIMENTS COMPLETED ===")
