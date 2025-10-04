"""
CS 599: Lab 1
Problem 1
Student: Smirthya Somaskantha Iyer
"""

import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Set seed 
tf.random.set_seed(12345)
np.random.seed(12345)

print("TensorFlow version:", tf.__version__)

# Data
NUM_EXAMPLES = 10000

def make_data(noise_std=0.5):
    X = tf.random.normal([NUM_EXAMPLES])
    noise = tf.random.normal([NUM_EXAMPLES], stddev=noise_std)
    y = X * 3.0 + 2.0 + noise
    return X, y

# Loss functions
def mse_loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

def l1_loss(y, y_pred):
    return tf.reduce_mean(tf.abs(y - y_pred))

def huber_loss(y, y_pred, delta=1.0):
    error = y - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear)

def hybrid_loss(y, y_pred):
    return 0.5 * l1_loss(y, y_pred) + 0.5 * mse_loss(y, y_pred)

# Model
def predict(W, b, x):
    return W * x + b


# ========== EXPERIMENT 1: Compare Loss Functions ==========
print("\n=== EXPERIMENT 1: Loss Functions ===")
X, y = make_data()

losses = {'MSE': mse_loss, 'L1': l1_loss, 'Huber': huber_loss, 'Hybrid': hybrid_loss}
results = {}

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)

for name, loss_fn in losses.items():
    W = tf.Variable(0.0)
    b = tf.Variable(0.0)
    history = []
    
    for step in range(1000):
        with tf.GradientTape() as tape:
            y_pred = predict(W, b, X)
            loss = loss_fn(y, y_pred)
        grads = tape.gradient(loss, [W, b])
        W.assign_sub(0.01 * grads[0])
        b.assign_sub(0.01 * grads[1])
        history.append(loss.numpy())
    
    print(f"{name}: W={W.numpy():.4f}, b={b.numpy():.4f}")
    results[name] = (W.numpy(), b.numpy())
    plt.plot(history, label=name)

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Plot predictions
plt.subplot(1, 2, 2)
X_plot = tf.sort(X[:100])
plt.scatter(X[:100], y[:100], alpha=0.3, s=10)
for name, (w, b_val) in results.items():
    plt.plot(X_plot, w * X_plot + b_val, label=name, linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()


# ========== EXPERIMENT 2: Learning Rates ==========
print("\n=== EXPERIMENT 2: Learning Rates ===")
X, y = make_data()

lrs = [0.001, 0.01, 0.05, 0.1]
plt.figure(figsize=(10, 4))

for lr in lrs:
    W = tf.Variable(0.0)
    b = tf.Variable(0.0)
    history = []
    
    for step in range(1000):
        with tf.GradientTape() as tape:
            y_pred = predict(W, b, X)
            loss = mse_loss(y, y_pred)
        grads = tape.gradient(loss, [W, b])
        W.assign_sub(lr * grads[0])
        b.assign_sub(lr * grads[1])
        history.append(loss.numpy())
        
        # check for divergence
        if loss.numpy() > 100:
            print(f"LR={lr}: DIVERGED")
            break
    else:
        print(f"LR={lr}: W={W.numpy():.4f}, b={b.numpy():.4f}")
        plt.plot(history, label=f'LR={lr}')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# ========== EXPERIMENT 3: Noise Levels ==========
print("\n=== EXPERIMENT 3: Noise Levels ===")

noise_levels = [0.1, 0.5, 1.0, 2.0]
plt.figure(figsize=(12, 3))

for i, noise_std in enumerate(noise_levels):
    X, y = make_data(noise_std=noise_std)
    W = tf.Variable(0.0)
    b = tf.Variable(0.0)
    history = []
    
    for step in range(1000):
        with tf.GradientTape() as tape:
            y_pred = predict(W, b, X)
            loss = mse_loss(y, y_pred)
        grads = tape.gradient(loss, [W, b])
        W.assign_sub(0.01 * grads[0])
        b.assign_sub(0.01 * grads[1])
        history.append(loss.numpy())
    
    print(f"Noise σ={noise_std}: W={W.numpy():.4f}, b={b.numpy():.4f}")
    
    plt.subplot(1, 4, i+1)
    plt.plot(history)
    plt.title(f'σ={noise_std}')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid()

plt.tight_layout()
plt.show()


# ========== EXPERIMENT 4: Different Initializations ==========
print("\n=== EXPERIMENT 4: Initializations ===")
X, y = make_data()

inits = [(0.0, 0.0), (5.0, 5.0), (-3.0, -2.0), (10.0, -5.0)]
plt.figure(figsize=(10, 4))

for W_init, b_init in inits:
    W = tf.Variable(W_init)
    b = tf.Variable(b_init)
    history = []
    
    for step in range(1000):
        with tf.GradientTape() as tape:
            y_pred = predict(W, b, X)
            loss = mse_loss(y, y_pred)
        grads = tape.gradient(loss, [W, b])
        W.assign_sub(0.01 * grads[0])
        b.assign_sub(0.01 * grads[1])
        history.append(loss.numpy())
    
    print(f"Init W={W_init}, b={b_init} → Final: W={W.numpy():.4f}, b={b.numpy():.4f}")
    plt.plot(history, label=f'({W_init},{b_init})')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# ========== EXPERIMENT 5: Weight Noise ==========
print("\n=== EXPERIMENT 5: Weight Noise ===")
X, y = make_data()

plt.figure(figsize=(10, 4))

# No noise
W = tf.Variable(0.0)
b = tf.Variable(0.0)
history = []
for step in range(1000):
    with tf.GradientTape() as tape:
        y_pred = predict(W, b, X)
        loss = mse_loss(y, y_pred)
    grads = tape.gradient(loss, [W, b])
    W.assign_sub(0.01 * grads[0])
    b.assign_sub(0.01 * grads[1])
    history.append(loss.numpy())
print(f"No noise: W={W.numpy():.4f}, b={b.numpy():.4f}")
plt.plot(history, label='No noise')

# With weight noise
W = tf.Variable(0.0)
b = tf.Variable(0.0)
history = []
for step in range(1000):
    if step > 0:  # add noise to weights
        W.assign_add(tf.random.normal([], stddev=0.01))
        b.assign_add(tf.random.normal([], stddev=0.01))
    
    with tf.GradientTape() as tape:
        y_pred = predict(W, b, X)
        loss = mse_loss(y, y_pred)
    grads = tape.gradient(loss, [W, b])
    W.assign_sub(0.01 * grads[0])
    b.assign_sub(0.01 * grads[1])
    history.append(loss.numpy())
print(f"Weight noise: W={W.numpy():.4f}, b={b.numpy():.4f}")
plt.plot(history, label='Weight noise')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# ========== EXPERIMENT 6: Training Duration ==========
print("\n=== EXPERIMENT 6: Training Duration ===")
X, y = make_data()

for num_steps in [100, 500, 1000, 2000]:
    W = tf.Variable(0.0)
    b = tf.Variable(0.0)
    
    for step in range(num_steps):
        with tf.GradientTape() as tape:
            y_pred = predict(W, b, X)
            loss = mse_loss(y, y_pred)
        grads = tape.gradient(loss, [W, b])
        W.assign_sub(0.01 * grads[0])
        b.assign_sub(0.01 * grads[1])
    
    print(f"{num_steps} steps: W={W.numpy():.4f}, b={b.numpy():.4f}")


# ========== EXPERIMENT 7: GPU vs CPU Timing ==========
print("\n=== EXPERIMENT 7: GPU vs CPU Timing ===")
X, y = make_data()

# CPU timing
print("Testing CPU...")
with tf.device('/CPU:0'):
    W = tf.Variable(0.0)
    b = tf.Variable(0.0)
    
    start = time.time()
    for step in range(1000):
        with tf.GradientTape() as tape:
            y_pred = predict(W, b, X)
            loss = mse_loss(y, y_pred)
        grads = tape.gradient(loss, [W, b])
        W.assign_sub(0.01 * grads[0])
        b.assign_sub(0.01 * grads[1])
    cpu_time = time.time() - start

print(f"CPU - Total: {cpu_time:.4f} seconds, Per epoch: {cpu_time/1000*1000:.2f} ms")

# GPU timing
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus}")
    print("Testing GPU...")
    with tf.device('/GPU:0'):
        W = tf.Variable(0.0)
        b = tf.Variable(0.0)
        
        start = time.time()
        for step in range(1000):
            with tf.GradientTape() as tape:
                y_pred = predict(W, b, X)
                loss = mse_loss(y, y_pred)
            grads = tape.gradient(loss, [W, b])
            W.assign_sub(0.01 * grads[0])
            b.assign_sub(0.01 * grads[1])
        gpu_time = time.time() - start
    
    print(f"GPU - Total: {gpu_time:.4f} seconds, Per epoch: {gpu_time/1000*1000:.2f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
else:
    print("No GPU detected")


# ========== EXPERIMENT 8: Reproducibility ==========
print("\n=== EXPERIMENT 8: Reproducibility ===")

for run in range(3):
    tf.random.set_seed(12345)
    np.random.seed(12345)
    
    X, y = make_data()
    W = tf.Variable(0.0)
    b = tf.Variable(0.0)
    
    for step in range(1000):
        with tf.GradientTape() as tape:
            y_pred = predict(W, b, X)
            loss = mse_loss(y, y_pred)
        grads = tape.gradient(loss, [W, b])
        W.assign_sub(0.01 * grads[0])
        b.assign_sub(0.01 * grads[1])
    
    print(f"Run {run+1}: W={W.numpy():.6f}, b={b.numpy():.6f}")

print("\n=== ALL DONE ===")
