# -*- coding: utf-8 -*-
"""CS599_Lab4_RNN_Clean.ipynb"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image
import tarfile

tf.random.set_seed(5695)
np.random.seed(5695)

# ============================================================================
# Download and Load notMNIST Dataset
# ============================================================================

def load_notmnist():
    """Load notMNIST dataset"""
    
    # Download dataset
    if not os.path.exists('notMNIST_small.tar.gz'):
        print('Downloading notMNIST dataset...')
        !wget -q http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
    
    # Extract
    if not os.path.exists('notMNIST_small'):
        print('Extracting dataset...')
        with tarfile.open('notMNIST_small.tar.gz', 'r:gz') as tar:
            tar.extractall()
    
    # Load from extracted files
    print('Loading notMNIST data...')
    data_root = 'notMNIST_small'
    
    folders = [os.path.join(data_root, d) for d in sorted(os.listdir(data_root))
               if os.path.isdir(os.path.join(data_root, d))]
    
    datasets = []
    labels = []
    
    for label_idx, folder in enumerate(folders):
        print(f'Loading {folder}...')
        image_files = os.listdir(folder)
        dataset = np.ndarray(shape=(len(image_files), 28, 28), dtype=np.float32)
        image_index = 0
        
        for image in image_files:
            image_file = os.path.join(folder, image)
            try:
                image_data = np.array(Image.open(image_file))
                if image_data.shape == (28, 28):
                    dataset[image_index, :, :] = image_data
                    image_index += 1
            except:
                pass
        
        dataset = dataset[0:image_index, :, :]
        datasets.append(dataset)
        labels.extend([label_idx] * image_index)
    
    # Combine all data
    all_data = np.concatenate(datasets)
    all_labels = np.array(labels)
    
    # Normalize
    all_data = (all_data.astype(float) - 128.0) / 128.0
    
    # Shuffle
    indices = np.random.permutation(len(all_data))
    all_data = all_data[indices]
    all_labels = all_labels[indices]
    
    # Split into train/test
    split_idx = int(0.8 * len(all_data))
    X_train = all_data[:split_idx]
    y_train = all_labels[:split_idx]
    X_test = all_data[split_idx:]
    y_test = all_labels[split_idx:]
    
    print(f'\nTraining set: {X_train.shape}, {y_train.shape}')
    print(f'Test set: {X_test.shape}, {y_test.shape}')
    
    return X_train, X_test, y_train, y_test

# ============================================================================
# RNN Cell Implementations
# ============================================================================

class BasicGRU(tf.keras.Model):
    def __init__(self, units, return_sequence=False, return_states=False, **kwargs):
        super(BasicGRU, self).__init__(**kwargs)
        self.units = units
        self.return_sequence = return_sequence
        self.return_states = return_states
        
        self.kernel_z = tf.keras.layers.Dense(units, use_bias=True)
        self.recurrent_kernel_z = tf.keras.layers.Dense(units, use_bias=False)
        
        self.kernel_r = tf.keras.layers.Dense(units, use_bias=True)
        self.recurrent_kernel_r = tf.keras.layers.Dense(units, use_bias=False)
        
        self.kernel_h = tf.keras.layers.Dense(units, use_bias=True)
        self.recurrent_kernel_h = tf.keras.layers.Dense(units, use_bias=False)
    
    def call(self, inputs, training=None, mask=None, initial_states=None):
        batch_size = tf.shape(inputs)[0]
        
        if initial_states is None:
            h_state = tf.zeros((batch_size, self.units))
        else:
            h_state = initial_states[0]
        
        h_list = []
        
        for t in range(inputs.shape[1]):
            ip = inputs[:, t, :]
            
            zt = tf.keras.activations.sigmoid(
                self.kernel_z(ip) + self.recurrent_kernel_z(h_state))
            
            rt = tf.keras.activations.sigmoid(
                self.kernel_r(ip) + self.recurrent_kernel_r(h_state))
            
            s_tilde = tf.nn.tanh(
                self.kernel_h(ip) + self.recurrent_kernel_h(rt * h_state))
            
            h_state = (1 - zt) * h_state + zt * s_tilde
            h_list.append(h_state)
        
        hidden_outputs = tf.stack(h_list, axis=1)
        
        if self.return_sequence:
            return hidden_outputs
        else:
            return hidden_outputs[:, -1, :]


class BasicMGU(tf.keras.Model):
    def __init__(self, units, return_sequence=False, return_states=False, **kwargs):
        super(BasicMGU, self).__init__(**kwargs)
        self.units = units
        self.return_sequence = return_sequence
        self.return_states = return_states
        
        self.kernel_f = tf.keras.layers.Dense(units, use_bias=True)
        self.recurrent_kernel_f = tf.keras.layers.Dense(units, use_bias=False)
        
        self.kernel_h = tf.keras.layers.Dense(units, use_bias=True)
        self.recurrent_kernel_h = tf.keras.layers.Dense(units, use_bias=False)
    
    def call(self, inputs, training=None, mask=None, initial_states=None):
        batch_size = tf.shape(inputs)[0]
        
        if initial_states is None:
            h_state = tf.zeros((batch_size, self.units))
        else:
            h_state = initial_states[0]
        
        h_list = []
        
        for t in range(inputs.shape[1]):
            ip = inputs[:, t, :]
            
            ft = tf.keras.activations.sigmoid(
                self.kernel_f(ip) + self.recurrent_kernel_f(h_state))
            
            s_tilde = tf.nn.tanh(
                self.kernel_h(ip) + self.recurrent_kernel_h(ft * h_state))
            
            h_state = (1 - ft) * h_state + ft * s_tilde
            h_list.append(h_state)
        
        hidden_outputs = tf.stack(h_list, axis=1)
        
        if self.return_sequence:
            return hidden_outputs
        else:
            return hidden_outputs[:, -1, :]

# ============================================================================
# Model Building and Training
# ============================================================================

def build_model(rnn_cell, units, num_classes=10):
    model = tf.keras.Sequential([
        rnn_cell(units=units, return_sequence=False),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=15, batch_size=128):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    
    return history, train_acc, test_acc

# ============================================================================
# Run Experiments
# ============================================================================

def run_experiments():
    print("="*80)
    print("Loading notMNIST Dataset...")
    print("="*80)
    X_train, X_test, y_train, y_test = load_notmnist()
    
    units = 128
    epochs = 15
    num_trials = 3
    
    results = {
        'GRU': {'histories': [], 'train_accs': [], 'test_accs': []},
        'MGU': {'histories': [], 'train_accs': [], 'test_accs': []}
    }
    
    for trial in range(num_trials):
        print("\n" + "="*80)
        print(f"TRIAL {trial + 1}/{num_trials}")
        print("="*80)
        
        tf.random.set_seed(5695 + trial)
        np.random.seed(5695 + trial)
        
        # Train GRU
        print(f"\n--- Training GRU (Trial {trial + 1}) ---")
        gru_model = build_model(BasicGRU, units=units)
        gru_history, gru_train_acc, gru_test_acc = train_model(
            gru_model, X_train, y_train, X_test, y_test, epochs=epochs)
        
        results['GRU']['histories'].append(gru_history)
        results['GRU']['train_accs'].append(gru_train_acc)
        results['GRU']['test_accs'].append(gru_test_acc)
        
        print(f"\nGRU Trial {trial + 1} - Train Acc: {gru_train_acc:.4f}, Test Acc: {gru_test_acc:.4f}")
        
        # Train MGU
        print(f"\n--- Training MGU (Trial {trial + 1}) ---")
        mgu_model = build_model(BasicMGU, units=units)
        mgu_history, mgu_train_acc, mgu_test_acc = train_model(
            mgu_model, X_train, y_train, X_test, y_test, epochs=epochs)
        
        results['MGU']['histories'].append(mgu_history)
        results['MGU']['train_accs'].append(mgu_train_acc)
        results['MGU']['test_accs'].append(mgu_test_acc)
        
        print(f"\nMGU Trial {trial + 1} - Train Acc: {mgu_train_acc:.4f}, Test Acc: {mgu_test_acc:.4f}")
    
    return results

# ============================================================================
# Plot Results
# ============================================================================

def plot_results(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('GRU vs MGU: Training and Test Accuracy over 3 Trials', fontsize=16)
    
    models = ['GRU', 'MGU']
    colors = {'GRU': 'blue', 'MGU': 'orange'}
    
    for model_idx, model_name in enumerate(models):
        for trial in range(3):
            ax = axes[model_idx, trial]
            history = results[model_name]['histories'][trial]
            
            ax.plot(history.history['accuracy'], label='Train', 
                   color=colors[model_name], linewidth=2)
            ax.plot(history.history['val_accuracy'], label='Test', 
                   color=colors[model_name], linestyle='--', linewidth=2)
            
            ax.set_title(f'{model_name} - Trial {trial + 1}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rnn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    for model_name in models:
        print(f"\n{model_name} Results:")
        print("-" * 40)
        train_accs = results[model_name]['train_accs']
        test_accs = results[model_name]['test_accs']
        
        for i in range(3):
            train_err = (1 - train_accs[i]) * 100
            test_err = (1 - test_accs[i]) * 100
            print(f"Trial {i+1}: Train Error = {train_err:.2f}%, Test Error = {test_err:.2f}%")
        
        avg_train_err = (1 - np.mean(train_accs)) * 100
        avg_test_err = (1 - np.mean(test_accs)) * 100
        print(f"\nAverage: Train Error = {avg_train_err:.2f}%, Test Error = {avg_test_err:.2f}%")

# ============================================================================
# Run Everything
# ============================================================================

print("CS 599 Deep Learning - Lab 4: RNN Cells Comparison")
print("Seed: 5695")
print("="*80)

results = run_experiments()
plot_results(results)

print("\n" + "="*80)
print("Experiment Complete! Results saved to 'rnn_comparison.png'")
print("="*80)
