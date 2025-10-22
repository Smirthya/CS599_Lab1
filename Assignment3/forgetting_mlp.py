# -*- coding: utf-8 -*-
"""
Author: Smirthya Somaskantha Iyer
Analyzing Forgetting in neural networks
"""

import numpy as np
import os
import sys
import random
import tensorflow as tf

# basic config
SEED = 5695 # my NAU ID
num_tasks_to_run = 10
num_epochs_first_task = 50
num_epochs_other_tasks = 20
minibatch_size = 32
learning_rate = 0.001

# model variants
HIDDEN_UNITS = 256
DEPTH = 3
DROPOUT_RATE = 0.5
OPT_NAME = "adam"

# loss/regularization variants:
# LOSS_NAME: "nll"
LOSS_NAME = "nll"
L1 = 0.0
L2 = 0.0
if LOSS_NAME.lower() == "l1":      L1, L2 = 1e-5, 0.0
elif LOSS_NAME.lower() == "l2":    L1, L2 = 0.0, 1e-4
elif LOSS_NAME.lower() == "l1+l2": L1, L2 = 1e-5, 1e-4

# seeds
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# data: MNIST
# Using tf.keras.datasets; will create train/val split
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

VAL_FRAC = 0.1
n_val = int(len(x_train) * VAL_FRAC)
x_val, y_val = x_train[:n_val], y_train[:n_val]
x_train, y_train = x_train[n_val:], y_train[n_val:]

# flatten utility
def permute_images(x, perm):
    n = x.shape[0]
    x_flat = x.reshape(n, 784)
    x_perm = x_flat[:, perm]
    return x_perm.reshape(n, 28, 28)

# task permutations
# keep permuted datasets for each task (train/val/test)

# create list of random permutations for each task
task_permutation = []
for t in range(num_tasks_to_run):
    task_permutation.append(np.random.RandomState(SEED + t).permutation(784))

tasks_data = []
for t in range(num_tasks_to_run):
    p = task_permutation[t]
    Xtr_t = permute_images(x_train, p)
    Xva_t = permute_images(x_val, p)
    Xte_t = permute_images(x_test, p)
    tasks_data.append({
        "train": (Xtr_t, y_train),
        "val":   (Xva_t, y_val),
        "test":  (Xte_t, y_test),
    })

# model builder
def build_mlp(depth=DEPTH, units=HIDDEN_UNITS, dropout=DROPOUT_RATE, l1=L1, l2=L2):
    reg = tf.keras.regularizers.L1L2(l1=l1, l2=l2) if (l1>0 or l2>0) else None
    layers = [tf.keras.layers.Input(shape=(28, 28)),
              tf.keras.layers.Flatten()]
    for _ in range(depth):
        layers.append(tf.keras.layers.Dense(units, activation="relu",
                                            kernel_initializer="he_normal",
                                            kernel_regularizer=reg,
                                            bias_regularizer=None))
        if dropout and dropout > 0:
            layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(10, activation="softmax"))
    model = tf.keras.Sequential(layers)
    return model

def get_optimizer(name, lr):
    name = name.lower()
    if name == "sgd":     return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0)
    if name == "rmsprop": return tf.keras.optimizers.RMSprop(learning_rate=lr)
    return tf.keras.optimizers.Adam(learning_rate=lr)  # default

def get_loss(loss_name):
    # NLL for classification is just (sparse) cross-entropy here
    return tf.keras.losses.SparseCategoricalCrossentropy()

# simple datasets
def make_ds(x, y, batch, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle: ds = ds.shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# training across tasks
model = build_mlp(depth=DEPTH, units=HIDDEN_UNITS, dropout=DROPOUT_RATE, l1=L1, l2=L2)
opt = get_optimizer(OPT_NAME, learning_rate)
loss_fn = get_loss(LOSS_NAME)
model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

T = num_tasks_to_run
R = np.zeros((T, T), dtype=np.float32)  # R[t, i] = test acc on task i after finishing training on task t
best_diag = np.zeros(T, dtype=np.float32)  # R[i,i] when learned

for t in range(T):
    # datasets for current task
    Xtr, Ytr = tasks_data[t]["train"]
    Xva, Yva = tasks_data[t]["val"]
    Xte, Yte = tasks_data[t]["test"]

    train_ds = make_ds(Xtr, Ytr, minibatch_size, shuffle=True)
    val_ds   = make_ds(Xva, Yva, minibatch_size, shuffle=False)

    # epochs schedule
    epochs = num_epochs_first_task if t == 0 else num_epochs_other_tasks

    # train current task
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=0
    )

    # after finishing task t, evaluate on all seen tasks 0..t
    for i in range(t + 1):
        Xte_i, Yte_i = tasks_data[i]["test"]
        te_ds = make_ds(Xte_i, Yte_i, minibatch_size, shuffle=False)
        _, acc = model.evaluate(te_ds, verbose=0)
        R[t, i] = acc

    # store diagonal at learn-time
    best_diag[t] = R[t, t]
    print(f"Finished Task {t+1}/{T} | diag acc (R[{t},{t}]): {R[t,t]:.4f}")

# metrics: ACC, BWT
# ACC = average accuracy on all tasks after finishing last task (row T-1)
ACC = R[T-1, :].mean()

# BWT = 1/(T-1) * sum_{i=1..T-1} (R[T-1, i] - R[i, i])
BWT = np.mean(R[T-1, :-1] - best_diag[:-1])

print("\n=== Config ===")
print(f"SEED={SEED}, DEPTH={DEPTH}, UNITS={HIDDEN_UNITS}, DROPOUT={DROPOUT_RATE}, OPT={OPT_NAME}, LOSS={LOSS_NAME}, L1={L1}, L2={L2}")
print("Schedule: 50 epochs for Task 1, 20 epochs for each subsequent task")

print("\n=== Results ===")
print("R (rows: after task t, cols: tested task i):")
np.set_printoptions(precision=4, suppress=True)
print(R)

print(f"\nACC (mean last-row acc): {ACC:.4f}")
print(f"BWT: {BWT:.4f}")

# helper: run a single variant without touching your baseline variables above
def run_variant(depth, dropout, opt_name, loss_name, l1, l2):
    # build fresh model
    model_v = build_mlp(depth=depth, units=HIDDEN_UNITS, dropout=dropout, l1=l1, l2=l2)
    opt_v = get_optimizer(opt_name, learning_rate)
    loss_fn_v = get_loss(loss_name)
    model_v.compile(optimizer=opt_v, loss=loss_fn_v, metrics=["accuracy"])

    T = num_tasks_to_run
    Rv = np.zeros((T, T), dtype=np.float32)
    diag_when_learned_v = np.zeros(T, dtype=np.float32)

    for t in range(T):
        # datasets for current task (reuse precomputed tasks_data)
        Xtr, Ytr = tasks_data[t]["train"]
        Xva, Yva = tasks_data[t]["val"]

        train_ds = make_ds(Xtr, Ytr, minibatch_size, shuffle=True)
        val_ds   = make_ds(Xva, Yva, minibatch_size, shuffle=False)

        epochs = num_epochs_first_task if t == 0 else num_epochs_other_tasks

        model_v.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)

        # evaluate on all seen tasks so far
        for i in range(t + 1):
            Xte_i, Yte_i = tasks_data[i]["test"]
            te_ds = make_ds(Xte_i, Yte_i, minibatch_size, shuffle=False)
            _, acc = model_v.evaluate(te_ds, verbose=0)
            Rv[t, i] = acc

        diag_when_learned_v[t] = Rv[t, t]
        print(f"[{opt_name}/{loss_name}/drop={dropout}/depth={depth}] Finished Task {t+1}/{T} | diag acc: {Rv[t,t]:.4f}")

    ACC_v = Rv[T-1, :].mean()
    BWT_v = np.mean(Rv[T-1, :-1] - diag_when_learned_v[:-1])
    return ACC_v, BWT_v, Rv

results = []

# Baseline from your run above (reuse ACC, BWT, R you already computed)
results.append(("Task A (Baseline)", DEPTH, DROPOUT_RATE, OPT_NAME, LOSS_NAME, L1, L2, float(ACC), float(BWT), R))

# Loss ablation (keep depth=3, dropout=0.5, optimizer=adam)
print("\n=== Loss Ablation ===")
ACC_l1, BWT_l1, R_l1 = run_variant(DEPTH, DROPOUT_RATE, "adam", "l1", 1e-5, 0.0)
ACC_l2, BWT_l2, R_l2 = run_variant(DEPTH, DROPOUT_RATE, "adam", "l2", 0.0, 1e-4)
ACC_l12, BWT_l12, R_l12 = run_variant(DEPTH, DROPOUT_RATE, "adam", "l1+l2", 1e-5, 1e-4)
results.append(("Loss ablation L1", DEPTH, DROPOUT_RATE, "adam", "l1", 1e-5, 0.0, ACC_l1, BWT_l1, R_l1))
results.append(("Loss ablation L2", DEPTH, DROPOUT_RATE, "adam", "l2", 0.0, 1e-4, ACC_l2, BWT_l2, R_l2))
results.append(("Loss ablation L1 & L2", DEPTH, DROPOUT_RATE, "adam", "l1+l2", 1e-5, 1e-4, ACC_l12, BWT_l12, R_l12))

# Optimizer ablation (keep depth=3, dropout=0.5, loss=nll)
print("\n=== Optimizer Ablation ===")
ACC_sgd, BWT_sgd, R_sgd = run_variant(DEPTH, DROPOUT_RATE, "sgd", "nll", 0.0, 0.0)
ACC_rms, BWT_rms, R_rms = run_variant(DEPTH, DROPOUT_RATE, "rmsprop", "nll", 0.0, 0.0)
results.append(("SGD", DEPTH, DROPOUT_RATE, "sgd", "nll", 0.0, 0.0, ACC_sgd, BWT_sgd, R_sgd))
results.append(("RMSprop", DEPTH, DROPOUT_RATE, "rmsprop", "nll", 0.0, 0.0, ACC_rms, BWT_rms, R_rms))

# Dropout ablation (keep depth=3, opt=adam, loss=nll)
print("\n=== Dropout Ablation ===")
ACC_drop0, BWT_drop0, R_drop0 = run_variant(DEPTH, 0.0, "adam", "nll", 0.0, 0.0)
results.append(("Dropout ablation", DEPTH, 0.0, "adam", "nll", 0.0, 0.0, ACC_drop0, BWT_drop0, R_drop0))

# Print a compact table
print("\n=== Summary ===")
print("Configuration | Depth | Dropout | Optimizer | Loss | ACC | BWT")
for tag, d, drop, optn, lossn, l1v, l2v, accv, bwtv, _ in results:
    print(f"{tag:11s} | {d:5d} | {drop:7.2f} | {optn:9s} | {lossn:6s} | {accv:.4f} | {bwtv:.4f}")

# Save the required plot: last row of baseline R

try:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, num_tasks_to_run+1), R[-1], marker='o')
    plt.xlabel("Task index (1..10)")
    plt.ylabel("Accuracy after training all 10 tasks")
    plt.title("Validation accuracy per task after Task 10 (baseline)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lastrow_baseline.png", dpi=150)
    print("Saved plot: lastrow_baseline.png")

    # baseline vs dropout=0.0
    plt.figure()
    plt.plot(range(1, num_tasks_to_run+1), R[-1], marker='o', label='dropout=0.5')
    plt.plot(range(1, num_tasks_to_run+1), R_drop0[-1], marker='s', label='dropout=0.0')
    plt.xlabel("Task index (1..10)")
    plt.ylabel("Accuracy after training all 10 tasks")
    plt.title("Baseline vs No Dropout (last row of R)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lastrow_baseline_vs_drop0.png", dpi=150)
    print("Saved plot: lastrow_baseline_vs_drop0.png")
except Exception as e:
    print("Plotting skipped:", e)
