import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Time Series Plot
def plot_time_series(signal, signal_name="Signal", frame=None):
    """
    Plots a basic time series line chart of the selected signal.
    """
    signal = signal.flatten()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(signal)
    ax.set_title(f"{signal_name} - Time Series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True)

    # If a Tkinter frame is provided, embed the plot into the frame
    if frame:
        for widget in frame.winfo_children():
            widget.destroy()  # Clear old plots
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    plt.close(fig)  # Close the figure.

# Boxplot
def plot_boxplot(signal, signal_name="Signal", frame=None):
    """
    Plots a boxplot to show signal distribution and potential outliers.
    """
    signal = signal.flatten()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=signal, ax=ax)
    ax.set_title(f"{signal_name} - Boxplot")
    ax.grid(True)

    if frame:
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    plt.close(fig)

# Scatter Plot between Two Signals
def plot_scatter(signal_x, signal_y, name_x="X", name_y="Y", frame=None):
    """
    Plots a scatter plot comparing two different signals.
    Useful to visualize their relationship.
    """
    signal_x, signal_y = signal_x.flatten(), signal_y.flatten()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=signal_x, y=signal_y, s=10, ax=ax)
    ax.set_title(f"Scatter Plot: {name_x} vs {name_y}")
    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.grid(True)

    if frame:
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    plt.close(fig)

# Correlation Heatmap
def plot_correlation_heatmap(signals_dict, frame=None):
    """
    Computes and visualizes the Pearson correlation matrix
    between multiple signals using a heatmap.
    """
    # Trim all signals to the same length to build a proper DataFrame
    min_len = min([v.shape[0] for v in signals_dict.values()])
    trimmed_signals = {k: v.flatten()[:min_len] for k, v in signals_dict.items()}
    df = pd.DataFrame(trimmed_signals)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")

    if frame:
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    plt.close(fig)

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names, frame=None):
    """
    Plots a confusion matrix to evaluate classification performance.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    if frame:
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    plt.close(fig)

# True vs Predicted Line Plot
def plot_target_vs_prediction(y_true, y_pred, frame=None):
    """
    Draws a line plot comparing true vs predicted labels over time.
    Useful to detect where the model diverges from ground truth.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(y_true, label='True Labels', color='blue', marker='o', linestyle='dashed')
    ax.plot(y_pred, label='Predicted Labels', color='orange', marker='x')
    ax.set_title("True vs Predicted (Time-aligned)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Label")
    ax.legend()
    ax.grid(True)

    if frame:
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    plt.close(fig)

# Scatter: True vs Predicted
def plot_scatter_true_vs_pred(y_true, y_pred, frame=None):
    """
    Plots a scatter plot of true vs predicted labels.
    Points far from the diagonal line suggest prediction errors.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.7, color="purple")
    ax.set_xlabel("True Labels")
    ax.set_ylabel("Predicted Labels")
    ax.set_title("Scatter Plot: True vs Predicted")
    ax.grid(True)

    if frame:
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    plt.close(fig)

# Distribution of True and Predicted Labels
def plot_prediction_distribution(y_true, y_pred, frame=None):
    """
    Shows the distribution (frequency count) of true vs predicted labels.
    Helps to detect model bias or label imbalance.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(y_true, label="True", color="blue", kde=False, bins=3, ax=ax)
    sns.histplot(y_pred, label="Predicted", color="orange", kde=False, bins=3, ax=ax)
    ax.set_title("Label Distributions")
    ax.set_xlabel("Label")
    ax.legend()

    if frame:
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    plt.close(fig)
