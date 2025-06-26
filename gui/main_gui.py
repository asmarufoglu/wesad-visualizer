import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
from src.predict import load_model, predict_sample
from src.preprocessing import load_subject_data, extract_features, downsample_labels_by_mode
from src.plot import (
    plot_time_series, plot_boxplot, plot_scatter, plot_correlation_heatmap,
    plot_confusion_matrix, plot_target_vs_prediction,
    plot_scatter_true_vs_pred, plot_prediction_distribution
)

# Define the main application class, which inherits from Tkinter's root window
class WESADVisualizerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set the title of the main window
        self.title("WESAD Visualizer")

        # Define the size of the window
        self.geometry("1000x750")

        # Set a light gray background color for the entire window
        self.configure(bg="#f0f0f0")

        # These attributes will store the selected file and the trained classifier
        self.selected_file = None
        self.clf = None

        # Call the method that creates and places all GUI components
        self.create_widgets()

    def create_widgets(self):
        # HEADER SECTION 
        # Create a horizontal header frame at the top of the window
        header_frame = tk.Frame(self, bg="#4a6baf")
        header_frame.pack(fill=tk.X, pady=10)

        # Add a title label to the header
        tk.Label(
            header_frame, text="WESAD Visualizer", font=("Arial", 16),
            bg="#4a6baf", fg="white"
        ).pack(side=tk.LEFT, padx=10)

        # Add a button to select a .pkl data file
        self.file_button = tk.Button(
            header_frame, text="Select .pkl file", bg="#4a6baf",
            fg="white", relief=tk.FLAT, command=self.select_file
        )
        self.file_button.pack(side=tk.RIGHT, padx=10)

        # MAIN SECTION
        # Create the main content area where all feature panels will be placed
        main_frame = tk.Frame(self, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # SIGNAL EXPLORER PANEL
        # This section allows users to explore and plot different physiological signals
        signal_explorer_frame = tk.LabelFrame(
            main_frame, text="Signal Explorer", font=("Arial", 12),
            bg="#f0f0f0", fg="#4a6baf"
        )
        signal_explorer_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Label to indicate signal selection
        tk.Label(signal_explorer_frame, text="Select Signal:", bg="#f0f0f0").pack(anchor=tk.W, pady=(10, 0))

        # Create radio buttons for each signal type
        self.signal_var = tk.StringVar(value="EDA")
        for signal in ["ACC", "BVP", "EDA", "TEMP"]:
            tk.Radiobutton(
                signal_explorer_frame, text=signal, variable=self.signal_var,
                value=signal, bg="#f0f0f0"
            ).pack(anchor=tk.W, pady=5)

        # Dropdown menu for selecting plot type
        tk.Label(signal_explorer_frame, text="Select Plot Type:", bg="#f0f0f0").pack(anchor=tk.W, pady=5)
        self.plot_type_selection = ttk.Combobox(
            signal_explorer_frame,
            values=["Line Plot", "Box Plot", "Scatter Plot", "Correlation Heatmap"]
        )
        self.plot_type_selection.pack(fill=tk.X, pady=5)

        # Button to generate the selected plot
        tk.Button(
            signal_explorer_frame, text="Generate Plot", bg="#4a6baf",
            fg="white", command=self.generate_plot
        ).pack(fill=tk.X, pady=5)

        # Area where plots will be displayed
        self.plot_area = tk.LabelFrame(
            signal_explorer_frame, text="Plot appears here", bg="#f0f0f0",
            fg="#4a6baf", height=300, width=500
        )
        self.plot_area.pack(fill=tk.BOTH, expand=False, pady=10)
        self.plot_area.pack_propagate(False)

        # FEATURE SUMMARY PANEL
        # Shows statistical summarys (mean, std, max, min) of selected features
        feature_summary_frame = tk.LabelFrame(
            main_frame, text="Feature Summary", font=("Arial", 12),
            bg="#f0f0f0", fg="#4a6baf", width=200, height=180
        )
        feature_summary_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        feature_summary_frame.pack_propagate(False)

        # Labels to display statistical results
        tk.Label(feature_summary_frame, text="Summary Statistics", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.mean_label = tk.Label(feature_summary_frame, text="Mean: ", bg="#f0f0f0")
        self.mean_label.pack(anchor=tk.W, pady=2)
        self.std_label = tk.Label(feature_summary_frame, text="Std: ", bg="#f0f0f0")
        self.std_label.pack(anchor=tk.W, pady=2)
        self.max_label = tk.Label(feature_summary_frame, text="Max: ", bg="#f0f0f0")
        self.max_label.pack(anchor=tk.W, pady=2)
        self.min_label = tk.Label(feature_summary_frame, text="Min: ", bg="#f0f0f0")
        self.min_label.pack(anchor=tk.W, pady=2)

        # MODEL CONTROL PANEL
        # Section for selecting, training and continuing with a model
        self.model_control_frame = tk.LabelFrame(
            main_frame, text="Model Control", font=("Arial", 12),
            bg="#f0f0f0", fg="#4a6baf"
        )
        self.model_control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Dropdown to choose the classification model
        tk.Label(self.model_control_frame, text="Model:", bg="#f0f0f0").pack(anchor=tk.W, pady=(10, 0))
        self.model_selection = ttk.Combobox(
            self.model_control_frame,
            values=["Decision Tree Classifier", "Random Forest Classifier", "SVM"]
        )
        self.model_selection.pack(fill=tk.X, pady=5)
        self.model_selection.set("Decision Tree Classifier")

        # Button to start training the selected model
        tk.Button(self.model_control_frame, text="Train Model", bg="#4a6baf", fg="white", command=self.train_model).pack(fill=tk.X, pady=5)

        # Button to continue to prediction screen
        tk.Button(self.model_control_frame, text="Continue", bg="#4a6baf", fg="white", command=self.continue_to_model).pack(fill=tk.X, pady=5)

        # Label to show the model's accuracy
        self.accuracy_label = tk.Label(self.model_control_frame, text="Accuracy: X%", bg="#f0f0f0")
        self.accuracy_label.pack(anchor=tk.W, pady=5)

        # PREDICTION PANEL
        # Displays prediction results based on model and feature input
        prediction_frame = tk.LabelFrame(
            main_frame, text="Predict Emotional State", font=("Arial", 12),
            bg="#f0f0f0", fg="#4a6baf"
        )
        prediction_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(prediction_frame, text="Model Output:", bg="#f0f0f0").pack(anchor=tk.W, pady=(10, 0))
        self.prediction_label = tk.Label(prediction_frame, text="Predicted: ", bg="#f0f0f0", font=("Arial", 10, "bold"), fg="black")
        self.prediction_label.pack(anchor=tk.W, pady=5)

        # Buttons to restart or load a different data file
        tk.Button(prediction_frame, text="Restart", bg="#4a6baf", fg="white", command=self.restart).pack(fill=tk.X, pady=5)
        tk.Button(prediction_frame, text="Load Another File", bg="#4a6baf", fg="white", command=self.select_file).pack(fill=tk.X, pady=5)

        # ANALYSIS PANELS
        # Bottom section includes advanced analysis panels
        bottom_frame = tk.Frame(main_frame, bg="#f0f0f0")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)

        # Confusion matrix display area
        self.confusion_frame = tk.LabelFrame(
            bottom_frame, text="Confusion Matrix", font=("Arial", 12),
            bg="#f0f0f0", fg="#4a6baf", height=300, width=600
        )
        self.confusion_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.confusion_frame.pack_propagate(False)

        # Analysis of predictions using different plot types
        self.pred_analysis_frame = tk.LabelFrame(
            bottom_frame, text="Prediction Analysis", font=("Arial", 12),
            bg="#f0f0f0", fg="#4a6baf", height=300, width=600
        )
        self.pred_analysis_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.pred_analysis_frame.pack_propagate(False)

        # Tabbed view for different types of prediction analysis plots
        self.analysis_notebook = ttk.Notebook(self.pred_analysis_frame)
        self.analysis_notebook.pack(fill=tk.BOTH, expand=True)

        # Create individual tabs for different visualizations. i cant understand how this part work but i keep this. 
        self.tab_line = tk.Frame(self.analysis_notebook, width=600, height=300)
        self.tab_scatter = tk.Frame(self.analysis_notebook, width=600, height=300)
        self.tab_dist = tk.Frame(self.analysis_notebook, width=600, height=300)

        # Add tabs to the notebook
        self.analysis_notebook.add(self.tab_line, text="Line Graph")
        self.analysis_notebook.add(self.tab_scatter, text="Scatter Plot")
        self.analysis_notebook.add(self.tab_dist, text="Distribution")
    # Method to open a file dialog and allow the user to choose a WESAD .pkl file
    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select WESAD pkl file",
            filetypes=[("Pickle Files", "*.pkl")]
        )
        if file_path:
            # Save selected file path and update the button text to show file name
            self.selected_file = file_path
            self.file_button.config(text=f"Selected: {file_path.split('/')[-1]}")

    # Method to generate a plot based on the selected signal and plot type
    def generate_plot(self):
        if not self.selected_file:
            # Show warning if no file is selected
            messagebox.showwarning("No File", "Please select a file first.")
            return
        try:
            # Load signal data from the selected file
            data = load_subject_data(self.selected_file)
            signal = np.array(data["signal"]["wrist"][self.signal_var.get()].flatten())
            plot_type = self.plot_type_selection.get()

            # Clear previous plot if any
            for widget in self.plot_area.winfo_children():
                widget.destroy()

            # Plot based on user-selected type
            if plot_type == "Line Plot":
                plot_time_series(signal, self.signal_var.get(), self.plot_area)
            elif plot_type == "Box Plot":
                plot_boxplot(signal, self.signal_var.get(), self.plot_area)
            elif plot_type == "Scatter Plot":
                # TEMP used as X, selected signal as Y
                temp = np.array(data["signal"]["wrist"]["TEMP"].flatten())
                plot_scatter(temp, signal, "TEMP", self.signal_var.get(), self.plot_area)
            elif plot_type == "Correlation Heatmap":
                # Prepare all signals for correlation
                all_signals = {k: np.array(v).flatten() for k, v in data["signal"]["wrist"].items()}
                plot_correlation_heatmap(all_signals, self.plot_area)

            # Update basic statistics (mean, std, etc.)
            self.update_summary_statistics(signal)
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))

    # Method to compute and display summary statistics of the signal
    def update_summary_statistics(self, signal):
        self.mean_label.config(text=f"Mean: {np.mean(signal):.2f}")
        self.std_label.config(text=f"Std: {np.std(signal):.2f}")
        self.max_label.config(text=f"Max: {np.max(signal):.2f}")
        self.min_label.config(text=f"Min: {np.min(signal):.2f}")

    # Method to move to prediction step after training. for additional graphs about prediction.
    def continue_to_model(self):
        if self.clf is not None:
            self.test_model()
        else:
            messagebox.showwarning("No Model", "Please train a model first.")

    # Method to train a selected classifier on the WESAD EDA signal
    def train_model(self):
        if not self.selected_file:
            messagebox.showwarning("No File", "Please select a file first.")
            return
        try:
            # Load and preprocess the EDA signal and labels
            data = load_subject_data(self.selected_file)
            eda = np.array(data["signal"]["wrist"]["EDA"].flatten())
            labels = np.array(data["label"])
            label_ds = downsample_labels_by_mode(labels, len(eda))
            X, y = extract_features(eda, label_ds)

            # Filter out labels not in [1,2,3] (Rest, Amusement, Stress)
            mask = np.isin(y, [1, 2, 3])
            X, y = X[mask], y[mask]

            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Select model based on user choice
            model_name = self.model_selection.get()
            model = RandomForestClassifier(n_estimators=100) if model_name == "Random Forest Classifier" \
                else SVC() if model_name == "SVM" \
                else DecisionTreeClassifier()

            # Train and evaluate the model
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            self.clf = model

            # Save trained model to file
            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)

            # Update accuracy label and notify user
            self.accuracy_label.config(text=f"Accuracy: {acc:.2f}")
            messagebox.showinfo("Training", f"Model training completed with accuracy: {acc:.2f}")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    # Method to test the model on full data and show visual evaluation
    def test_model(self):
        if not self.selected_file:
            messagebox.showwarning("No File", "Please select a file first.")
            return
        try:
            # Load and preprocess EDA signal and labels again for full dataset testing
            data = load_subject_data(self.selected_file)
            eda = np.array(data["signal"]["wrist"]["EDA"].flatten())
            labels = np.array(data["label"])
            label_ds = downsample_labels_by_mode(labels, len(eda))
            X, y = extract_features(eda, label_ds)
            mask = np.isin(y, [1, 2, 3])
            X, y = X[mask], y[mask]

            # Load model from file if needed
            if not self.clf:
                self.clf = load_model("trained_model.pkl")

            # Perform prediction
            y_pred = self.clf.predict(X)
            pred_label, pred_class = predict_sample(self.clf, [X[-1]])

            # Update prediction label with styling
            self.set_prediction(pred_class)

            # Clear previous plots
            for widget in self.confusion_frame.winfo_children(): widget.destroy()
            for tab in [self.tab_line, self.tab_scatter, self.tab_dist]:
                for widget in tab.winfo_children():
                    widget.destroy()

            # Visualize model performance
            plot_confusion_matrix(y, y_pred, [1, 2, 3], self.confusion_frame)
            plot_target_vs_prediction(y, y_pred, self.tab_line)
            plot_scatter_true_vs_pred(y, y_pred, self.tab_scatter)
            plot_prediction_distribution(y, y_pred, self.tab_dist)
        except Exception as e:
            messagebox.showerror("Testing Error", str(e))

    # Method to update prediction text and background color based on emotion
    def set_prediction(self, pred):
        self.prediction_label.config(text=f"Predicted: {pred}")
        colors = {"Stress": ("red", "white"), "Rest": ("green", "white"), "Amusement": ("blue", "white")}
        bg, fg = colors.get(pred, ("#f0f0f0", "black"))
        self.prediction_label.config(bg=bg, fg=fg)

    # Method to reset the entire interface to its initial state
    def restart(self):
        # Clear file, model, and all dynamic outputs
        self.selected_file = None
        self.clf = None
        self.file_button.config(text="Select .pkl file")
        self.prediction_label.config(text="Predicted: ", bg="#f0f0f0", fg="black")

        # Clear all plots and analysis areas
        for area in [self.plot_area, self.confusion_frame, self.tab_line, self.tab_scatter, self.tab_dist]:
            for widget in area.winfo_children():
                widget.destroy()

        # Reset all statistics labels
        for lbl in [self.mean_label, self.std_label, self.max_label, self.min_label]:
            lbl.config(text=lbl.cget("text").split(":")[0] + ": ")

        # Reset accuracy text
        self.accuracy_label.config(text="Accuracy: X%")

    # Entry point: launch the app
    if __name__ == "__main__":
        app = WESADVisualizerApp()
        app.mainloop()
