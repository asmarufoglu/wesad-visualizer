# WESAD Visualizer

This is a Python-based GUI application for visualizing and analyzing physiological signals from the WESAD dataset.

### File structure

wesad-visualizer-app/
├── main.py launches the GUI app
├── requirements.txt # required Python packages
├── README.md #project description and usage
│
├── gui/ 
│ └── main_gui.py # Main Tkinter application logic
│
├── src/ # 
│ ├── preprocessing.py # Signal processing & feature extraction
│ └── predict.py # Model loading, training, and prediction
  └── plot.py # graphs, data visualizations 
│
├── model/ 
│ └── trained_model.pkl saved model file after training
│ 


###  Features
- Load `.pkl` files from the WESAD dataset  
- Visualize signals (EDA, TEMP, BVP, ACC) with various plots  
- Train models (Decision Tree, Random Forest, SVM)  
- Display predictions, confusion matrix, and analysis tabs

###  How to Run

1. Extract the ZIP file  
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
3. Run the code
   ```bash
   python main.py
   


