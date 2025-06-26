# Entry point of the application
# Imports the main GUI app class from the gui module
from gui.main_gui import WESADVisualizerApp

# Run the app if this file is executed directly
if __name__ == "__main__":
    # Create an instance of the app
    app = WESADVisualizerApp()
    
    # Start the Tkinter event loop
    app.mainloop()
