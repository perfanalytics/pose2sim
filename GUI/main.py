import customtkinter as ctk
from GUI.app import Pose2SimApp

def main():
    # Set appearance mode and color theme
    ctk.set_appearance_mode("System")  # Options: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Options: "blue" (default), "green", "dark-blue"
    
    # Create the Tkinter root window
    root = ctk.CTk()
    
    # Initialize and run the application
    app = Pose2SimApp(root)
    
    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()