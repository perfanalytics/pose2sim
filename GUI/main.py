import customtkinter as ctk
from GUI.app import Pose2SimApp
from GUI.intro import IntroWindow

def main():
    # Set appearance mode and color theme
    ctk.set_appearance_mode("System")  # Options: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Options: "blue" (default), "green", "dark-blue"

    # Run Intro Window
    # Determine appearance mode for IntroWindow
    current_appearance_mode = ctk.get_appearance_mode().lower()
    if current_appearance_mode not in ['light', 'dark']:
        current_appearance_mode = 'dark' # Default to dark

    intro = IntroWindow(color=current_appearance_mode)
    intro.run()
    
    # Create the Tkinter root window
    root = ctk.CTk()
    
    # Initialize and run the application
    app = Pose2SimApp(root)
    
    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()