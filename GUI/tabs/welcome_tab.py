import customtkinter as ctk
from PIL import Image, ImageTk
from pathlib import Path

class WelcomeTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        # Create main frame
        self.frame = ctk.CTkFrame(parent)
        self.frame.pack(expand=True, fill='both')
        
        # Show welcome screen
        self.show_welcome()
    
    def show_welcome(self):
        """Show the welcome screen with Pose2Sim logo and language selection"""
        # Add logo above the title
        favicon_path = Path(__file__).parents[1]/"assets/Pose2Sim_logo.png"
        self.top_image = Image.open(favicon_path)
        self.top_photo = ctk.CTkImage(light_image=self.top_image, dark_image=self.top_image, size=(246,246))
        image_label = ctk.CTkLabel(self.frame, image=self.top_photo, text="")
        image_label.pack(pady=(50, 20))

        # Title
        title_label = ctk.CTkLabel(
            self.frame, 
            text="Pose2Sim", 
            font=("Helvetica", 72, "bold")
        )
        title_label.pack(pady=(50, 10))
      
       
       # Cards container
        cards_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        cards_frame.pack(pady=20)
        
        # 2D Analysis Card
        analysis_2d_card = ctk.CTkFrame(cards_frame)
        analysis_2d_card.pack(side="left", padx=20, fill="both")
        
        analysis_2d_label = ctk.CTkLabel(
            analysis_2d_card, 
            text=self.app.lang_manager.get_text("2d_analysis"), 
            font=("Helvetica", 22, "bold")
        )
        analysis_2d_label.pack(pady=(20, 10))
        analysis_2d_label.translation_key = "2d_analysis"
        
        analysis_2d_single = ctk.CTkLabel(
            analysis_2d_card, 
            text=self.app.lang_manager.get_text("single_camera"), 
            font=("Helvetica", 14),
            height=80,
            wraplength=250
        )
        analysis_2d_single.pack(pady=(0, 10), padx=30)
        analysis_2d_single.translation_key = "single_camera"
        
        analysis_2d_button = ctk.CTkButton(
            analysis_2d_card,
            text=self.app.lang_manager.get_text("select"),
            width=200,
            height=40,
            font=("Helvetica", 14),
            command=lambda: self.select_analysis_mode("2d")
        )
        analysis_2d_button.pack(pady=(0, 10))
        analysis_2d_button.translation_key = "select"
        
        # 3D Analysis Card
        analysis_3d_card = ctk.CTkFrame(cards_frame)
        analysis_3d_card.pack(side="left", padx=10, fill="both")
        
        analysis_3d_label = ctk.CTkLabel(
            analysis_3d_card, 
            text=self.app.lang_manager.get_text("3d_analysis"), 
            font=("Helvetica", 22, "bold")
        )
        analysis_3d_label.pack(pady=(20, 10))
        analysis_3d_label.translation_key = "3d_analysis"
        
        analysis_3d_multi = ctk.CTkLabel(
            analysis_3d_card, 
            text=self.app.lang_manager.get_text("multi_camera"), 
            font=("Helvetica", 14),
            height=80,
            wraplength=250
        )
        analysis_3d_multi.pack(pady=(0, 10), padx=30)
        analysis_3d_multi.translation_key = "multi_camera"
        
        analysis_3d_button = ctk.CTkButton(
            analysis_3d_card,
            text=self.app.lang_manager.get_text("select"),
            width=200,
            height=40,
            font=("Helvetica", 14),
            command=lambda: self.select_analysis_mode("3d")
        )
        analysis_3d_button.pack(pady=(0, 10))
        analysis_3d_button.translation_key = "select"
        
        # Version info
        version_label = ctk.CTkLabel(
            self.frame, 
            text="Version 2.0", 
            font=("Helvetica", 12)
        )
        version_label.pack(side="bottom", pady=20)
    
    def set_language(self, lang):
        """Set the language and move to analysis mode selection"""
        self.app.language = lang
        self.app.lang_manager.set_language(lang)
        self.app.change_language(lang)
        
        # Clear the frame
        for widget in self.frame.winfo_children():
            widget.destroy()
        
        # Show analysis mode selection
        self.show_analysis_mode_selection()
    
    
    def select_analysis_mode(self, mode):
        """Set the analysis mode and move to process mode selection"""
        self.analysis_mode = mode
        
        # Clear the frame
        for widget in self.frame.winfo_children():
            widget.destroy()
        
        # Show process mode selection
        self.show_process_mode_selection()
    
    def show_process_mode_selection(self):
        """Show the process mode selection screen"""
        # Title
        title_label = ctk.CTkLabel(
            self.frame, 
            text=self.app.lang_manager.get_text("Select the process mode"), 
            font=("Helvetica", 30, "bold")
        )
        title_label.pack(pady=(80, 40))
        title_label.translation_key = "Select the process mode"
        
        # Disable batch mode for 2D analysis
        if self.analysis_mode == "2d":
            # Skip process mode selection for 2D - always use single mode
            self.process_mode = "single"
            self.show_participant_name_input()
            return
        
        # Cards container for 3D analysis
        cards_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        cards_frame.pack(pady=20)
        
        # Single Mode Card
        single_card = ctk.CTkFrame(cards_frame)
        single_card.pack(side="left", padx=20, fill="both")
        
        single_trial_label = ctk.CTkLabel(
            single_card, 
            text=self.app.lang_manager.get_text("Single Trial"), 
            font=("Helvetica", 22, "bold")
        )
        single_trial_label.pack(pady=(30, 20))
        single_trial_label.translation_key = "Single Trial"
        
        
        single_trial_explanation_label = ctk.CTkLabel(
            single_card, 
            text="Process one recording session\nSimpler setup for single experiments", 
            font=("Helvetica", 14),
            height=80
        )
        single_trial_explanation_label.pack(pady=(0, 20), padx=30)
        single_trial_explanation_label.translation_key = "Process one recording session\nSimpler setup for single experiments"
        
        single_trial_button_label = ctk.CTkButton(
            single_card,
            text=self.app.lang_manager.get_text("Select"),
            width=200,
            height=40,
            font=("Helvetica", 14),
            command=lambda: self.select_process_mode("single")
        )
        single_trial_button_label.pack(pady=(0, 30))
        single_trial_button_label.translation_key = "Select"
        
        # Batch Mode Card
        batch_card = ctk.CTkFrame(cards_frame)
        batch_card.pack(side="left", padx=20, fill="both")
        
        batch_label = ctk.CTkLabel(
            batch_card, 
            text=self.app.lang_manager.get_text("batch_mode"), 
            font=("Helvetica", 22, "bold")
        )
        batch_label.pack(pady=(30, 20))
        batch_label.translation_key = "batch_mode"
        
        batch_explanation_label = ctk.CTkLabel(
            batch_card, 
            text="Process multiple trials at once\nIdeal for larger research studies", 
            font=("Helvetica", 14),
            height=80
        )
        batch_explanation_label.pack(pady=(0, 20), padx=30)
        batch_explanation_label.translation_key = "Process multiple trials at once\nIdeal for larger research studies"
        
        batch_select_button = ctk.CTkButton(
            batch_card,
            text=self.app.lang_manager.get_text("Select"),
            width=200,
            height=40,
            font=("Helvetica", 14),
            command=lambda: self.select_process_mode("Batch")
        )
        batch_select_button.pack(pady=(0, 30))
        batch_select_button.translation_key = "Select"
    
    def select_process_mode(self, mode):
        """Set the process mode and move to participant input"""
        self.process_mode = mode
        
        # Clear the frame
        for widget in self.frame.winfo_children():
            widget.destroy()
        
        # Show participant name input
        self.show_participant_name_input()
    
    def show_participant_name_input(self):
        """Show the participant name input screen"""
        # Create input frame
        input_frame = ctk.CTkFrame(self.frame)
        input_frame.pack(expand=True, fill="none", pady=100)
        
        # Header
        project_label = ctk.CTkLabel(
            input_frame, 
            text=self.app.lang_manager.get_text("Project Name"), 
            font=("Helvetica", 24, "bold")
        )
        project_label.pack(pady=(20, 30))
        project_label.translation_key = "Project Name"
        
        # Name input
        name_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        name_frame.pack(pady=20)
        
        project_prompt_label = ctk.CTkLabel(
            name_frame, 
            text=self.app.lang_manager.get_text("Enter a project name"), 
            font=("Helvetica", 16)
        )
        project_prompt_label.pack(side="left", padx=10)
        project_prompt_label.translation_key = "Enter a project name"
        
        self.participant_name_var = ctk.StringVar(value="my_project")
        name_entry = ctk.CTkEntry(name_frame, textvariable=self.participant_name_var, width=200, height=40)
        name_entry.pack(side="left", padx=10)
        
        # For batch mode, also ask for number of trials
        if hasattr(self, 'process_mode') and self.process_mode == "batch":
            trials_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
            trials_frame.pack(pady=20)
            
            trial_number_label = ctk.CTkLabel(
                trials_frame, 
                text=self.app.lang_manager.get_text("enter the trials number"), 
                font=("Helvetica", 16)
            )
            trial_number_label.pack(side="left", padx=10)
            trial_number_label.translation_key = "enter the trials number"
            
            self.num_trials_var = ctk.StringVar(value="3")
            trials_entry = ctk.CTkEntry(trials_frame, textvariable=self.num_trials_var, width=100, height=40)
            trials_entry.pack(side="left", padx=10)
        
        # Continue button
        button_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        button_frame.pack(pady=40)
        
        next_label = ctk.CTkButton(
            button_frame,
            text=self.app.lang_manager.get_text("next"),
            width=200,
            height=40,
            font=("Helvetica", 16),
            command=self.finalize_setup
        )
        next_label.pack(side="bottom")
        next_label.translation_key = "next"

        # Version info
        version_label = ctk.CTkLabel(
            self.frame, 
            text="Version 2.0", 
            font=("Helvetica", 12)
        )
        version_label.pack(side="bottom", pady=20)

    def finalize_setup(self):
        """Finalize setup and start the configuration process"""
        participant_name = self.participant_name_var.get().strip()
        if not participant_name:
            participant_name = "Participant"
        
        # For batch mode, get the number of trials
        num_trials = 0
        if hasattr(self, 'process_mode') and self.process_mode == "batch":
            try:
                num_trials = int(self.num_trials_var.get())
                if num_trials < 1:
                    raise ValueError
            except ValueError:
                num_trials = 3  # Default to 3 trials
        
        # Start the main configuration process
        self.app.start_configuration(
            analysis_mode=self.analysis_mode,
            process_mode=self.process_mode,
            participant_name=participant_name,
            num_trials=num_trials
        )
    
    def clear(self):
        """Clear the welcome tab frame when done"""
        self.frame.pack_forget()