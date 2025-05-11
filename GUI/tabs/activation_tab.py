from pathlib import Path
import subprocess
import customtkinter as ctk
from tkinter import messagebox

from GUI.utils import activate_pose2sim

class ActivationTab:
    def __init__(self, parent, app, simplified=False):
        self.parent = parent
        self.app = app
        self.simplified = simplified  # Flag for 2D mode
        
        # Create main frame
        self.frame = ctk.CTkFrame(parent)
        
        # Build the UI
        self.build_ui()
    
    def get_title(self):
        """Return the tab title"""
        return self.app.lang_manager.get_text('activation_tab')
    
    def get_settings(self):
        """Get the activation settings"""
        return {}  # This tab doesn't need to add settings to the config file
    
    def build_ui(self):
        # Create main container
        self.content_frame = ctk.CTkFrame(self.frame)
        self.content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Tab title
        ctk.CTkLabel(
            self.content_frame,
            text=self.get_title(),
            font=('Helvetica', 24, 'bold')
        ).pack(pady=20)
        
        # Description
        launch_text = "Launch Sports2D" if self.app.analysis_mode == '2d' else "Launch Pose2Sim"
        ctk.CTkLabel(
            self.content_frame,
            text=f"Start {launch_text} with Anaconda Prompt:",
            font=('Helvetica', 16)
        ).pack(pady=10)
        
        # Card frame for activation options
        card_frame = ctk.CTkFrame(self.content_frame)
        card_frame.pack(pady=40)
        
        # Only show Anaconda Prompt Button
        ctk.CTkButton(
            card_frame,
            text=self.app.lang_manager.get_text('launch_conda'),
            command=lambda: self.activate_with_method('conda'),
            width=250,
            height=50,
            font=('Helvetica', 16),
            fg_color="#4CAF50",
            hover_color="#388E3C"
        ).pack(pady=20)
        
        # Setup and configuration notice
        notice_frame = ctk.CTkFrame(self.content_frame, fg_color=("gray90", "gray20"))
        notice_frame.pack(fill='x', pady=20, padx=20)
        
        env_name = "Sports2D" if self.app.analysis_mode == '2d' else "Pose2Sim"
        ctk.CTkLabel(
            notice_frame,
            text=f"💡 Make sure your {env_name} conda environment is properly set up before launching.",
            wraplength=600,
            font=('Helvetica', 14),
            text_color=("gray20", "gray90")
        ).pack(pady=10, padx=10)
    
    def merge_nested_dicts(self, d1, d2):
        """Recursively merge two nested dictionaries"""
        for key, value in d2.items():
            if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                self.merge_nested_dicts(d1[key], value)
            else:
                d1[key] = value
    
    def activate_with_method(self, method):
        """Activate Pose2Sim or Sports2D with the specified method"""
        # Update the config file first
        if self.app.analysis_mode == '2d':
            config_path = Path(self.app.participant_name) / 'Config_demo.toml'
        else:
            config_path = Path(self.app.participant_name) / 'Config.toml'
        
        # Collect all settings from tabs
        settings = {}
        for name, tab in self.app.tabs.items():
            if hasattr(tab, 'get_settings'):
                tab_settings = tab.get_settings()
                print(f"Collecting settings from tab '{name}':", tab_settings)  # Debug print
                # Merge settings
                for section, data in tab_settings.items():
                    if section not in settings:
                        settings[section] = {}
                    if isinstance(data, dict) and isinstance(settings[section], dict):
                        self.merge_nested_dicts(settings[section], data)
                    else:
                        settings[section] = data
        
        print("Final settings to be applied:", settings)  # Debug print
        
        # Generate config file
        if self.app.analysis_mode == '2d':
            success = self.app.config_generator.generate_2d_config(config_path, settings)
        else:
            success = self.app.config_generator.generate_3d_config(config_path, settings)
            
            # For batch mode, also generate configs for each trial
            if self.app.process_mode == 'batch':
                for i in range(1, self.app.num_trials + 1):
                    trial_config_path = Path(self.app.participant_name) / f'Trial_{i}' / 'Config.toml'
                    success = success and self.app.config_generator.generate_3d_config(
                        trial_config_path, settings
                    )
        
        if not success:
            messagebox.showerror(
                "Error",
                "Failed to generate configuration file. Please check your settings."
            )
            return
        
        try:
            # Determine skip flags based on mode
            skip_pose_estimation = False
            skip_synchronization = False
            
            if self.app.analysis_mode == '3d':
                pose_model = self.app.tabs['pose_model'].pose_model_var.get()
                if pose_model != 'Body_with_feet':
                    # Warn user about pose model compatibility
                    response = messagebox.askyesno(
                        "Warning",
                        f"The selected pose model '{pose_model}' may not be fully integrated in Pose2Sim. "
                        "This might require manual pose estimation.\n\n"
                        "Do you want to continue?"
                    )
                    if not response:
                        return
                    skip_pose_estimation = True
                
                # Check synchronization setting
                skip_synchronization = self.app.tabs['synchronization'].sync_videos_var.get() == 'yes'
            
            # Create activation script
            script_path = activate_pose2sim(
                self.app.participant_name,
                method=method,
                skip_pose_estimation=skip_pose_estimation,
                skip_synchronization=skip_synchronization,
                analysis_mode=self.app.analysis_mode
            )
            
            # Launch the script
            process = subprocess.Popen(script_path, 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.STDOUT, 
                                       text=True,
                                       shell=True)
            for line in process.stdout:
                print(line, end='')
            return_code = process.wait()
            
            # Update progress
            if hasattr(self.app, 'update_tab_indicator'):
                self.app.update_tab_indicator('activation', True)
            if hasattr(self.app, 'update_progress_bar'):
                self.app.update_progress_bar(100)  # Activation is the final step - 100%
            
            # Show success message
            app_name = "Sports2D" if self.app.analysis_mode == '2d' else "Pose2Sim"
            messagebox.showinfo(
                "Activation Started",
                f"{app_name} has been launched with Anaconda Prompt."
            )
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to activate: {str(e)}"
            )