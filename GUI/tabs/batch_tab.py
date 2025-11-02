from pathlib import Path
import customtkinter as ctk
from tkinter import messagebox
import toml

class BatchTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        # Create main frame
        self.frame = ctk.CTkFrame(parent)
        
        # Build the UI
        self.build_ui()
    
    def get_title(self):
        """Return the tab title"""
        return self.app.lang_manager.get_text('batch_tab')
    
    def get_settings(self):
        """Get the batch settings"""
        # Batch tab doesn't add specific settings to the main config
        # as it manages individual trial configs separately
        return {}
    
    def build_ui(self):
        # Create scrollable container
        self.content_frame = ctk.CTkScrollableFrame(self.frame)
        self.content_frame.pack(fill='both', expand=True, padx=0, pady=0)
        
        # Header
        header_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        header_frame.pack(fill='x', pady=(0, 20))
        
        ctk.CTkLabel(
            header_frame, 
            text='Trial-Specific Configuration', 
            font=('Helvetica', 20, 'bold')
        ).pack(anchor='w')
        
        # Information label
        info_label = ctk.CTkLabel(
            self.content_frame,
            text="Configure trial-specific parameters. Other settings will be inherited from the main configuration.",
            wraplength=800
        )
        info_label.pack(pady=10)
        
        # Buttons for trials
        self.trials_frame = ctk.CTkFrame(self.content_frame)
        self.trials_frame.pack(fill='x', pady=10)
        
        # Create buttons for each trial
        self.create_trial_buttons()
    
    def create_trial_buttons(self):
        """Create buttons for each trial"""
        # Clear any existing buttons
        for widget in self.trials_frame.winfo_children():
            widget.destroy()
        
        # Create grid layout for trial buttons
        rows = (self.app.num_trials + 1) // 2
        cols = 2
        
        for i in range(1, self.app.num_trials + 1):
            row = (i - 1) // cols
            col = (i - 1) % cols
            
            frame = ctk.CTkFrame(self.trials_frame)
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Trial label
            ctk.CTkLabel(
                frame,
                text=f"Trial {i}",
                font=("Helvetica", 16, "bold")
            ).pack(anchor='w', padx=15, pady=(15, 5))
            
            # Status indicator - check if config file exists
            config_path = Path(self.app.participant_name) / f'Trial_{i}' / 'Config.toml'
            status = "○ Not configured" if not config_path.exist() else "● Configured"
            status_color = "gray" if not config_path.exists() else "green"
            
            status_frame = ctk.CTkFrame(frame, fg_color="transparent")
            status_frame.pack(fill='x', padx=15, pady=5)
            
            status_indicator = ctk.CTkLabel(
                status_frame,
                text=status,
                text_color=status_color
            )
            status_indicator.pack(side="left")
            
            # Configure button
            configure_button = ctk.CTkButton(
                frame,
                text="Configure Trial",
                command=lambda trial_num=i: self.configure_trial(trial_num),
                height=30
            )
            configure_button.pack(pady=15, padx=15)
        
        # Make sure rows and columns expand properly
        for i in range(rows):
            self.trials_frame.grid_rowconfigure(i, weight=1)
        for i in range(cols):
            self.trials_frame.grid_columnconfigure(i, weight=1)
    
    def configure_trial(self, trial_number):
        """Open configuration window for trial-specific settings"""
        config_window = ctk.CTkToplevel(self.app.root)
        config_window.title(f"Configure Trial_{trial_number}")
        config_window.geometry("800x600")
        
        main_frame = ctk.CTkFrame(config_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        scroll_frame = ctk.CTkScrollableFrame(main_frame)
        scroll_frame.pack(fill='both', expand=True)
        
        # Load trial configuration
        config_path = Path(self.app.participant_name) / f'Trial_{trial_number}' / 'Config.toml'
        try:
            if config_path.exists():
                config = toml.load(config_path)
            else:
                # Use parent config as base
                parent_config_path = Path(self.app.participant_name) / 'Config.toml'
                if parent_config_path.exists():
                    config = toml.load(parent_config_path)
                else:
                    # Create a new config
                    config = {}
        except Exception as e:
            messagebox.showerror("Error", f"Could not load configuration for Trial_{trial_number}: {str(e)}")
            return
        
        # Dictionary to store all settings variables
        settings_vars = {}
        
        # Sections to configure
        sections = {
            'Project Settings': [
                ('frame_range', '[]', 'entry'),
                ('multi_person', True, 'checkbox')
            ],
            'Pose Estimation': [
                ('pose_model', 'Body_with_feet', 'combobox', ['Body_with_feet', 'Whole_body_wrist', 'Whole_body', 'Body']),
                ('mode', 'balanced', 'combobox', ['lightweight', 'balanced', 'performance']),
                ('det_frequency', 60, 'entry')
            ],
            'Synchronization': [
                ('keypoints_to_consider', 'all', 'entry'),
                ('approx_time_maxspeed', 'auto', 'entry'),
                ('time_range_around_maxspeed', 2.0, 'entry')
            ],
            'Filtering': [
                ('type', 'butterworth', 'combobox', ['butterworth', 'kalman', 'gaussian', 'LOESS', 'median']),
                ('cut_off_frequency', 6, 'entry')
            ]
        }
        
        row = 0
        for section_name, settings in sections.items():
            # Section header
            ctk.CTkLabel(
                scroll_frame, 
                text=section_name, 
                font=('Helvetica', 16, 'bold')
            ).grid(row=row, column=0, columnspan=2, pady=(15,5), sticky='w')
            row += 1
            
            for setting_info in settings:
                setting_name = setting_info[0]
                default_value = setting_info[1]
                input_type = setting_info[2]
                
                # Get current value from config if available
                current_value = self.get_config_value(config, setting_name, default_value)
                
                # Create label
                ctk.CTkLabel(
                    scroll_frame, 
                    text=setting_name.replace('_', ' ').title() + ':',
                    anchor='w'
                ).grid(row=row, column=0, pady=2, padx=5, sticky='w')
                
                # Create appropriate input widget based on type
                if input_type == 'checkbox':
                    var = ctk.BooleanVar(value=current_value)
                    ctk.CTkCheckBox(
                        scroll_frame, 
                        text="", 
                        variable=var,
                        onvalue=True,
                        offvalue=False
                    ).grid(row=row, column=1, pady=2, padx=5, sticky='w')
                
                elif input_type == 'combobox':
                    options = setting_info[3]
                    var = ctk.StringVar(value=current_value)
                    ctk.CTkOptionMenu(
                        scroll_frame,
                        variable=var,
                        values=options
                    ).grid(row=row, column=1, pady=2, padx=5, sticky='w')
                
                else:  # Default to entry
                    var = ctk.StringVar(value=str(current_value))
                    ctk.CTkEntry(
                        scroll_frame,
                        textvariable=var,
                        width=200
                    ).grid(row=row, column=1, pady=2, padx=5, sticky='w')
                
                # Store variable reference for later retrieval
                settings_vars[setting_name] = (var, input_type)
                row += 1
        
        # Save button at the bottom
        save_button = ctk.CTkButton(
            main_frame,
            text="Save Trial Configuration",
            command=lambda: self.save_trial_configuration(config_path, config, settings_vars, trial_number, config_window),
            width=200,
            height=40
        )
        save_button.pack(pady=10)
    
    def get_config_value(self, config, setting_name, default_value):
        """Get a value from the config, handling nested paths"""
        try:
            # Project settings
            if setting_name in ['frame_range', 'multi_person']:
                return config.get('project', {}).get(setting_name, default_value)
            
            # Pose settings
            elif setting_name in ['pose_model', 'mode', 'det_frequency']:
                return config.get('pose', {}).get(setting_name, default_value)
            
            # Synchronization settings
            elif setting_name in ['keypoints_to_consider', 'approx_time_maxspeed', 'time_range_around_maxspeed']:
                return config.get('synchronization', {}).get(setting_name, default_value)
            
            # Filtering settings
            elif setting_name == 'type':
                return config.get('filtering', {}).get(setting_name, default_value)
            elif setting_name == 'cut_off_frequency':
                return config.get('filtering', {}).get('butterworth', {}).get(setting_name, default_value)
            
            return default_value
        except:
            return default_value
    
    def save_trial_configuration(self, config_path, config, settings_vars, trial_number, config_window):
        """Save trial-specific configuration"""
        try:
            # Ensure the directories exist
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update config with new values
            for setting_name, (var, input_type) in settings_vars.items():
                value = var.get()
                
                # Convert value based on input type
                if input_type == 'checkbox':
                    # Boolean values
                    pass  # Already a boolean
                elif input_type in ['entry', 'combobox']:
                    # Try to convert to appropriate type
                    if setting_name == 'frame_range':
                        # Special handling for frame_range which should be a list
                        try:
                            value = eval(value)  # Safely evaluate as Python expression
                            if not isinstance(value, list):
                                value = []
                        except:
                            value = []
                    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                        # Convert numeric strings to numbers
                        value = float(value)
                        # Convert to int if it's a whole number
                        if value.is_integer():
                            value = int(value)
                
                # Update the appropriate section in the config
                self.set_config_value(config, setting_name, value)
            
            # Write the updated config
            with open(config_path, 'w', encoding='utf-8') as f:
                toml.dump(config, f)
            
            messagebox.showinfo("Success", f"Configuration for Trial_{trial_number} has been saved successfully!")
            
            # Update the trial buttons to reflect new configuration status
            self.create_trial_buttons()
            
            # Close the config window
            config_window.destroy()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def set_config_value(self, config, setting_name, value):
        """Set a value in the config, handling nested paths"""
        try:
            # Project settings
            if setting_name in ['frame_range', 'multi_person']:
                if 'project' not in config:
                    config['project'] = {}
                config['project'][setting_name] = value
            
            # Pose settings
            elif setting_name in ['pose_model', 'mode', 'det_frequency']:
                if 'pose' not in config:
                    config['pose'] = {}
                config['pose'][setting_name] = value
            
            # Synchronization settings
            elif setting_name in ['keypoints_to_consider', 'approx_time_maxspeed', 'time_range_around_maxspeed']:
                if 'synchronization' not in config:
                    config['synchronization'] = {}
                config['synchronization'][setting_name] = value
            
            # Filtering settings
            elif setting_name == 'type':
                if 'filtering' not in config:
                    config['filtering'] = {}
                config['filtering'][setting_name] = value
            elif setting_name == 'cut_off_frequency':
                if 'filtering' not in config:
                    config['filtering'] = {}
                if 'butterworth' not in config['filtering']:
                    config['filtering']['butterworth'] = {}
                config['filtering']['butterworth'][setting_name] = value
        
        except Exception as e:
            print(f"Error setting {setting_name}: {str(e)}")