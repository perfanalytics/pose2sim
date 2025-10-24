import os
import shutil
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog

class PoseModelTab:
    def __init__(self, parent, app, simplified=False):
        self.parent = parent
        self.app = app
        self.simplified = simplified  # Flag for 2D mode
        
        # Create main frame
        self.frame = ctk.CTkFrame(parent)
        
        # Initialize variables
        self.multiple_persons_var = ctk.StringVar(value='single')
        self.participant_height_var = ctk.StringVar(value='1.72')
        self.participant_mass_var = ctk.StringVar(value='70.0')
        self.pose_model_var = ctk.StringVar(value='Body_with_feet')
        self.mode_var = ctk.StringVar(value='balanced')
        self.tracking_mode_var = ctk.StringVar(value='sports2d')  # Added tracking mode variable
        self.video_extension_var = ctk.StringVar(value='mp4')
        
        # For 2D mode
        if simplified:
            self.video_input_var = ctk.StringVar(value='')
            self.visible_side_var = ctk.StringVar(value='auto')
            self.video_input_type_var = ctk.StringVar(value='file')  # 'file', 'webcam', or 'multiple'
            self.multiple_videos_list = []  # Store multiple video paths
        
        # For multiple people
        self.num_people_var = ctk.StringVar(value='2')
        self.people_details_vars = []
        self.participant_heights = []
        self.participant_masses = []
        
        # Build the UI
        self.build_ui()
    
    def get_title(self):
        """Return the tab title"""
        return self.app.lang_manager.get_text('pose_model_tab')
    
    def get_settings(self):
        """Get the pose model settings"""
        # Common settings for both 2D and 3D
        settings = {
            'pose': {
                'pose_model': self.pose_model_var.get(),
                'mode': self.mode_var.get(),
                'tracking_mode': self.tracking_mode_var.get(),
                'vid_img_extension': self.video_extension_var.get()
            },
            'project': {
                'multi_person': self.multiple_persons_var.get() == 'multiple'
            }
        }
        
        # Add participant details for 3D mode
        if not self.simplified:
            if self.multiple_persons_var.get() == 'single':
                settings['project']['participant_height'] = float(self.participant_height_var.get())
                settings['project']['participant_mass'] = float(self.participant_mass_var.get())
            else:
                settings['project']['participant_height'] = self.participant_heights
                settings['project']['participant_mass'] = self.participant_masses
        
        # Add 2D-specific settings
        if self.simplified:
            # CRITICAL FIX: Use 'base' section for 2D settings, not 'project'
            if 'base' not in settings:
                settings['base'] = {}
            
            # Handle different video input types for 2D mode
            if self.video_input_type_var.get() == 'webcam':
                settings['base']['video_input'] = 'webcam'
            elif self.video_input_type_var.get() == 'multiple' and self.multiple_videos_list:
                settings['base']['video_input'] = self.multiple_videos_list
            else:
                settings['base']['video_input'] = self.video_input_var.get()
            
            settings['base']['visible_side'] = self.visible_side_var.get()
            settings['base']['first_person_height'] = float(self.participant_height_var.get())
            
            # DEBUG: Print what we're returning
            print(f"DEBUG pose_model get_settings: video_input = {settings['base']['video_input']}")
        
        # CRITICAL FIX: Actually return the settings!
        return settings
    
    def build_ui(self):
        # Create scrollable container
        self.content_frame = ctk.CTkScrollableFrame(self.frame)
        self.content_frame.pack(fill='both', expand=True, padx=0, pady=0)
        
        # Tab title
        ctk.CTkLabel(
            self.content_frame,
            text=self.get_title(),
            font=('Helvetica', 24, 'bold')
        ).pack(anchor='w', pady=(0, 20))
        
        # Build appropriate UI based on mode
        if self.simplified:
            self.build_2d_ui()
        else:
            self.build_3d_ui()
    
    def build_2d_ui(self):
        """Build the UI for 2D analysis"""
        # Video input section
        video_frame = ctk.CTkFrame(self.content_frame)
        video_frame.pack(fill='x', pady=10)
        
        ctk.CTkLabel(
            video_frame,
            text="Video Input:",
            font=("Helvetica", 16, "bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        # Video input type selection (File, Webcam, Multiple files)
        input_type_frame = ctk.CTkFrame(video_frame)
        input_type_frame.pack(fill='x', padx=10, pady=5)
        
        ctk.CTkLabel(
            input_type_frame,
            text="Input Type:",
            width=100
        ).pack(side='left', padx=5)
        
        input_type_radio_frame = ctk.CTkFrame(input_type_frame, fg_color="transparent")
        input_type_radio_frame.pack(side='left', fill='x', expand=True)
        
        ctk.CTkRadioButton(
            input_type_radio_frame,
            text="Single Video File",
            variable=self.video_input_type_var,
            value='file',
            command=self.on_video_input_type_change
        ).pack(side='left', padx=10)
        
        ctk.CTkRadioButton(
            input_type_radio_frame,
            text="Webcam",
            variable=self.video_input_type_var,
            value='webcam',
            command=self.on_video_input_type_change
        ).pack(side='left', padx=10)
        
        ctk.CTkRadioButton(
            input_type_radio_frame,
            text="Multiple Videos",
            variable=self.video_input_type_var,
            value='multiple',
            command=self.on_video_input_type_change
        ).pack(side='left', padx=10)
        
        # Container for video input options (changes based on selection)
        self.video_input_container = ctk.CTkFrame(video_frame)
        self.video_input_container.pack(fill='x', padx=10, pady=5)
        
        # Initialize with single file option as default
        self.build_single_video_input()
        
        # Person details section
        self.build_person_section()
        
        # Pose model section
        self.build_pose_model_section()
        
        # Visible side selection
        side_frame = ctk.CTkFrame(self.content_frame)
        side_frame.pack(fill='x', pady=10)
        
        ctk.CTkLabel(
            side_frame,
            text="Visible Side:",
            font=("Helvetica", 16, "bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        side_options = ['auto', 'right', 'left', 'front', 'back', 'none']
        side_menu = ctk.CTkOptionMenu(
            side_frame,
            variable=self.visible_side_var,
            values=side_options,
            width=150
        )
        side_menu.pack(anchor='w', padx=30, pady=10)
        
        # Proceed button
        ctk.CTkButton(
            self.content_frame,
            text=self.app.lang_manager.get_text('proceed_pose_estimation'),
            command=self.proceed_pose_estimation,
            height=40,
            width=200,
            font=("Helvetica", 14),
            fg_color=("#4CAF50", "#2E7D32")
        ).pack(side='bottom', pady=20)
    
    def build_single_video_input(self):
        """Build the UI for single video file input"""
        # Clear existing content
        for widget in self.video_input_container.winfo_children():
            widget.destroy()
        
        # Video path display and browse button
        path_frame = ctk.CTkFrame(self.video_input_container, fg_color="transparent")
        path_frame.pack(fill='x', pady=5)
        
        self.path_entry = ctk.CTkEntry(
            path_frame,
            textvariable=self.video_input_var,
            width=400
        )
        self.path_entry.pack(side='left', padx=(0, 10))
        
        ctk.CTkButton(
            path_frame,
            text="Browse",
            command=self.browse_video,
            width=100
        ).pack(side='left')
    
    def build_webcam_input(self):
        """Build the UI for webcam input"""
        # Clear existing content
        for widget in self.video_input_container.winfo_children():
            widget.destroy()
        
        # Webcam info label
        webcam_info_frame = ctk.CTkFrame(self.video_input_container, fg_color="transparent")
        webcam_info_frame.pack(fill='x', pady=10)
        
        ctk.CTkLabel(
            webcam_info_frame,
            text="Webcam will be used when Sports2D is launched.\nNo additional configuration needed.",
            wraplength=500,
            font=("Helvetica", 14),
            text_color=("gray20", "gray90")
        ).pack(pady=10)
        
        # Set value for config
        self.video_input_var.set("webcam")
    
    def build_multiple_videos_input(self):
        """Build the UI for multiple video files input"""
        # Clear existing content
        for widget in self.video_input_container.winfo_children():
            widget.destroy()
        
        # Create a frame for the list and controls
        list_frame = ctk.CTkFrame(self.video_input_container)
        list_frame.pack(fill='x', pady=5)
        
        # Video list (scrollable)
        self.videos_list_frame = ctk.CTkScrollableFrame(list_frame, height=150)
        self.videos_list_frame.pack(fill='x', expand=True, pady=5)
        
        # Add controls
        controls_frame = ctk.CTkFrame(list_frame, fg_color="transparent")
        controls_frame.pack(fill='x', pady=5)
        
        ctk.CTkButton(
            controls_frame,
            text="Add Video",
            command=self.add_video_to_list,
            width=120
        ).pack(side='left', padx=5)
        
        ctk.CTkButton(
            controls_frame,
            text="Clear All",
            command=self.clear_videos_list,
            width=120
        ).pack(side='left', padx=5)
        
        # Update the videos list display
        self.update_videos_list_display()
    
    def update_videos_list_display(self):
        """Update the display of multiple videos list"""
        # Clear current list display
        for widget in self.videos_list_frame.winfo_children():
            widget.destroy()
        
        if not self.multiple_videos_list:
            ctk.CTkLabel(
                self.videos_list_frame,
                text="No videos added yet. Click 'Add Video' to begin.",
                text_color="gray"
            ).pack(pady=10)
            return
        
        # Add each video to the list
        for i, video_path in enumerate(self.multiple_videos_list):
            video_frame = ctk.CTkFrame(self.videos_list_frame)
            video_frame.pack(fill='x', pady=2)
            
            # Show just the filename to save space
            filename = os.path.basename(video_path)
            ctk.CTkLabel(
                video_frame,
                text=f"{i+1}. {filename}",
                width=400,
                anchor="w"
            ).pack(side='left', padx=5)
            
            # Remove button
            ctk.CTkButton(
                video_frame,
                text="✕",
                width=30,
                command=lambda idx=i: self.remove_video_from_list(idx)
            ).pack(side='right', padx=5)
    
    def add_video_to_list(self):
        """Add a video to the multiple videos list"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mpeg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.multiple_videos_list.append(file_path)
            self.update_videos_list_display()
    
    def remove_video_from_list(self, index):
        """Remove a video from the multiple videos list"""
        if 0 <= index < len(self.multiple_videos_list):
            del self.multiple_videos_list[index]
            self.update_videos_list_display()
    
    def clear_videos_list(self):
        """Clear all videos from the list"""
        self.multiple_videos_list = []
        self.update_videos_list_display()
    
    def on_video_input_type_change(self):
        """Handle change in video input type selection"""
        input_type = self.video_input_type_var.get()
        
        if input_type == 'file':
            self.build_single_video_input()
        elif input_type == 'webcam':
            self.build_webcam_input()
        elif input_type == 'multiple':
            self.build_multiple_videos_input()
    
    def build_3d_ui(self):
        """Build the UI for 3D analysis"""
        # Multiple persons section
        multiple_frame = ctk.CTkFrame(self.content_frame)
        multiple_frame.pack(fill='x', pady=10)
        
        ctk.CTkLabel(
            multiple_frame,
            text=self.app.lang_manager.get_text('multiple_persons'),
            width=150
        ).pack(side='left', padx=10, pady=10)
        
        radio_frame = ctk.CTkFrame(multiple_frame, fg_color="transparent")
        radio_frame.pack(side='left', fill='x', expand=True)
        
        ctk.CTkRadioButton(
            radio_frame,
            text=self.app.lang_manager.get_text('single_person'),
            variable=self.multiple_persons_var,
            value='single',
            command=self.on_multiple_persons_change
        ).pack(side='left', padx=10)
        
        ctk.CTkRadioButton(
            radio_frame,
            text=self.app.lang_manager.get_text('multiple_persons'),
            variable=self.multiple_persons_var,
            value='multiple',
            command=self.on_multiple_persons_change
        ).pack(side='left', padx=10)
        
        # Person details frame
        self.person_frame = ctk.CTkFrame(self.content_frame)
        self.person_frame.pack(fill='x', pady=10)
        
        # Initially show single person details
        self.build_single_person_details()
        
        # Pose model section
        self.build_pose_model_section()
        
        # Proceed button
        ctk.CTkButton(
            self.content_frame,
            text=self.app.lang_manager.get_text('proceed_pose_estimation'),
            command=self.proceed_pose_estimation,
            height=40,
            width=200,
            font=("Helvetica", 14),
            fg_color=("#4CAF50", "#2E7D32")
        ).pack(pady=20)
    
    def build_pose_model_section(self):
        """Build the pose model selection section"""
        model_frame = ctk.CTkFrame(self.content_frame)
        model_frame.pack(fill='x', pady=10)
        
        ctk.CTkLabel(
            model_frame,
            text=self.app.lang_manager.get_text('pose_model_selection'),
            font=("Helvetica", 16, "bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        # Pose model selection
        model_menu_frame = ctk.CTkFrame(model_frame)
        model_menu_frame.pack(fill='x', padx=10, pady=5)
        
        ctk.CTkLabel(
            model_menu_frame,
            text="Model:",
            width=100
        ).pack(side='left', padx=5)
        
        # Available pose models
        pose_models = [
            'Body_with_feet', 'Whole_body_wrist', 'Whole_body', 'Body',
            'Hand', 'Face', 'Animal'
        ]
        
        self.pose_model_menu = ctk.CTkOptionMenu(
            model_menu_frame,
            variable=self.pose_model_var,
            values=pose_models,
            width=200,
            command=self.on_pose_model_change
        )
        self.pose_model_menu.pack(side='left', padx=5)
        
        # Mode selection
        self.mode_frame = ctk.CTkFrame(model_frame)
        self.mode_frame.pack(fill='x', padx=10, pady=5)
        
        ctk.CTkLabel(
            self.mode_frame,
            text=self.app.lang_manager.get_text('mode'),
            width=100
        ).pack(side='left', padx=5)
        
        mode_options = ['lightweight', 'balanced', 'performance']
        self.mode_menu = ctk.CTkOptionMenu(
            self.mode_frame,
            variable=self.mode_var,
            values=mode_options,
            width=200
        )
        self.mode_menu.pack(side='left', padx=5)
        
        # Add tracking mode selection (new section)
        self.tracking_frame = ctk.CTkFrame(model_frame)
        self.tracking_frame.pack(fill='x', padx=10, pady=5)
        
        ctk.CTkLabel(
            self.tracking_frame,
            text="Tracking Mode:",
            width=100
        ).pack(side='left', padx=5)
        
        # Tracking mode options - added "deepsort" as requested
        tracking_options = ['sports2d', 'deepsort']
        self.tracking_mode_menu = ctk.CTkOptionMenu(
            self.tracking_frame,
            variable=self.tracking_mode_var,
            values=tracking_options,
            width=200
        )
        self.tracking_mode_menu.pack(side='left', padx=5)
        
        # Tracking mode info button
        ctk.CTkButton(
            self.tracking_frame,
            text="?",
            width=30,
            command=self.show_tracking_info
        ).pack(side='left', padx=5)
        
        # Video extension
        extension_frame = ctk.CTkFrame(model_frame)
        extension_frame.pack(fill='x', padx=10, pady=5)
        
        ctk.CTkLabel(
            extension_frame,
            text=self.app.lang_manager.get_text('video_extension'),
            width=150
        ).pack(side='left', padx=5)
        
        ctk.CTkEntry(
            extension_frame,
            textvariable=self.video_extension_var,
            width=100
        ).pack(side='left', padx=5)
        
        # Apply the current pose model selection
        self.on_pose_model_change(self.pose_model_var.get())
    
    def show_tracking_info(self):
        """Show info about tracking modes"""
        messagebox.showinfo(
            "Tracking Modes",
            "sports2d: Default tracking optimized for sports applications\n\n"
            "deepsort: Advanced tracking algorithm with better multi-person ID consistency"
        )
    
    def build_person_section(self):
        """Build the section for personal information in 2D mode"""
        person_frame = ctk.CTkFrame(self.content_frame)
        person_frame.pack(fill='x', pady=10)
        
        ctk.CTkLabel(
            person_frame,
            text="Participant Information:",
            font=("Helvetica", 16, "bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        # Multiple persons selection
        multiple_frame = ctk.CTkFrame(person_frame)
        multiple_frame.pack(fill='x', padx=10, pady=5)
        
        ctk.CTkLabel(
            multiple_frame,
            text=self.app.lang_manager.get_text('multiple_persons'),
            width=150
        ).pack(side='left', padx=5)
        
        radio_frame = ctk.CTkFrame(multiple_frame, fg_color="transparent")
        radio_frame.pack(side='left')
        
        ctk.CTkRadioButton(
            radio_frame,
            text=self.app.lang_manager.get_text('single_person'),
            variable=self.multiple_persons_var,
            value='single',
            command=self.on_multiple_persons_change
        ).pack(side='left', padx=10)
        
        ctk.CTkRadioButton(
            radio_frame,
            text=self.app.lang_manager.get_text('multiple_persons'),
            variable=self.multiple_persons_var,
            value='multiple',
            command=self.on_multiple_persons_change
        ).pack(side='left', padx=10)
        
        # Person details container
        self.person_frame = ctk.CTkFrame(person_frame)
        self.person_frame.pack(fill='x', padx=10, pady=5)
        
        # Initially show single person details
        self.build_single_person_details()
    
    def build_single_person_details(self):
        """Build form for single person details"""
        # Clear existing widgets
        for widget in self.person_frame.winfo_children():
            widget.destroy()
        
        # Height input
        height_frame = ctk.CTkFrame(self.person_frame)
        height_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            height_frame,
            text=self.app.lang_manager.get_text('participant_height'),
            width=150
        ).pack(side='left', padx=5)
        
        ctk.CTkEntry(
            height_frame,
            textvariable=self.participant_height_var,
            width=100
        ).pack(side='left', padx=5)
        
        # Mass input
        mass_frame = ctk.CTkFrame(self.person_frame)
        mass_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            mass_frame,
            text=self.app.lang_manager.get_text('participant_mass'),
            width=150
        ).pack(side='left', padx=5)
        
        ctk.CTkEntry(
            mass_frame,
            textvariable=self.participant_mass_var,
            width=100
        ).pack(side='left', padx=5)
    
    def build_multiple_persons_form(self):
        """Build form for multiple persons details"""
        # Clear existing widgets
        for widget in self.person_frame.winfo_children():
            widget.destroy()
        
        # Input for number of people
        num_people_frame = ctk.CTkFrame(self.person_frame)
        num_people_frame.pack(fill='x', pady=10)
        
        ctk.CTkLabel(
            num_people_frame,
            text=self.app.lang_manager.get_text('number_of_people'),
            width=150
        ).pack(side='left', padx=5)
        
        ctk.CTkEntry(
            num_people_frame,
            textvariable=self.num_people_var,
            width=100
        ).pack(side='left', padx=5)
        
        ctk.CTkButton(
            num_people_frame,
            text=self.app.lang_manager.get_text('submit_number'),
            command=self.create_people_details_inputs,
            width=100
        ).pack(side='left', padx=10)
    
    def create_people_details_inputs(self):
        """Create input fields for each person's details"""
        try:
            num_people = int(self.num_people_var.get())
            if num_people < 1:
                raise ValueError("Number of people must be positive")
        except ValueError as e:
            messagebox.showerror(
                "Error",
                f"Invalid number of people: {str(e)}"
            )
            return
        
        # Clear previous inputs except for the number of people frame
        for widget in list(self.person_frame.winfo_children())[1:]:
            widget.destroy()
        
        # Create scrollable frame for many people
        details_frame = ctk.CTkScrollableFrame(self.person_frame, height=200)
        details_frame.pack(fill='x', pady=10)
        
        # Clear previous vars
        self.people_details_vars = []
        
        # Create input fields for each person
        for i in range(num_people):
            person_frame = ctk.CTkFrame(details_frame)
            person_frame.pack(fill='x', pady=5)
            
            ctk.CTkLabel(
                person_frame,
                text=f"Person {i+1}",
                font=("Helvetica", 14, "bold")
            ).pack(anchor='w', padx=10, pady=(10, 5))
            
            # Height
            height_frame = ctk.CTkFrame(person_frame)
            height_frame.pack(fill='x', pady=2)
            
            ctk.CTkLabel(
                height_frame,
                text=self.app.lang_manager.get_text('height'),
                width=100
            ).pack(side='left', padx=5)
            
            height_var = ctk.StringVar(value="1.72")
            ctk.CTkEntry(
                height_frame,
                textvariable=height_var,
                width=100
            ).pack(side='left', padx=5)
            
            # Mass
            mass_frame = ctk.CTkFrame(person_frame)
            mass_frame.pack(fill='x', pady=2)
            
            ctk.CTkLabel(
                mass_frame,
                text=self.app.lang_manager.get_text('mass'),
                width=100
            ).pack(side='left', padx=5)
            
            mass_var = ctk.StringVar(value="70.0")
            ctk.CTkEntry(
                mass_frame,
                textvariable=mass_var,
                width=100
            ).pack(side='left', padx=5)
            
            # Store vars
            self.people_details_vars.append((height_var, mass_var))
        
        # Add submit button
        ctk.CTkButton(
            self.person_frame,
            text=self.app.lang_manager.get_text('submit'),
            command=self.submit_people_details,
            width=150,
            height=40
        ).pack(pady=10)
    
    def submit_people_details(self):
        """Process and validate people details"""
        heights = []
        masses = []
        
        try:
            for i, (height_var, mass_var) in enumerate(self.people_details_vars):
                height = float(height_var.get())
                mass = float(mass_var.get())
                
                if height <= 0 or mass <= 0:
                    raise ValueError(f"Invalid values for person {i+1}")
                
                heights.append(height)
                masses.append(mass)
            
            # Store values
            self.participant_heights = heights
            self.participant_masses = masses
            
            messagebox.showinfo(
                "Success",
                "Participant details saved successfully."
            )
            
        except ValueError as e:
            messagebox.showerror(
                "Error",
                f"Invalid input: {str(e)}"
            )
    
    def on_multiple_persons_change(self):
        """Handle change in multiple persons selection"""
        if self.multiple_persons_var.get() == 'single':
            self.build_single_person_details()
        else:
            self.build_multiple_persons_form()
    
    def on_pose_model_change(self, value):
        """Show/hide mode selection based on pose model"""
        if value in ['Body_with_feet', 'Whole_body_wrist', 'Whole_body', 'Body']:
            self.mode_frame.pack(fill='x', padx=10, pady=5)
        else:
            self.mode_frame.pack_forget()
    
    def browse_video(self):
        """Browse for video file in 2D mode"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mpeg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_input_var.set(file_path)
    
    def proceed_pose_estimation(self):
        """Handle pose estimation configuration and video input"""
        try:
            # Validate inputs
            if not self.video_extension_var.get():
                messagebox.showerror(
                    "Error",
                    "Please specify a video extension."
                )
                return
            
            # Validate person details
            if self.multiple_persons_var.get() == 'multiple':
                if not hasattr(self, 'participant_heights') or not self.participant_heights:
                    messagebox.showerror(
                        "Error",
                        "Please enter and submit participant details for multiple persons."
                    )
                    return
            else:
                try:
                    height = float(self.participant_height_var.get())
                    mass = float(self.participant_mass_var.get())
                    
                    if height <= 0 or mass <= 0:
                        raise ValueError("Height and mass must be positive.")
                except ValueError as e:
                    messagebox.showerror(
                        "Error",
                        f"Invalid height or mass: {str(e)}"
                    )
                    return
            
            # Handle 2D or 3D mode specifically
            if self.simplified:
                # 2D mode: check input type and handle accordingly
                input_type = self.video_input_type_var.get()
                
                if input_type == 'webcam':
                    # Just set the value to 'webcam' in config, no file copying needed
                    messagebox.showinfo(
                        "Webcam Setup",
                        "Webcam will be used when Sports2D is launched.\nNo additional setup needed."
                    )
                elif input_type == 'multiple':
                    # Check if multiple videos are selected
                    if not self.multiple_videos_list:
                        messagebox.showerror(
                            "Error",
                            "Please add at least one video for multiple videos mode."
                        )
                        return
                    
                    # For multiple videos, store paths but don't copy
                    # (Paths will be saved to config as a list)
                    pass
                else:  # single file
                    # Check that video is selected for single file mode
                    if not self.video_input_var.get():
                        messagebox.showerror(
                            "Error",
                            "Please select a video file."
                        )
                        return
                    
                    # Copy video to participant directory if not already there
                    dest_dir = os.path.join(self.app.participant_name)
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # Check if file needs to be copied
                    if not os.path.dirname(self.video_input_var.get()) == dest_dir:
                        # Get just the filename (preserve the same name for config)
                        filename = os.path.basename(self.video_input_var.get())
                        dest_path = os.path.join(dest_dir, filename)
                        
                        # Copy the file
                        shutil.copy(self.video_input_var.get(), dest_path)
                        
                        # Update the path to only the filename for config_demo.toml
                        self.video_input_var.set(filename)
            else:
                # 3D mode: input videos for each camera
                self.input_videos()
            
            # Update progress
            if hasattr(self.app, 'update_tab_indicator'):
                self.app.update_tab_indicator('pose_model', True)
            if hasattr(self.app, 'update_progress_bar') and hasattr(self.app, 'progress_steps'):
                progress_value = self.app.progress_steps.get('pose_model', 50)
                self.app.update_progress_bar(progress_value)
            
            # Show success message
            messagebox.showinfo(
                "Pose Model Settings",
                f"Pose model settings have been saved. Tracking mode: {self.tracking_mode_var.get()}"
            )
            
            # Move to next tab
            if hasattr(self.app, 'show_tab'):
                tab_order = list(self.app.tabs.keys())
                current_idx = tab_order.index('pose_model')
                if current_idx + 1 < len(tab_order):
                    next_tab = tab_order[current_idx + 1]
                    self.app.show_tab(next_tab)
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"An unexpected error occurred: {str(e)}"
            )
    
    def input_videos(self):
        """Input videos for 3D mode"""
        try:
            # Get number of cameras
            num_cameras = int(self.app.tabs['calibration'].num_cameras_var.get())
            
            # Define target directory
            if self.app.process_mode == 'batch':
                # For batch mode, ask which trial to import videos for
                trial_num = simpledialog.askinteger(
                    "Trial Selection",
                    f"Enter trial number (1-{self.app.num_trials}):",
                    minvalue=1,
                    maxvalue=self.app.num_trials
                )
                
                if not trial_num:
                    return False
                
                target_path = os.path.join(self.app.participant_name, f'Trial_{trial_num}', 'videos')
            else:
                # For single mode
                target_path = os.path.join(self.app.participant_name, 'videos')
            
            # Create the directory if it doesn't exist
            os.makedirs(target_path, exist_ok=True)
            
            # Check for existing videos
            existing_videos = [
                f for f in os.listdir(target_path) 
                if f.endswith(self.video_extension_var.get())
            ]
            
            if existing_videos:
                response = messagebox.askyesno(
                    "Existing Videos",
                    "Existing videos found. Do you want to replace them?"
                )
                
                if response:
                    # Delete existing videos
                    for video in existing_videos:
                        try:
                            os.remove(os.path.join(target_path, video))
                        except Exception as e:
                            messagebox.showerror(
                                "Error",
                                f"Could not remove {video}: {str(e)}"
                            )
                            return False
                else:
                    # User chose not to replace
                    return False
            
            # Input new videos
            for cam in range(1, num_cameras + 1):
                file_path = filedialog.askopenfilename(
                    title=f"Select video for Camera {cam}",
                    filetypes=[(f"Video files", f"*.{self.video_extension_var.get()}")]
                )
                
                if not file_path:
                    messagebox.showerror(
                        "Error", 
                        f"No file selected for camera {cam}"
                    )
                    return False
                
                # Copy and rename the file
                dest_filename = f"cam{cam}.{self.video_extension_var.get()}"
                dest_path = os.path.join(target_path, dest_filename)
                
                # Copy the file
                shutil.copy(file_path, dest_path)
            
            # Show completion message
            messagebox.showinfo(
                "Videos Imported",
                "All videos have been imported successfully."
            )
            
            return True
            
        except ValueError:
            messagebox.showerror(
                "Error",
                "Invalid number of cameras."
            )
            return False
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Error importing videos: {str(e)}"
            )
            return False