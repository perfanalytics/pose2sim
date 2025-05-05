import os
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox

class SynchronizationTab:
    def __init__(self, parent, app):
        """Initialize the Synchronization tab"""
        self.parent = parent
        self.app = app
        
        # Create the main frame
        self.frame = ctk.CTkFrame(parent)
        
        # Initialize variables
        self.sync_videos_var = ctk.StringVar(value='no')  # Default to 'no' (need synchronization)
        self.use_gui_var = ctk.StringVar(value='yes')     # Default to 'yes' (use GUI)
        self.keypoints_var = ctk.StringVar(value='all')
        self.approx_time_var = ctk.StringVar(value='auto')
        self.time_range_var = ctk.StringVar(value='2.0')
        self.likelihood_threshold_var = ctk.StringVar(value='0.4')
        self.filter_cutoff_var = ctk.StringVar(value='6')
        self.filter_order_var = ctk.StringVar(value='4')
        self.approx_time_entries = []
        self.approx_times = []
        
        # Build the UI
        self.build_ui()
    
    def get_title(self):
        """Return the tab title"""
        return "Synchronization"
    
    def get_settings(self):
        """Get the synchronization settings"""
        settings = {
            'synchronization': {}
        }
        
        # If skipping synchronization, disable GUI
        if self.sync_videos_var.get() == 'yes':
            settings['synchronization']['synchronization_gui'] = False
            return settings
        
        # Set GUI flag based on selection
        settings['synchronization']['synchronization_gui'] = self.use_gui_var.get() == 'yes'
        
        # If using GUI, we don't need the other settings as they can be set in the GUI
        if self.use_gui_var.get() == 'yes':
            return settings
        
        # Otherwise, add all manual synchronization settings
        # Get keypoints setting (all or specific keypoint)
        keypoints = self.keypoints_var.get()
        if keypoints == 'all':
            keypoints_setting = 'all'
        else:
            keypoints_setting = [keypoints]
        
        # Get approximate times
        if self.approx_time_var.get() == 'yes' and self.approx_time_entries:
            try:
                approx_times = [float(entry.get()) for entry in self.approx_time_entries]
            except (ValueError, TypeError):
                # Default to auto if conversion fails
                approx_times = 'auto'
        else:
            approx_times = 'auto'
        
        # Get other numeric settings with validation
        try:
            time_range = float(self.time_range_var.get())
        except ValueError:
            time_range = 2.0
            
        try:
            likelihood_threshold = float(self.likelihood_threshold_var.get())
        except ValueError:
            likelihood_threshold = 0.4
            
        try:
            filter_cutoff = int(self.filter_cutoff_var.get())
        except ValueError:
            filter_cutoff = 6
            
        try:
            filter_order = int(self.filter_order_var.get())
        except ValueError:
            filter_order = 4
        
        # Add all manual settings
        settings['synchronization'].update({
            'keypoints_to_consider': keypoints_setting,
            'approx_time_maxspeed': approx_times,
            'time_range_around_maxspeed': time_range,
            'likelihood_threshold': likelihood_threshold,
            'filter_cutoff': filter_cutoff,
            'filter_order': filter_order
        })
        
        return settings
    
    def build_ui(self):
        """Build the tab user interface"""
        # Create scrollable frame for content
        content_frame = ctk.CTkScrollableFrame(self.frame)
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title header
        ctk.CTkLabel(
            content_frame,
            text="Synchronization",
            font=("Helvetica", 24, "bold")
        ).pack(anchor='w', pady=(0, 20))
        
        # Information text
        ctk.CTkLabel(
            content_frame,
            text="Configure video synchronization settings. Videos must be synchronized for accurate 3D reconstruction.",
            font=("Helvetica", 14),
            wraplength=800,
            justify="left"
        ).pack(anchor='w', pady=(0, 20))
        
        # First decision: Skip synchronization or not
        skip_frame = ctk.CTkFrame(content_frame)
        skip_frame.pack(fill='x', pady=10)
        
        ctk.CTkLabel(
            skip_frame,
            text="Are your videos already synchronized?",
            font=("Helvetica", 14, "bold")
        ).pack(side='left', padx=10)
        
        ctk.CTkRadioButton(
            skip_frame,
            text="Yes (Skip synchronization)",
            variable=self.sync_videos_var,
            value='yes',
            command=self.update_ui_based_on_selections
        ).pack(side='left', padx=10)
        
        ctk.CTkRadioButton(
            skip_frame,
            text="No (Need synchronization)",
            variable=self.sync_videos_var,
            value='no',
            command=self.update_ui_based_on_selections
        ).pack(side='left', padx=10)
        
        # Second decision: Use GUI or not (initially hidden)
        self.gui_frame = ctk.CTkFrame(content_frame)
        self.gui_frame.pack(fill='x', pady=10)
        
        gui_title_frame = ctk.CTkFrame(self.gui_frame, fg_color="transparent")
        gui_title_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            gui_title_frame,
            text="Would you like to use the GUI for synchronization?",
            font=("Helvetica", 14, "bold")
        ).pack(side='left', padx=10)
        
        # Add "Recommended" tag with a distinct visual
        recommended_label = ctk.CTkLabel(
            gui_title_frame,
            text="✓ Recommended",
            font=("Helvetica", 12),
            text_color="#4CAF50"
        )
        recommended_label.pack(side='left', padx=10)
        
        # Radio buttons for GUI option
        gui_radio_frame = ctk.CTkFrame(self.gui_frame, fg_color="transparent")
        gui_radio_frame.pack(fill='x', pady=5, padx=10)
        
        ctk.CTkRadioButton(
            gui_radio_frame,
            text="Yes (Interactive synchronization interface)",
            variable=self.use_gui_var,
            value='yes',
            command=self.update_ui_based_on_selections
        ).pack(side='left', padx=20)
        
        ctk.CTkRadioButton(
            gui_radio_frame,
            text="No (Manual parameter configuration)",
            variable=self.use_gui_var,
            value='no',
            command=self.update_ui_based_on_selections
        ).pack(side='left', padx=20)
        
        # GUI info text
        gui_info_frame = ctk.CTkFrame(self.gui_frame, fg_color=("gray95", "gray25"))
        gui_info_frame.pack(fill='x', padx=30, pady=(0, 10))
        
        ctk.CTkLabel(
            gui_info_frame,
            text="The GUI option provides an interactive interface to visualize and manually adjust synchronization. "
                 "It's the recommended approach for achieving the best synchronization results.",
            wraplength=700,
            justify="left",
            font=("Helvetica", 12),
            text_color=("gray30", "gray80")
        ).pack(pady=10, padx=10)
        
        # Hide GUI frame initially - will be shown based on selections
        self.gui_frame.pack_forget()
        
        # Manual synchronization settings frame (initially hidden)
        self.manual_sync_frame = ctk.CTkFrame(content_frame)
        self.manual_sync_frame.pack(fill='x', pady=10)
        
        # Keypoints to consider
        keypoints_frame = ctk.CTkFrame(self.manual_sync_frame)
        keypoints_frame.pack(fill='x', pady=10, padx=10)
        
        ctk.CTkLabel(
            keypoints_frame,
            text="Select keypoints to consider for synchronization:",
            font=("Helvetica", 14)
        ).pack(side='left', padx=10)
        
        keypoints_options = ['all', 'CHip', 'RHip', 'RKnee', 'RAnkle', 'RBigToe', 'RSmallToe', 'RHeel',
                             'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel', 'Neck', 'Head',
                             'Nose', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist']
        
        self.keypoints_menu = ctk.CTkOptionMenu(
            keypoints_frame,
            variable=self.keypoints_var,
            values=keypoints_options,
            width=150
        )
        self.keypoints_menu.pack(side='left', padx=10)
        
        # Approximate time of movement
        approx_time_frame = ctk.CTkFrame(self.manual_sync_frame)
        approx_time_frame.pack(fill='x', pady=10, padx=10)
        
        ctk.CTkLabel(
            approx_time_frame,
            text="Do you want to specify approximate times of movement?",
            font=("Helvetica", 14)
        ).pack(side='left', padx=10)
        
        ctk.CTkRadioButton(
            approx_time_frame,
            text="Yes (Recommended)",
            variable=self.approx_time_var,
            value='yes',
            command=self.on_approx_time_change
        ).pack(side='left', padx=10)
        
        ctk.CTkRadioButton(
            approx_time_frame,
            text="Auto (Uses whole video)",
            variable=self.approx_time_var,
            value='auto',
            command=self.on_approx_time_change
        ).pack(side='left', padx=10)
        
        # Frame for camera-specific times (initially hidden)
        self.camera_times_frame = ctk.CTkFrame(self.manual_sync_frame)
        self.camera_times_frame.pack(fill='x', pady=10, padx=10)
        self.camera_times_frame.pack_forget()  # Hide initially
        
        # Separator
        ctk.CTkFrame(self.manual_sync_frame, height=1, fg_color="gray").pack(
            fill='x', pady=10, padx=20)
        
        # Parameters frame
        params_frame = ctk.CTkFrame(self.manual_sync_frame)
        params_frame.pack(fill='x', pady=10, padx=10)
        
        # Time range around max speed
        time_range_frame = ctk.CTkFrame(params_frame)
        time_range_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            time_range_frame,
            text="Time interval around max speed (seconds):",
            font=("Helvetica", 14),
            width=300
        ).pack(side='left', padx=10)
        
        ctk.CTkEntry(
            time_range_frame,
            textvariable=self.time_range_var,
            width=100
        ).pack(side='left', padx=10)
        
        # Likelihood threshold
        likelihood_frame = ctk.CTkFrame(params_frame)
        likelihood_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            likelihood_frame,
            text="Likelihood Threshold:",
            font=("Helvetica", 14),
            width=300
        ).pack(side='left', padx=10)
        
        ctk.CTkEntry(
            likelihood_frame,
            textvariable=self.likelihood_threshold_var,
            width=100
        ).pack(side='left', padx=10)
        
        # Filter settings
        filter_frame = ctk.CTkFrame(params_frame)
        filter_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            filter_frame,
            text="Filter Cutoff (Hz):",
            font=("Helvetica", 14),
            width=300
        ).pack(side='left', padx=10)
        
        ctk.CTkEntry(
            filter_frame,
            textvariable=self.filter_cutoff_var,
            width=100
        ).pack(side='left', padx=10)
        
        # Filter order
        filter_order_frame = ctk.CTkFrame(params_frame)
        filter_order_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            filter_order_frame,
            text="Filter Order:",
            font=("Helvetica", 14),
            width=300
        ).pack(side='left', padx=10)
        
        ctk.CTkEntry(
            filter_order_frame,
            textvariable=self.filter_order_var,
            width=100
        ).pack(side='left', padx=10)
        
        # Hide manual sync frame initially
        self.manual_sync_frame.pack_forget()
        
        # Buttons for saving settings
        self.skip_button_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        self.skip_button_frame.pack(pady=20)
        
        self.confirm_skip_button = ctk.CTkButton(
            self.skip_button_frame,
            text="Confirm Skip Synchronization",
            command=self.confirm_skip_synchronization,
            font=("Helvetica", 14),
            height=40,
            width=250
        )
        
        self.confirm_gui_button = ctk.CTkButton(
            self.skip_button_frame,
            text="Confirm GUI Synchronization",
            command=self.confirm_gui_synchronization,
            font=("Helvetica", 14),
            height=40,
            width=250,
            fg_color="#4CAF50",
            hover_color="#388E3C"
        )
        
        self.save_manual_button = ctk.CTkButton(
            self.skip_button_frame,
            text="Save Manual Synchronization Settings",
            command=self.save_manual_settings,
            font=("Helvetica", 14),
            height=40,
            width=250
        )
        
        # Status label for feedback
        self.status_label = ctk.CTkLabel(
            content_frame,
            text="",
            font=("Helvetica", 12),
            text_color="gray"
        )
        self.status_label.pack(pady=10)
        
        # Initialize UI based on current settings
        self.update_ui_based_on_selections()
    
    def update_ui_based_on_selections(self):
        """Update which UI elements are shown based on current selections"""
        # Clear all buttons first
        for widget in self.skip_button_frame.winfo_children():
            widget.pack_forget()
            
        # If skipping synchronization (videos already synced)
        if self.sync_videos_var.get() == 'yes':
            # Hide GUI and manual frames
            self.gui_frame.pack_forget()
            self.manual_sync_frame.pack_forget()
            
            # Show only the skip confirmation button
            self.confirm_skip_button.pack(pady=10)
            
            # Update status
            self.status_label.configure(
                text="Videos will be treated as already synchronized. No synchronization will be performed.",
                text_color="blue"
            )
            
        # If need synchronization (videos not synced)
        else:
            # Show GUI choice frame
            self.gui_frame.pack(fill='x', pady=10)
            
            # If using GUI
            if self.use_gui_var.get() == 'yes':
                # Hide manual sync frame
                self.manual_sync_frame.pack_forget()
                
                # Show GUI confirmation button
                self.confirm_gui_button.pack(pady=10)
                
                # Update status
                self.status_label.configure(
                    text="You will use the interactive GUI for synchronization during processing.",
                    text_color="blue"
                )
                
            # If not using GUI
            else:
                # Show manual sync frame
                self.manual_sync_frame.pack(fill='x', pady=10)
                
                # Update camera times frame if needed
                if self.approx_time_var.get() == 'yes':
                    self.setup_camera_times_input()
                    self.camera_times_frame.pack(fill='x', pady=10, padx=10)
                else:
                    self.camera_times_frame.pack_forget()
                
                # Show save manual settings button
                self.save_manual_button.pack(pady=10)
                
                # Update status
                self.status_label.configure(
                    text="Configure manual synchronization parameters above.",
                    text_color="blue"
                )
    
    def on_approx_time_change(self):
        """Handle changes to the approximate time option"""
        # Update UI
        self.update_ui_based_on_selections()
    
    def setup_camera_times_input(self):
        """Create input fields for camera-specific times"""
        # Clear existing widgets
        for widget in self.camera_times_frame.winfo_children():
            widget.destroy()
        
        # Instructions
        ctk.CTkLabel(
            self.camera_times_frame,
            text="Enter approximate times (in seconds) of sync movement for each camera:",
            font=("Helvetica", 14)
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        # Create scrollable frame for camera inputs (if many cameras)
        times_scroll_frame = ctk.CTkScrollableFrame(
            self.camera_times_frame,
            width=700,
            height=200
        )
        times_scroll_frame.pack(fill='x', pady=5)
        
        # Get number of cameras
        try:
            num_cameras = int(self.app.tabs['calibration'].num_cameras_var.get())
        except (AttributeError, ValueError):
            # Default to 2 if can't get from calibration tab
            num_cameras = 2
        
        # Reset time entries list
        self.approx_time_entries = []
        
        # Create entry for each camera
        for cam in range(1, num_cameras + 1):
            # Frame for this camera
            cam_frame = ctk.CTkFrame(times_scroll_frame)
            cam_frame.pack(fill='x', pady=2)
            
            # Label
            ctk.CTkLabel(
                cam_frame,
                text=f"Camera {cam}:",
                width=100
            ).pack(side='left', padx=10)
            
            # Entry field
            time_var = ctk.StringVar(value="0.0")
            entry = ctk.CTkEntry(
                cam_frame,
                textvariable=time_var,
                width=100
            )
            entry.pack(side='left', padx=10)
            
            # Add to entry list
            self.approx_time_entries.append(entry)
        
        # Help text
        ctk.CTkLabel(
            self.camera_times_frame,
            text="Tip: Enter the time (in seconds) when a clear movement is visible in each camera.",
            font=("Helvetica", 12),
            text_color="gray"
        ).pack(anchor='w', padx=10, pady=5)
    
    def confirm_skip_synchronization(self):
        """Handle confirmation when skipping synchronization"""
        # Update status
        self.status_label.configure(
            text="Synchronization will be skipped. Videos will be treated as already synchronized.",
            text_color="green"
        )
        
        # Update progress
        if hasattr(self.app, 'progress_steps') and 'synchronization' in self.app.progress_steps:
            progress_value = self.app.progress_steps['synchronization']
        else:
            progress_value = 70  # Default value
        
        self.app.update_progress_bar(progress_value)
        
        # Update tab indicator
        self.app.update_tab_indicator('synchronization', True)
        
        # Disable skip button
        self.confirm_skip_button.configure(state="disabled")
        
        # Show success message
        messagebox.showinfo(
            "Synchronization Skipped",
            "Synchronization will be skipped. Videos will be treated as already synchronized."
        )
        
        # Automatically move to next tab if available
        if hasattr(self.app, 'show_tab'):
            tab_order = list(self.app.tabs.keys())
            current_idx = tab_order.index('synchronization')
            if current_idx + 1 < len(tab_order):
                next_tab = tab_order[current_idx + 1]
                self.app.show_tab(next_tab)
    
    def confirm_gui_synchronization(self):
        """Handle confirmation when using GUI for synchronization"""
        # Update status
        self.status_label.configure(
            text="GUI synchronization mode enabled. You will use the interactive interface during processing.",
            text_color="green"
        )
        
        # Update progress
        if hasattr(self.app, 'progress_steps') and 'synchronization' in self.app.progress_steps:
            progress_value = self.app.progress_steps['synchronization']
        else:
            progress_value = 70  # Default value
        
        self.app.update_progress_bar(progress_value)
        
        # Update tab indicator
        self.app.update_tab_indicator('synchronization', True)
        
        # Disable GUI button
        self.confirm_gui_button.configure(state="disabled")
        
        # Show success message
        messagebox.showinfo(
            "GUI Synchronization Enabled",
            "Interactive GUI synchronization will be used during processing. This is the recommended approach."
        )
        
        # Automatically move to next tab if available
        if hasattr(self.app, 'show_tab'):
            tab_order = list(self.app.tabs.keys())
            current_idx = tab_order.index('synchronization')
            if current_idx + 1 < len(tab_order):
                next_tab = tab_order[current_idx + 1]
                self.app.show_tab(next_tab)
    
    def save_manual_settings(self):
        """Save manual synchronization settings"""
        try:
            # Validate inputs
            if self.approx_time_var.get() == 'yes':
                # Validate time entries
                for i, entry in enumerate(self.approx_time_entries, 1):
                    try:
                        time_value = float(entry.get())
                        if time_value < 0:
                            messagebox.showerror(
                                "Invalid Input",
                                f"Camera {i} time must be a positive number."
                            )
                            return
                    except ValueError:
                        messagebox.showerror(
                            "Invalid Input",
                            f"Camera {i} time must be a number."
                        )
                        return
            
            # Get other float values
            try:
                time_range = float(self.time_range_var.get())
                if time_range <= 0:
                    messagebox.showerror(
                        "Invalid Input",
                        "Time range must be a positive number."
                    )
                    return
            except ValueError:
                messagebox.showerror(
                    "Invalid Input",
                    "Time range must be a number."
                )
                return
            
            try:
                likelihood = float(self.likelihood_threshold_var.get())
                if not 0 <= likelihood <= 1:
                    messagebox.showerror(
                        "Invalid Input",
                        "Likelihood threshold must be between 0 and 1."
                    )
                    return
            except ValueError:
                messagebox.showerror(
                    "Invalid Input",
                    "Likelihood threshold must be a number."
                )
                return
            
            # Get integer values
            try:
                filter_cutoff = int(self.filter_cutoff_var.get())
                if filter_cutoff <= 0:
                    messagebox.showerror(
                        "Invalid Input",
                        "Filter cutoff must be a positive integer."
                    )
                    return
            except ValueError:
                messagebox.showerror(
                    "Invalid Input",
                    "Filter cutoff must be an integer."
                )
                return
            
            try:
                filter_order = int(self.filter_order_var.get())
                if filter_order <= 0:
                    messagebox.showerror(
                        "Invalid Input",
                        "Filter order must be a positive integer."
                    )
                    return
            except ValueError:
                messagebox.showerror(
                    "Invalid Input",
                    "Filter order must be an integer."
                )
                return
            
            # Update status
            self.status_label.configure(
                text="Manual synchronization settings saved successfully. GUI is disabled.",
                text_color="green"
            )
            
            # Update progress
            if hasattr(self.app, 'progress_steps') and 'synchronization' in self.app.progress_steps:
                progress_value = self.app.progress_steps['synchronization']
            else:
                progress_value = 70  # Default value
            
            self.app.update_progress_bar(progress_value)
            
            # Update tab indicator
            self.app.update_tab_indicator('synchronization', True)
            
            # Disable inputs after saving
            self.disable_all_widgets(self.manual_sync_frame)
            self.save_manual_button.configure(state="disabled")
            
            # Show success message
            messagebox.showinfo(
                "Settings Saved",
                "Manual synchronization settings have been saved successfully. GUI mode is disabled."
            )
            
            # Automatically move to next tab if available
            if hasattr(self.app, 'show_tab'):
                tab_order = list(self.app.tabs.keys())
                current_idx = tab_order.index('synchronization')
                if current_idx + 1 < len(tab_order):
                    next_tab = tab_order[current_idx + 1]
                    self.app.show_tab(next_tab)
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"An error occurred while saving settings: {str(e)}"
            )
    
    def disable_all_widgets(self, parent):
        """Recursively disable all input widgets in a parent widget"""
        for child in parent.winfo_children():
            if isinstance(child, (ctk.CTkEntry, ctk.CTkRadioButton, ctk.CTkOptionMenu)):
                child.configure(state="disabled")
            if hasattr(child, 'winfo_children') and callable(child.winfo_children):
                self.disable_all_widgets(child)