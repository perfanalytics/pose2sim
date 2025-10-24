from pathlib import Path
import cv2
import customtkinter as ctk
from tkinter import messagebox
import threading
import subprocess
from PIL import Image
from customtkinter import CTkImage

class PrepareVideoTab:
    def __init__(self, parent, app):
        """Initialize the Prepare Video tab"""
        self.parent = parent
        self.app = app
        
        # Create the main frame
        self.frame = ctk.CTkFrame(parent)
        
        # Initialize state variables
        self.editing_mode_var = ctk.StringVar(value='simple')
        self.only_checkerboard_var = ctk.StringVar(value='yes')
        self.time_interval_var = ctk.StringVar(value='1')
        self.extrinsic_format_var = ctk.StringVar(value='png')
        self.change_intrinsics_extension = False
        self.current_camera_index = 0
        self.camera_image_list = []
        self.image_vars = []
        
        # Build the tab UI
        self.build_ui()
    
    def get_settings(self):
        """Get the prepare video settings"""
        settings = {}
        # No specific settings to return for the prepare video tab
        # as these are handled directly in the calibration settings
        return settings
    
    def build_ui(self):
        """Build the tab user interface"""
        # Create a scrollable frame for content
        content_frame = ctk.CTkScrollableFrame(self.frame)
        content_frame.pack(fill='both', expand=True, padx=0, pady=0)

        # Tab title
        self.title_label = ctk.CTkLabel(
            content_frame,
            text="Prepare Video",
            font=("Helvetica", 24, "bold")
        )
        self.title_label.pack(pady=(0, 20))

        # Editing mode selection - using a card-style UI
        mode_frame = ctk.CTkFrame(content_frame)
        mode_frame.pack(fill='x', pady=15)
        
        ctk.CTkLabel(
            mode_frame,
            text="Select Editing Mode:",
            font=("Helvetica", 18, "bold")
        ).pack(anchor="w", padx=15, pady=(10, 15))
        
        # Mode selection buttons in a horizontal layout
        buttons_frame = ctk.CTkFrame(mode_frame, fg_color="transparent")
        buttons_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        # Simple mode button (as a card)
        simple_card = ctk.CTkFrame(buttons_frame)
        simple_card.pack(side='left', padx=10, pady=5, fill='x', expand=True)
        
        self.simple_mode_btn = ctk.CTkButton(
            simple_card,
            text="Simple Mode",
            command=lambda: self.set_editing_mode('simple'),
            width=150,
            height=40,
            font=("Helvetica", 14),
            fg_color=("#3a7ebf", "#1f538d")  # Default selected color
        )
        self.simple_mode_btn.pack(pady=10, padx=10, fill='x')
        
        ctk.CTkLabel(
            simple_card,
            text="Basic extraction and processing",
            font=("Helvetica", 12),
            text_color="gray"
        ).pack(pady=(0, 10), padx=10)
        
        # Advanced mode button (as a card)
        advanced_card = ctk.CTkFrame(buttons_frame)
        advanced_card.pack(side='left', padx=10, pady=5, fill='x', expand=True)
        
        self.advanced_mode_btn = ctk.CTkButton(
            advanced_card,
            text="Advanced Editing",
            command=lambda: self.set_editing_mode('advanced'),
            width=150,
            height=40,
            font=("Helvetica", 14)
        )
        self.advanced_mode_btn.pack(pady=10, padx=10, fill='x')
        
        ctk.CTkLabel(
            advanced_card,
            text="Full-featured video editing tools",
            font=("Helvetica", 12),
            text_color="gray"
        ).pack(pady=(0, 10), padx=10)
        
        # Divider
        divider = ctk.CTkFrame(content_frame, height=2, fg_color="gray75")
        divider.pack(fill='x', pady=15)
        
        # Simple mode frame
        self.simple_mode_frame = ctk.CTkFrame(content_frame)
        self.simple_mode_frame.pack(fill='x', pady=10)
        
        # Checkerboard-only option frame
        checkerboard_frame = ctk.CTkFrame(self.simple_mode_frame)
        checkerboard_frame.pack(fill='x', pady=10, padx=10)
        
        ctk.CTkLabel(
            checkerboard_frame,
            text="Do your videos contain only checkerboard images?",
            font=("Helvetica", 14, "bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        radio_frame = ctk.CTkFrame(checkerboard_frame, fg_color="transparent")
        radio_frame.pack(fill='x', padx=10, pady=5)
        
        ctk.CTkRadioButton(
            radio_frame,
            text="Yes",
            variable=self.only_checkerboard_var,
            value='yes',
            command=self.on_only_checkerboard_change
        ).pack(side='left', padx=20)
        
        ctk.CTkRadioButton(
            radio_frame,
            text="No",
            variable=self.only_checkerboard_var,
            value='no',
            command=self.on_only_checkerboard_change
        ).pack(side='left', padx=20)
        
        # Frame for time interval input (initially hidden)
        self.time_extraction_frame = ctk.CTkFrame(self.simple_mode_frame)
        
        ctk.CTkLabel(
            self.time_extraction_frame,
            text="Enter time interval in seconds for image extraction:",
            font=("Helvetica", 14, "bold")
        ).pack(anchor='w', padx=15, pady=(10, 5))
        
        time_input_frame = ctk.CTkFrame(self.time_extraction_frame, fg_color="transparent")
        time_input_frame.pack(fill='x', padx=15, pady=5)
        
        ctk.CTkEntry(
            time_input_frame,
            textvariable=self.time_interval_var,
            width=100
        ).pack(side='left', padx=5)
        
        ctk.CTkLabel(
            time_input_frame,
            text="seconds",
            font=("Helvetica", 12)
        ).pack(side='left', padx=5)
        
        # Extrinsic Format Frame
        format_frame = ctk.CTkFrame(self.simple_mode_frame)
        format_frame.pack(fill='x', pady=10, padx=10)
        
        ctk.CTkLabel(
            format_frame,
            text="Enter the image format (e.g., png, jpg):",
            font=("Helvetica", 14, "bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        format_input_frame = ctk.CTkFrame(format_frame, fg_color="transparent")
        format_input_frame.pack(fill='x', padx=10, pady=5)
        
        ctk.CTkEntry(
            format_input_frame,
            textvariable=self.extrinsic_format_var,
            width=100
        ).pack(side='left', padx=5)
        
        # Confirm button for "Yes" option
        self.confirm_button = ctk.CTkButton(
            self.simple_mode_frame,
            text="Confirm",
            command=self.confirm_checkerboard_only,
            width=200,
            height=40,
            font=("Helvetica", 14),
            fg_color=("#4CAF50", "#2E7D32")
        )
        self.confirm_button.pack(pady=20, side='bottom')
        
        # Proceed button for "No" option (initially hidden)
        self.proceed_button = ctk.CTkButton(
            self.simple_mode_frame,
            text="Proceed with Video Preparation",
            command=self.proceed_prepare_video,
            width=200,
            height=40,
            font=("Helvetica", 14),
            fg_color=("#4CAF50", "#2E7D32")
        )
        
        # Advanced mode frame (initially hidden)
        self.advanced_mode_frame = ctk.CTkFrame(content_frame)
        
        # Advanced mode content - with clear black text as requested
        advanced_title = ctk.CTkLabel(
            self.advanced_mode_frame,
            text="Advanced Video Editing",
            font=("Helvetica", 22, "bold"),
            text_color="black"  # Explicitly setting to black for clarity
        )
        advanced_title.pack(pady=(20, 5))
        
        # Divider below title
        title_divider = ctk.CTkFrame(self.advanced_mode_frame, height=2, fg_color="gray75")
        title_divider.pack(fill='x', pady=10, padx=40)
        
        # Description with improved visibility
        description_frame = ctk.CTkFrame(self.advanced_mode_frame, fg_color=("gray95", "gray20"))
        description_frame.pack(fill='x', padx=30, pady=15)
        
        ctk.CTkLabel(
            description_frame,
            text="Use this mode to run the external blur.py editor for advanced video processing.",
            wraplength=600,
            font=("Helvetica", 14),
            text_color="black"  # Explicitly setting to black for clarity
        ).pack(pady=15, padx=20)
        
        # Button frame
        button_frame = ctk.CTkFrame(self.advanced_mode_frame, fg_color="transparent")
        button_frame.pack(pady=20)
        
        # Launch button for blur.py
        self.launch_editor_btn = ctk.CTkButton(
            button_frame,
            text="Launch Video Editor",
            command=self.launch_external_editor,
            width=200,
            height=45,
            font=("Helvetica", 14),
            fg_color=("#4CAF50", "#2E7D32")  # Green color
        )
        self.launch_editor_btn.pack(side='left', padx=10)
        
        # Done editing button
        self.done_editing_btn = ctk.CTkButton(
            button_frame,
            text="Done Editing",
            command=self.complete_advanced_editing,
            width=200,
            height=45,
            font=("Helvetica", 14),
            state="disabled",  # Initially disabled
            fg_color=("#FF9500", "#FF7000")  # Orange color
        )
        self.done_editing_btn.pack(side='left', padx=10)
        
        # Status label for feedback (common to both modes)
        self.status_frame = ctk.CTkFrame(content_frame, fg_color=("gray90", "gray25"), corner_radius=8)
        self.status_frame.pack(fill='x', pady=15, padx=10)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Select an editing mode to begin",
            font=("Helvetica", 13),
            text_color=("gray30", "gray80")
        )
        self.status_label.pack(pady=10, padx=10)
        
        # Show/hide elements based on initial editing mode
        self.set_editing_mode('simple')
    
    def set_editing_mode(self, mode):
        """Switch between simple and advanced editing modes"""
        self.editing_mode_var.set(mode)
        
        # Update button colors
        if mode == 'simple':
            self.simple_mode_btn.configure(text_color='white', fg_color=("#3a7ebf", "#1f538d"))
            self.advanced_mode_btn.configure(text_color="grey20")
            self.simple_mode_frame.pack(fill='x', pady=10)
            self.advanced_mode_frame.pack_forget()
            self.update_status("Simple mode: Configure video extraction settings", "blue")
        else:  # advanced
            self.simple_mode_btn.configure(text_color="grey20")
            self.advanced_mode_btn.configure(text_color='white', fg_color=("#3a7ebf", "#1f538d"))
            self.simple_mode_frame.pack_forget()
            self.advanced_mode_frame.pack(fill='x', pady=10)
            self.update_status("Advanced mode: Use external editor for video processing", "blue")
        
        # Apply current checkerboard setting in simple mode
        if mode == 'simple':
            self.on_only_checkerboard_change()
    
    def on_only_checkerboard_change(self):
        """Handle changes to the checkerboard-only option"""
        if self.only_checkerboard_var.get() == 'no':
            self.time_extraction_frame.pack(fill='x', pady=10, after=self.confirm_button)
            self.confirm_button.pack_forget()
            self.proceed_button.pack(pady=20)
        else:
            self.time_extraction_frame.pack_forget()
            self.proceed_button.pack_forget()
            self.confirm_button.pack(pady=20)
    
    def confirm_checkerboard_only(self):
        """Handle confirmation when 'Yes' is selected for checkerboard-only option"""
        # Keep existing extension for both intrinsics and extrinsics
        self.change_intrinsics_extension = False
        
        # Update status
        self.update_status("Prepare video step completed. Checkerboard videos will be used directly.", "green")
        
        # Update progress bar (use existing method)
        if hasattr(self.app, 'progress_steps') and 'prepare_video' in self.app.progress_steps:
            progress_value = self.app.progress_steps['prepare_video']
        else:
            progress_value = 30  # Default value for prepare_video step
            
        self.app.update_progress_bar(progress_value)
        
        # Update tab indicator
        self.app.update_tab_indicator('prepare_video', True)
        
        # Disable inputs
        for widget in self.frame.winfo_descendants():
            if isinstance(widget, (ctk.CTkEntry, ctk.CTkRadioButton)):
                widget.configure(state="disabled")
        
        self.confirm_button.configure(state="disabled")
        
        # Show success message
        messagebox.showinfo("Complete", "Prepare video step completed. You can proceed to the next tab.")
        
        # Automatically switch to the next tab if available
        if hasattr(self.app, 'show_tab'):
            tab_order = list(self.app.tabs.keys())
            current_idx = tab_order.index('prepare_video')
            if current_idx + 1 < len(tab_order):
                next_tab = tab_order[current_idx + 1]
                self.app.show_tab(next_tab)
    
    def proceed_prepare_video(self):
        """Handle video preparation when 'No' is selected"""
        try:
            time_interval = float(self.time_interval_var.get())
            if time_interval <= 0:
                raise ValueError("Time interval must be a positive number")
            
            # Set flag to change intrinsics extension to png
            self.change_intrinsics_extension = True
            
            # Update status
            self.update_status("Processing videos... Please wait.", "blue")
            
            # Disable the Proceed button to prevent multiple clicks
            self.proceed_button.configure(state='disabled')
            
            # Start extraction in a separate thread
            extraction_thread = threading.Thread(target=lambda: self.extract_frames(time_interval))
            extraction_thread.start()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid time interval: {str(e)}")
            self.update_status("Error: Please enter a valid time interval.", "red")
            self.proceed_button.configure(state='normal')
    
    def launch_external_editor(self):
        """Launch the external blur.py editor"""
        try:
            # Path to the blur.py script (in the same directory as the app)
            script_path = Path(__file__).parent.parent / "blur.py"
            
            # Check if the file exists
            if not script_path.exists():
                self.update_status("Error: blur.py not found in the application directory", "red")
                return
            
            # Update status
            self.update_status("Launching external video editor...", "blue")
            
            # Disable the launch button
            self.launch_editor_btn.configure(state="disabled")
            
            # Launch the script in a separate process
            process = subprocess.Popen(["python", script_path])
            
            # Enable the done button
            self.done_editing_btn.configure(state="normal")
            
            # Update status
            self.update_status("External editor launched. Click 'Done Editing' when finished.", "orange")
            
        except Exception as e:
            self.update_status(f"Error launching editor: {str(e)}", "red")
            self.launch_editor_btn.configure(state="normal")
    
    def complete_advanced_editing(self):
        """Complete the advanced editing process"""
        # Update status
        self.update_status("Advanced editing completed successfully.", "green")
        
        # Update progress
        if hasattr(self.app, 'progress_steps') and 'prepare_video' in self.app.progress_steps:
            progress_value = self.app.progress_steps['prepare_video']
        else:
            progress_value = 30  # Default value for prepare_video step
            
        self.app.update_progress_bar(progress_value)
        
        # Update tab indicator
        self.app.update_tab_indicator('prepare_video', True)
        
        # Disable buttons
        self.done_editing_btn.configure(state="disabled")
        self.launch_editor_btn.configure(state="disabled")
        
        # Show success message
        messagebox.showinfo("Complete", "Advanced video editing completed. You can proceed to the next tab.")
        
        # Automatically switch to the next tab if available
        if hasattr(self.app, 'show_tab'):
            tab_order = list(self.app.tabs.keys())
            current_idx = tab_order.index('prepare_video')
            if current_idx + 1 < len(tab_order):
                next_tab = tab_order[current_idx + 1]
                self.app.show_tab(next_tab)
    
    def extract_frames(self, time_interval):
        """Extract frames from videos at given time intervals"""
        # Determine the base path based on app mode
        base_path = Path(self.app.participant_name) / 'calibration' / 'intrinsics'
        
        if not base_path.exists():
            self.update_status(f"Error: Directory '{base_path}' does not exist.", "red")
            self.proceed_button.configure(state='normal')
            return
        
        video_extensions = ('.mp4', '.avi', '.mov', '.mpeg')
        extracted_images = []
        
        # Collect all video files
        video_files = [file for file in base_path.rglob('*') if file.suffix.lower() in video_extensions]
        
        total_videos = len(video_files)
        
        if not video_files:
            self.update_status("Warning: No video files found.", "orange")
            
            # Still mark as complete using the app's method
            if hasattr(self.app, 'progress_steps') and 'prepare_video' in self.app.progress_steps:
                progress_value = self.app.progress_steps['prepare_video']
            else:
                progress_value = 30
            
            self.app.update_progress_bar(progress_value)
            self.app.update_tab_indicator('prepare_video', True)
            
            self.proceed_button.configure(state='normal')
            return
        
        try:
            self.update_status(f"Processing {total_videos} videos...", "blue")
            
            for idx, video_path in enumerate(video_files):
                video_dir = video_path.parent
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    self.update_status(f"Error: Failed to open video: {video_path}", "red")
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30  # Default to 30 fps if detection fails
                
                interval_frames = int(fps * time_interval)
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % interval_frames == 0:
                        image_name = f"{Path(video_path).stem}_frame{frame_count}.png"
                        save_path = video_dir  / image_name
                        cv2.imwrite(save_path, frame)
                        extracted_images.append(save_path)
                    
                    frame_count += 1
                
                cap.release()
                
                # Update progress for this video (15-30% range for extraction)
                progress = 15 + (15 * (idx + 1) / total_videos)
                self.app.update_progress_bar(int(progress))
                
                # Update status
                self.update_status(f"Processed {idx+1}/{total_videos} videos...", "blue")
            
            # If images were extracted, show the review interface
            if extracted_images:
                self.sort_images_by_camera(extracted_images)
            else:
                self.update_status("Process completed. No frames were extracted.", "green")
                
                # Complete the prepare_video step
                if hasattr(self.app, 'progress_steps') and 'prepare_video' in self.app.progress_steps:
                    progress_value = self.app.progress_steps['prepare_video']
                else:
                    progress_value = 30
                    
                self.app.update_progress_bar(progress_value)
                self.app.update_tab_indicator('prepare_video', True)
        
        except Exception as e:
            self.update_status(f"Error during extraction: {str(e)}", "red")
            self.proceed_button.configure(state='normal')
    
    def update_status(self, message, color="black"):
        """Update the status label with a message and color"""
        # Schedule UI update on the main thread
        self.frame.after(0, lambda: self.status_label.configure(text=message, text_color=color))
    
    def sort_images_by_camera(self, image_paths):
        """Sort extracted images by camera directory"""
        images_by_camera = {}
        
        for img_path in image_paths:
            camera_dir = Path(img_path).parent.name
            if camera_dir not in images_by_camera:
                images_by_camera[camera_dir] = []
            images_by_camera[camera_dir].append(img_path)
        
        self.camera_image_list = list(images_by_camera.items())
        self.current_camera_index = 0
        
        if self.camera_image_list:
            camera_dir, imgs = self.camera_image_list[self.current_camera_index]
            self.review_camera_images(camera_dir, imgs)
        else:
            self.update_status("No images to review.", "orange")
            
            # Complete the prepare_video step
            if hasattr(self.app, 'progress_steps') and 'prepare_video' in self.app.progress_steps:
                progress_value = self.app.progress_steps['prepare_video']
            else:
                progress_value = 30
                
            self.app.update_progress_bar(progress_value)
            self.app.update_tab_indicator('prepare_video', True)
            
            self.proceed_button.configure(state='normal')
    
    def review_camera_images(self, camera_dir, image_paths):
        """Create a review window for a specific camera's images"""
        # Create a new toplevel window for reviewing images
        review_window = ctk.CTkToplevel(self.frame)
        review_window.title(f"Review Images - {camera_dir}")
        review_window.geometry("900x700")
        review_window.grab_set()  # Make modal
        
        # Header frame
        header_frame = ctk.CTkFrame(review_window)
        header_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            header_frame,
            text=f"Review Images for {camera_dir}",
            font=("Helvetica", 18, "bold")
        ).pack(side="left", padx=10)
        
        ctk.CTkLabel(
            header_frame,
            text=f"Camera {self.current_camera_index + 1} of {len(self.camera_image_list)}",
            font=("Helvetica", 14)
        ).pack(side="right", padx=10)
        
        # Create scrollable frame for images
        scroll_frame = ctk.CTkScrollableFrame(review_window)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # List to hold image vars for this camera
        self.image_vars = []
        
        # Organize images in a grid (4 columns)
        num_columns = 4
        row, col = 0, 0
        
        for idx, img_path in enumerate(image_paths):
            # Create frame for this image
            img_frame = ctk.CTkFrame(scroll_frame)
            img_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            try:
                # Load and display the image
                img = Image.open(img_path)
                img.thumbnail((200, 150))  # Resize for thumbnail display
                
                ctk_img = CTkImage(light_image=img, dark_image=img, size=(200, 150))
                
                img_label = ctk.CTkLabel(img_frame, image=ctk_img, text="")
                img_label.image = ctk_img  # Keep a reference
                img_label.pack(padx=5, pady=5)
                
                # Add filename below image
                ctk.CTkLabel(
                    img_frame,
                    text=img_path.name,
                    font=("Helvetica", 10),
                    wraplength=200
                ).pack(pady=(0, 5))
                
                # Checkbox to keep this image
                var = ctk.BooleanVar(value=True)  # Default to keeping images
                check = ctk.CTkCheckBox(img_frame, text="Keep", variable=var)
                check.pack(pady=5)
                
                # Store reference to this image
                self.image_vars.append({'var': var, 'path': img_path})
                
                # Update grid position
                col += 1
                if col >= num_columns:
                    col = 0
                    row += 1
                
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        # Button frame
        button_frame = ctk.CTkFrame(review_window)
        button_frame.pack(fill="x", padx=20, pady=10)
        
        # Function to process this camera and move to next
        def process_camera():
            # Handle image deletion
            to_delete = [img['path'] for img in self.image_vars if not img['var'].get()]
            
            for img_path in to_delete:
                try:
                    img_path.unlink()  # Delete the image file
                    print(f"Deleted {img_path}")
                except Exception as e:
                    print(f"Failed to delete {img_path}: {e}")
            
            # Close the review window
            review_window.destroy()
            
            # Move to next camera
            self.current_camera_index += 1
            if self.current_camera_index < len(self.camera_image_list):
                next_camera, next_images = self.camera_image_list[self.current_camera_index]
                self.review_camera_images(next_camera, next_images)
            else:
                # All cameras processed
                self.update_status("Image review completed. All cameras processed.", "green")
                
                # Complete the prepare_video step
                if hasattr(self.app, 'progress_steps') and 'prepare_video' in self.app.progress_steps:
                    progress_value = self.app.progress_steps['prepare_video']
                else:
                    progress_value = 30
                    
                self.app.update_progress_bar(progress_value)
                self.app.update_tab_indicator('prepare_video', True)
                
                # Show final confirmation
                messagebox.showinfo(
                    "Processing Complete",
                    "All camera images have been processed successfully."
                )
                
                # Automatically move to next tab
                if hasattr(self.app, 'show_tab'):
                    tab_order = list(self.app.tabs.keys())
                    current_idx = tab_order.index('prepare_video')
                    if current_idx + 1 < len(tab_order):
                        next_tab = tab_order[current_idx + 1]
                        self.app.show_tab(next_tab)
        
        # Add buttons
        ctk.CTkButton(
            button_frame,
            text="Save and Continue",
            command=process_camera,
            width=150,
            height=35,
            font=("Helvetica", 14)
        ).pack(side="right", padx=10)
        
        # Select/Deselect All buttons
        def select_all():
            for item in self.image_vars:
                item['var'].set(True)
        
        def deselect_all():
            for item in self.image_vars:
                item['var'].set(False)
        
        ctk.CTkButton(
            button_frame,
            text="Select All",
            command=select_all,
            width=100,
            height=35
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Deselect All",
            command=deselect_all,
            width=100,
            height=35
        ).pack(side="left", padx=10)