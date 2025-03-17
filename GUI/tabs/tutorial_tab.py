import os
import sys
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
import subprocess
import threading
import cv2
from PIL import Image, ImageTk

class TutorialTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        # Create main frame
        self.frame = ctk.CTkFrame(parent)
        
        # Initialize variables
        self.current_step = 0
        self.total_steps = 4  # Number of tutorial steps
        self.marker_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tutorial_completed")
        
        # Video player variables
        self.video_path = None
        self.video_cap = None
        self.playing = False
        self.current_frame = 0
        self.play_after_id = None
        
        # Dependency check results
        self.dependencies = {
            "anaconda": {"installed": False, "name": "Anaconda"},
            "path": {"installed": False, "name": "Anaconda Path"},
            "pose2sim": {"installed": False, "name": "Pose2Sim"},
            "opensim": {"installed": False, "name": "OpenSim"},
            "pytorch": {"installed": False, "name": "PyTorch"},
            "onnxruntime-gpu": {"installed": False, "name": "ONNX Runtime GPU"}
        }
        
        # Build the UI
        self.build_ui()
        
        # Check for tutorial marker file
        self.check_tutorial_status()
        
        # Start dependency check in background thread
        threading.Thread(target=self.check_dependencies, daemon=True).start()
    
    def get_title(self):
        """Return the tab title"""
        return "Tutorial"
    
    def get_settings(self):
        """Get the tutorial settings"""
        return {}  # This tab doesn't add settings to the config file
    
    def build_ui(self):
        """Build the tutorial UI"""
        # Create a scrollable content frame
        self.content_frame = ctk.CTkScrollableFrame(self.frame)
        self.content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.content_frame,
            text="Welcome to Pose2Sim Tutorial",
            font=("Helvetica", 24, "bold")
        )
        self.title_label.pack(pady=(0, 20))
        
        # Video player frame
        self.video_frame = ctk.CTkFrame(self.content_frame)
        self.video_frame.pack(fill='x', pady=10)
        
        # Video canvas
        self.video_canvas = tk.Canvas(self.video_frame, bg="black", height=400)
        self.video_canvas.pack(fill='x', pady=10)
        
        # Initial message
        self.video_message = ctk.CTkLabel(
            self.video_canvas,
            text="Loading tutorial...",
            font=("Helvetica", 16),
            text_color="white"
        )
        self.video_message.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Video controls frame
        self.controls_frame = ctk.CTkFrame(self.video_frame)
        self.controls_frame.pack(fill='x', pady=5)
        
        # Play/Pause button
        self.play_button = ctk.CTkButton(
            self.controls_frame,
            text="▶️ Play",
            command=self.toggle_play,
            width=100
        )
        self.play_button.pack(side='left', padx=10)
        
        # Previous button
        self.prev_button = ctk.CTkButton(
            self.controls_frame,
            text="⏮️ Previous",
            command=self.previous_step,
            width=100
        )
        self.prev_button.pack(side='left', padx=10)
        
        # Next button
        self.next_button = ctk.CTkButton(
            self.controls_frame,
            text="Next ⏭️",
            command=self.next_step,
            width=100
        )
        self.next_button.pack(side='left', padx=10)
        
        # Progress bar
        self.progress_var = ctk.DoubleVar(value=0)
        self.progress_bar = ctk.CTkProgressBar(self.video_frame)
        self.progress_bar.pack(fill='x', padx=10, pady=5)
        self.progress_bar.set(0)
        
        # Current step indicator
        self.step_label = ctk.CTkLabel(
            self.video_frame,
            text="Step 1 of 4: Introduction",
            font=("Helvetica", 14)
        )
        self.step_label.pack(pady=5)
        
        # Description text
        self.description_frame = ctk.CTkFrame(self.content_frame)
        self.description_frame.pack(fill='x', pady=10)
        
        self.description_text = ctk.CTkTextbox(
            self.description_frame,
            height=100,
            font=("Helvetica", 12)
        )
        self.description_text.pack(fill='x', padx=10, pady=10)
        self.description_text.insert("1.0", "Welcome to the Pose2Sim tutorial. This guide will help you set up and use Pose2Sim effectively. Please follow along with the video instructions.")
        self.description_text.configure(state="disabled")
        
        # Dependency check frame
        self.dependency_frame = ctk.CTkFrame(self.content_frame)
        self.dependency_frame.pack(fill='x', pady=10)
        
        ctk.CTkLabel(
            self.dependency_frame,
            text="System Requirements Check",
            font=("Helvetica", 16, "bold")
        ).pack(pady=(10, 5))
        
        # Create a frame for each dependency
        self.dependency_items_frame = ctk.CTkFrame(self.dependency_frame)
        self.dependency_items_frame.pack(fill='x', padx=10, pady=10)
        
        # Create indicators for each dependency
        row = 0
        for dep_id, dep_info in self.dependencies.items():
            dep_frame = ctk.CTkFrame(self.dependency_items_frame, fg_color="transparent")
            dep_frame.grid(row=row, column=0, sticky="w", pady=2)
            
            # Status indicator
            status_label = ctk.CTkLabel(
                dep_frame,
                text="⏳",
                font=("Helvetica", 14),
                width=30
            )
            status_label.pack(side='left', padx=5)
            
            # Dependency name
            name_label = ctk.CTkLabel(
                dep_frame,
                text=dep_info["name"],
                font=("Helvetica", 14),
                width=150,
                anchor="w"
            )
            name_label.pack(side='left', padx=5)
            
            # Install button (hidden initially)
            install_button = ctk.CTkButton(
                dep_frame,
                text="Install",
                width=80,
                command=lambda d=dep_id: self.install_dependency(d)
            )
            install_button.pack(side='left', padx=5)
            install_button.pack_forget()
            
            # Store references to update later
            dep_info["status_label"] = status_label
            dep_info["install_button"] = install_button
            
            row += 1
        
        # Bottom buttons frame
        self.bottom_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.bottom_frame.pack(fill='x', pady=20)
        
        # Skip tutorial button (initially hidden)
        self.skip_button = ctk.CTkButton(
            self.bottom_frame,
            text="Skip Tutorial",
            command=self.skip_tutorial,
            width=150,
            fg_color="#FF9500",
            hover_color="#FF7000"
        )
        
        # Complete tutorial button (initially hidden)
        self.complete_button = ctk.CTkButton(
            self.bottom_frame,
            text="Complete Tutorial",
            command=self.complete_tutorial,
            width=150,
            fg_color="#4CAF50",
            hover_color="#388E3C"
        )
        
        # Load the first tutorial step
        self.load_tutorial_step(0)
    
    def check_tutorial_status(self):
        """Check if the tutorial has been completed before"""
        if os.path.exists(self.marker_file):
            # Tutorial has been completed before, show skip button
            self.skip_button.pack(side='right', padx=10)
        else:
            # First time user, show complete button on last step
            if self.current_step == self.total_steps - 1:
                self.complete_button.pack(side='right', padx=10)
    
    def load_tutorial_step(self, step_index):
        """Load a specific tutorial step"""
        # Update step counter
        self.current_step = step_index
        
        # Update step label
        step_titles = [
            "Introduction",
            "Installation & Setup",
            "Configuration Workflow",
            "Advanced Features"
        ]
        
        if 0 <= step_index < len(step_titles):
            self.step_label.configure(text=f"Step {step_index + 1} of {self.total_steps}: {step_titles[step_index]}")
        
        # Load the appropriate video for this step
        tutorial_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tutorial")
        video_path = os.path.join(tutorial_folder, f"tutorial_step{step_index + 1}.mp4")
        
        # Check if video exists
        if os.path.exists(video_path):
            self.load_video(video_path)
        else:
            # Show message if video doesn't exist
            self.video_message.configure(text=f"Tutorial video not found:\n{video_path}")
            self.video_message.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Update description text
        descriptions = [
            "Welcome to Pose2Sim! This tutorial will guide you through the setup and use of this powerful 3D pose estimation tool. "
            "Pose2Sim enables you to transform 2D keypoints from multiple camera views into accurate 3D pose data.",
            
            "For Pose2Sim to work correctly, you need to install several dependencies. We'll check your system for Anaconda, "
            "OpenSim, PyTorch, and other required components. Missing components can be installed with the buttons below.",
            
            "The Pose2Sim workflow has several steps: calibration, pose estimation, synchronization, triangulation, filtering, and more. "
            "This interface guides you through each step with clear instructions.",
            
            "Pose2Sim includes advanced features like batch processing, marker augmentation, and customizable filtering. "
            "These options help you adapt the tool to different research needs."
        ]
        
        if 0 <= step_index < len(descriptions):
            self.description_text.configure(state="normal")
            self.description_text.delete("1.0", tk.END)
            self.description_text.insert("1.0", descriptions[step_index])
            self.description_text.configure(state="disabled")
        
        # Enable/disable navigation buttons
        self.prev_button.configure(state="normal" if step_index > 0 else "disabled")
        self.next_button.configure(state="normal" if step_index < self.total_steps - 1 else "disabled")
        
        # Show/hide complete button based on step
        self.complete_button.pack_forget()
        if step_index == self.total_steps - 1 and not os.path.exists(self.marker_file):
            self.complete_button.pack(side='right', padx=10)
    
    def load_video(self, video_path):
        """Load a video file for the tutorial"""
        try:
            # Close any previously open video
            if self.video_cap is not None:
                self.video_cap.release()
            
            # Hide video message
            self.video_message.place_forget()
            
            # Open the video file
            self.video_cap = cv2.VideoCapture(video_path)
            
            if not self.video_cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            self.video_path = video_path
            self.current_frame = 0
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Display first frame
            self.update_video_frame()
            
            # Enable play button
            self.play_button.configure(state="normal", text="▶️ Play")
            self.playing = False
            
        except Exception as e:
            self.video_message.configure(text=f"Error loading video:\n{str(e)}")
            self.video_message.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def update_video_frame(self):
        """Update video display with current frame"""
        if self.video_cap is None:
            return
            
        # Seek to the current frame
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        # Read the frame
        ret, frame = self.video_cap.read()
        
        if not ret:
            # End of video
            self.playing = False
            self.play_button.configure(text="▶️ Play")
            return
            
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get canvas dimensions
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 800
            canvas_height = 400
        
        # Calculate scaling to fit the canvas while maintaining aspect ratio
        frame_h, frame_w = frame_rgb.shape[:2]
        scale = min(canvas_width / frame_w, canvas_height / frame_h)
        
        new_width = int(frame_w * scale)
        new_height = int(frame_h * scale)
        
        # Resize the frame
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        
        # Convert to PIL Image
        image = Image.fromarray(frame_resized)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image=image)
        
        # Update canvas
        self.video_canvas.delete("all")
        
        # Center the image
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        
        self.video_canvas.create_image(x_offset, y_offset, anchor="nw", image=self.photo)
        
        # Update progress bar
        progress = self.current_frame / max(1, self.total_frames - 1)
        self.progress_bar.set(progress)
    
    def toggle_play(self):
        """Toggle video playback"""
        if self.video_cap is None:
            return
            
        self.playing = not self.playing
        
        if self.playing:
            self.play_button.configure(text="⏸️ Pause")
            self.play_video()
        else:
            self.play_button.configure(text="▶️ Play")
            # Cancel scheduled frame updates
            if self.play_after_id:
                self.frame.after_cancel(self.play_after_id)
                self.play_after_id = None
    
    def play_video(self):
        """Play the video from the current position"""
        if not self.playing or self.video_cap is None:
            return
        
        # Move to next frame
        self.current_frame += 1
        
        # Check if we reached the end
        if self.current_frame >= self.total_frames:
            self.current_frame = 0  # Loop back to beginning
        
        # Update the display
        self.update_video_frame()
        
        # Schedule next frame update
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / max(1, fps))  # milliseconds between frames
        
        self.play_after_id = self.frame.after(delay, self.play_video)
    
    def next_step(self):
        """Move to the next tutorial step"""
        if self.current_step < self.total_steps - 1:
            self.load_tutorial_step(self.current_step + 1)
    
    def previous_step(self):
        """Move to the previous tutorial step"""
        if self.current_step > 0:
            self.load_tutorial_step(self.current_step - 1)
    
    def skip_tutorial(self):
        """Skip the tutorial and move to the main app"""
        # Confirm the user wants to skip
        response = messagebox.askyesno(
            "Skip Tutorial",
            "Are you sure you want to skip the tutorial? You can access it again from the Tutorial tab later."
        )
        
        if response:
            # Stop video playback
            self.playing = False
            if self.play_after_id:
                self.frame.after_cancel(self.play_after_id)
            
            # Release video resource
            if self.video_cap:
                self.video_cap.release()
            
            # Move to the next tab
            if hasattr(self.app, 'show_tab'):
                tab_order = list(self.app.tabs.keys())
                current_idx = tab_order.index('tutorial')
                if current_idx + 1 < len(tab_order):
                    next_tab = tab_order[current_idx + 1]
                    self.app.show_tab(next_tab)
    
    def complete_tutorial(self):
        """Mark the tutorial as completed and continue to the app"""
        # Create marker file to indicate tutorial completion
        try:
            with open(self.marker_file, 'w') as f:
                f.write("Tutorial completed")
                
            messagebox.showinfo(
                "Tutorial Complete",
                "You have completed the Pose2Sim tutorial. You can access it again at any time from the Tutorial tab."
            )
            
            # Move to the next tab
            if hasattr(self.app, 'show_tab'):
                tab_order = list(self.app.tabs.keys())
                current_idx = tab_order.index('tutorial')
                if current_idx + 1 < len(tab_order):
                    next_tab = tab_order[current_idx + 1]
                    self.app.show_tab(next_tab)
                    
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to mark tutorial as completed: {str(e)}"
            )
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        # Check for Anaconda
        self.check_anaconda()
        
        # Check for anaconda in PATH
        self.check_anaconda_path()
        
        # Check for pose2sim
        self.check_package("pose2sim")
        
        # Check for OpenSim
        self.check_package("opensim")
        
        # Check for PyTorch
        self.check_pytorch()
        
        # Check for ONNX Runtime GPU
        self.check_package("onnxruntime-gpu")
        
        # Update UI with results
        self.frame.after(0, self.update_dependency_ui)
    
    def check_anaconda(self):
        """Check if Anaconda is installed"""
        try:
            # Check for conda executable
            if sys.platform == 'win32':
                result = subprocess.run(["where", "conda"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            else:
                result = subprocess.run(["which", "conda"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                self.dependencies["anaconda"]["installed"] = True
            else:
                self.dependencies["anaconda"]["installed"] = False
        except Exception:
            self.dependencies["anaconda"]["installed"] = False
    
    def check_anaconda_path(self):
        """Check if Anaconda is in PATH"""
        try:
            # Try to run conda command
            result = subprocess.run(["conda", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0:
                self.dependencies["path"]["installed"] = True
            else:
                self.dependencies["path"]["installed"] = False
        except Exception:
            self.dependencies["path"]["installed"] = False
    
    def check_package(self, package_name):
        """Check if a Python package is installed"""
        try:
            if package_name == "opensim":
                # Special check for OpenSim
                cmd = ["conda", "list", "opensim"]
            else:
                # Check with pip
                cmd = ["pip", "show", package_name]
                
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            dep_key = package_name.lower().replace("-", "-")
            if result.returncode == 0 and result.stdout.strip():
                self.dependencies[dep_key]["installed"] = True
            else:
                self.dependencies[dep_key]["installed"] = False
        except Exception:
            self.dependencies[package_name.lower().replace("-", "-")]["installed"] = False
    
    def check_pytorch(self):
        """Check if PyTorch with CUDA is installed"""
        try:
            # Execute a Python script to check PyTorch and CUDA
            check_cmd = [
                sys.executable,
                "-c",
                "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
            ]
            
            result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0 and "CUDA: True" in result.stdout:
                self.dependencies["pytorch"]["installed"] = True
            else:
                self.dependencies["pytorch"]["installed"] = False
        except Exception:
            self.dependencies["pytorch"]["installed"] = False
    
    def update_dependency_ui(self):
        """Update UI with dependency check results"""
        for dep_id, dep_info in self.dependencies.items():
            status_label = dep_info["status_label"]
            install_button = dep_info["install_button"]
            
            if dep_info["installed"]:
                status_label.configure(text="✅", text_color="#4CAF50")
                install_button.pack_forget()
            else:
                status_label.configure(text="❌", text_color="#F44336")
                install_button.pack(side='left', padx=5)
    
    def install_dependency(self, dependency_id):
        """Install a missing dependency"""
        commands = {
            "anaconda": {
                "message": "Please download and install Anaconda from:\nhttps://www.anaconda.com/products/distribution",
                "command": None  # Manual installation required
            },
            "path": {
                "message": "Anaconda is installed but not in PATH. Please add it to your system PATH.",
                "command": None  # Manual configuration required
            },
            "pose2sim": {
                "message": "Installing Pose2Sim...",
                "command": ["pip", "install", "pose2sim"]
            },
            "opensim": {
                "message": "Installing OpenSim...",
                "command": ["conda", "install", "-c", "opensim-org", "opensim", "-y"]
            },
            "pytorch": {
                "message": "Installing PyTorch with CUDA...",
                "command": ["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu124"]
            },
            "onnxruntime-gpu": {
                "message": "Installing ONNX Runtime GPU...",
                "command": ["pip", "uninstall", "onnxruntime", "-y", "&&", "pip", "install", "onnxruntime-gpu"]
            }
        }
        
        if dependency_id not in commands:
            messagebox.showerror("Error", f"Unknown dependency: {dependency_id}")
            return
        
        dep_info = commands[dependency_id]
        
        if dep_info["command"] is None:
            # Manual installation required
            messagebox.showinfo("Manual Installation", dep_info["message"])
            return
        
        # Show installation dialog
        progress_window = ctk.CTkToplevel(self.frame)
        progress_window.title(f"Installing {self.dependencies[dependency_id]['name']}")
        progress_window.geometry("400x200")
        progress_window.transient(self.frame)
        progress_window.grab_set()
        
        # Message
        message_label = ctk.CTkLabel(
            progress_window,
            text=dep_info["message"],
            font=("Helvetica", 14)
        )
        message_label.pack(pady=(20, 10))
        
        # Progress indicator
        progress = ctk.CTkProgressBar(progress_window)
        progress.pack(fill='x', padx=20, pady=10)
        progress.configure(mode="indeterminate")
        progress.start()
        
        # Status
        status_label = ctk.CTkLabel(
            progress_window,
            text="Starting installation...",
            font=("Helvetica", 12)
        )
        status_label.pack(pady=10)
        
        # Run installation in a separate thread
        def install_thread():
            try:
                # Update status
                self.frame.after(0, lambda: status_label.configure(text="Installation in progress..."))
                
                # Execute command
                if "&&" in dep_info["command"]:
                    # Handle compound commands (uninstall and then install)
                    cmd1 = dep_info["command"][:dep_info["command"].index("&&")]
                    cmd2 = dep_info["command"][dep_info["command"].index("&&")+1:]
                    
                    # Run first command
                    result1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    # Run second command
                    result2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    success = result2.returncode == 0
                else:
                    # Single command
                    result = subprocess.run(dep_info["command"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    success = result.returncode == 0
                
                # Update UI based on result
                if success:
                    self.frame.after(0, lambda: status_label.configure(
                        text="Installation completed successfully!", 
                        text_color="#4CAF50"
                    ))
                    
                    # Update dependency status
                    self.dependencies[dependency_id]["installed"] = True
                    self.frame.after(500, self.update_dependency_ui)
                else:
                    self.frame.after(0, lambda: status_label.configure(
                        text="Installation failed. Please try manual installation.",
                        text_color="#F44336"
                    ))
                
                # Add close button
                self.frame.after(0, lambda: ctk.CTkButton(
                    progress_window,
                    text="Close",
                    command=progress_window.destroy
                ).pack(pady=10))
                
                # Stop progress animation
                self.frame.after(0, progress.stop)
                
            except Exception as e:
                # Show error
                self.frame.after(0, lambda: status_label.configure(
                    text=f"Error: {str(e)}",
                    text_color="#F44336"
                ))
                
                # Add close button
                self.frame.after(0, lambda: ctk.CTkButton(
                    progress_window,
                    text="Close",
                    command=progress_window.destroy
                ).pack(pady=10))
                
                # Stop progress animation
                self.frame.after(0, progress.stop)
        
        # Start installation thread
        threading.Thread(target=install_thread, daemon=True).start()