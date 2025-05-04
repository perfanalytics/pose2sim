import os
import sys
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
import subprocess
import threading
import webbrowser
import cv2
from PIL import Image, ImageTk

class TutorialTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        # Create main frame
        self.frame = ctk.CTkFrame(parent)
        
        # Initialize variables
        self.marker_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tutorial_completed")
        
        # Video links
        self.video_links = {
            '2d': "https://drive.google.com/file/d/1Lglv-1tdO4FFKUl2LA7dKhYvPcsWbLmJ/view?usp=drive_link",
            '3d': "https://drive.google.com/file/d/1fNQDtc0f1jYOrgqkQcVHQ3XPfbdTIqTr/view?usp=drive_link"
        }
        
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
        
        # Video information section
        video_info_frame = ctk.CTkFrame(self.content_frame, fg_color=("gray95", "gray20"))
        video_info_frame.pack(fill='x', pady=10, padx=20)
        
        ctk.CTkLabel(
            video_info_frame,
            text="Due to size, tutorial videos are hosted on Google Drive",
            font=("Helvetica", 16, "bold"),
            wraplength=600
        ).pack(pady=(10, 5))
        
        # Get the analysis mode
        analysis_mode = getattr(self.app, 'analysis_mode', '3d')
        
        # Video buttons frame
        video_buttons_frame = ctk.CTkFrame(video_info_frame, fg_color="transparent")
        video_buttons_frame.pack(pady=10)
        
        # Button for current mode video
        current_mode_text = "Watch 2D Tutorial Video" if analysis_mode == '2d' else "Watch 3D Tutorial Video"
        ctk.CTkButton(
            video_buttons_frame,
            text=current_mode_text,
            command=lambda: self.open_video_link(analysis_mode),
            font=("Helvetica", 14, "bold"),
            width=250,
            height=40,
            fg_color="#4CAF50",
            hover_color="#388E3C"
        ).pack(side='left', padx=10)
        
        # Button for other mode video
        other_mode = '3d' if analysis_mode == '2d' else '2d'
        other_mode_text = "Watch 3D Tutorial Video" if analysis_mode == '2d' else "Watch 2D Tutorial Video"
        ctk.CTkButton(
            video_buttons_frame,
            text=other_mode_text,
            command=lambda: self.open_video_link(other_mode),
            font=("Helvetica", 14),
            width=250,
            height=40
        ).pack(side='left', padx=10)
        
        # Tutorial image placeholder
        self.tutorial_img_frame = ctk.CTkFrame(self.content_frame, height=300)
        self.tutorial_img_frame.pack(fill='x', pady=10)
        
        # Load a placeholder image or tutorial screenshot if available
        tutorial_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "tutorial_preview.png")
        
        if os.path.exists(tutorial_img_path):
            try:
                # Load and display image
                img = Image.open(tutorial_img_path)
                img = img.resize((800, 300), Image.LANCZOS)
                self.tutorial_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 300))
                
                img_label = ctk.CTkLabel(self.tutorial_img_frame, image=self.tutorial_img, text="")
                img_label.pack(pady=10)
            except Exception as e:
                ctk.CTkLabel(
                    self.tutorial_img_frame,
                    text="Tutorial Preview Image Not Available",
                    font=("Helvetica", 16)
                ).pack(expand=True)
        else:
            ctk.CTkLabel(
                self.tutorial_img_frame,
                text="Tutorial Preview Image Not Available",
                font=("Helvetica", 16)
            ).pack(expand=True)
        
        # Add beta version message box
        self.beta_message_frame = ctk.CTkFrame(self.content_frame, fg_color="white")
        self.beta_message_frame.pack(fill='x', pady=10, padx=40)

        self.beta_message = ctk.CTkLabel(
            self.beta_message_frame,
            text="This GUI is a beta version. If you have recommendations, errors, or suggestions please send them to yacine.pose2sim@gmail.com",
            font=("Helvetica", 12),
            text_color="black",
            wraplength=600
        )
        self.beta_message.pack(pady=10, padx=10)
        
        # Description text
        self.description_frame = ctk.CTkFrame(self.content_frame)
        self.description_frame.pack(fill='x', pady=10)
        
        self.description_text = ctk.CTkTextbox(
            self.description_frame,
            height=100,
            font=("Helvetica", 12)
        )
        self.description_text.pack(fill='x', padx=10, pady=10)
        
        description = (
            "Welcome to the Pose2Sim tutorial. This guide will help you set up and use Pose2Sim effectively.\n\n"
            "The tutorial videos cover:\n"
            "• Configuration workflow\n"
            "• Data processing\n"
            "• Advanced features\n\n"
            "Click on the video link above to watch the complete tutorial on Google Drive."
        )
        
        self.description_text.insert("1.0", description)
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
        
        # Skip tutorial button
        self.skip_button = ctk.CTkButton(
            self.bottom_frame,
            text="Skip Tutorial",
            command=self.skip_tutorial,
            width=150,
            fg_color="#FF9500",
            hover_color="#FF7000"
        )
        self.skip_button.pack(side='right', padx=10)
        
        # Complete tutorial button
        self.complete_button = ctk.CTkButton(
            self.bottom_frame,
            text="Complete Tutorial",
            command=self.complete_tutorial,
            width=150,
            fg_color="#4CAF50",
            hover_color="#388E3C"
        )
        self.complete_button.pack(side='right', padx=10)
    
    def open_video_link(self, mode):
        """Open the video link in a web browser"""
        if mode in self.video_links:
            webbrowser.open(self.video_links[mode])
    
    def check_tutorial_status(self):
        """Check if the tutorial has been completed before"""
        if os.path.exists(self.marker_file):
            # Tutorial has been completed before, only show skip button
            self.complete_button.pack_forget()
        else:
            # First time user, show both buttons
            pass
    
    def skip_tutorial(self):
        """Skip the tutorial and move to the main app"""
        # Confirm the user wants to skip
        response = messagebox.askyesno(
            "Skip Tutorial",
            "Are you sure you want to skip the tutorial? You can access it again from the Tutorial tab later."
        )
        
        if response:
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