import os
import sys
import platform
import subprocess
import tempfile
import urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import threading
import io

class Pose2SimInstaller:
    def __init__(self):
        self.os_type = platform.system()
        self.miniconda_installed = self.check_miniconda()
        self.home_dir = str(Path.home())
        
    def check_miniconda(self):
        """Check if Miniconda/Anaconda is installed"""
        try:
            subprocess.run(["conda", "--version"], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
            
    def download_miniconda(self):
        """Download appropriate Miniconda installer"""
        print("Downloading Miniconda installer...")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        installer_path = os.path.join(temp_dir, "miniconda_installer")
        
        # Select correct installer based on OS
        if self.os_type == "Windows":
            url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
            installer_path += ".exe"
        elif self.os_type == "Darwin":  # macOS
            if platform.machine() == "arm64":  # Apple Silicon
                url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
            else:  # Intel
                url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
            installer_path += ".sh"
        else:  # Linux
            url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            installer_path += ".sh"
            
        # Download the installer
        urllib.request.urlretrieve(url, installer_path)
        
        return installer_path
        
    def install_miniconda(self, installer_path):
        """Install Miniconda"""
        print("Installing Miniconda...")
        
        if self.os_type == "Windows":
            subprocess.run([installer_path, "/InstallationType=JustMe", 
                           "/RegisterPython=0", "/S", f"/D={self.home_dir}\\Miniconda3"], 
                           check=True)
        else:  # macOS or Linux
            subprocess.run(["bash", installer_path, "-b", "-p", 
                           f"{self.home_dir}/miniconda3"], check=True)
            
        # Update PATH for current process
        if self.os_type == "Windows":
            os.environ["PATH"] = f"{self.home_dir}\\Miniconda3;{self.home_dir}\\Miniconda3\\Scripts;" + os.environ["PATH"]
        else:
            os.environ["PATH"] = f"{self.home_dir}/miniconda3/bin:" + os.environ["PATH"]
    
    def setup_pose2sim(self):
        """Set up Pose2Sim environment and dependencies"""
        print("\nSetting up Pose2Sim environment...")
        
        # Determine conda executable path
        conda_exec = "conda"
        if self.os_type == "Windows":
            conda_exec = f"{self.home_dir}\\Miniconda3\\Scripts\\conda.exe"
        
        # Create and configure environment
        commands = [
            [conda_exec, "create", "-n", "Pose2Sim", "python=3.10", "-y"],
            [conda_exec, "install", "-c", "opensim-org", "opensim", "-y", "--name", "Pose2Sim"],
        ]
        
        # For Windows, activate environment first
        if self.os_type == "Windows":
            activate_cmd = f"{self.home_dir}\\Miniconda3\\Scripts\\activate.bat"
            pip_commands = [
                f"call {activate_cmd} Pose2Sim && pip install pose2sim",
                f"call {activate_cmd} Pose2Sim && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
                f"call {activate_cmd} Pose2Sim && pip uninstall -y onnxruntime",
                f"call {activate_cmd} Pose2Sim && pip install onnxruntime-gpu"
            ]
        else:  # macOS/Linux
            activate_cmd = f"source {self.home_dir}/miniconda3/bin/activate Pose2Sim"
            pip_commands = [
                f"{activate_cmd} && pip install pose2sim",
                f"{activate_cmd} && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
                f"{activate_cmd} && pip uninstall -y onnxruntime",
                f"{activate_cmd} && pip install onnxruntime-gpu"
            ]
        
        # Execute conda commands
        for cmd in commands:
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        
        # Execute pip commands through shell
        for cmd in pip_commands:
            print(f"Running: {cmd}")
            if self.os_type == "Windows":
                subprocess.run(cmd, shell=True, check=True)
            else:
                subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)
    
    def create_launcher(self):
        """Create a launcher for Pose2Sim"""
        print("\nCreating Pose2Sim launcher...")
        
        if self.os_type == "Windows":
            launcher_path = os.path.join(os.environ["USERPROFILE"], "Desktop", "Pose2Sim.bat")
            with open(launcher_path, "w") as f:
                f.write(f"@echo off\n")
                f.write(f"call {self.home_dir}\\Miniconda3\\Scripts\\activate.bat Pose2Sim\n")
                f.write(f"echo Pose2Sim environment activated!\n")
                f.write(f"cmd /k\n")
                f.write(f"pose2sim\n")
        else:  # macOS/Linux
            launcher_path = os.path.join(self.home_dir, "Desktop", "Pose2Sim.sh")
            with open(launcher_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"source {self.home_dir}/miniconda3/bin/activate Pose2Sim\n")
                f.write("echo 'Pose2Sim environment activated!'\n")
                f.write("exec $SHELL\n")
                f.write(f"pose2sim\n")
            
            # Make executable
            os.chmod(launcher_path, 0o755)
            
        print(f"Launcher created at: {launcher_path}")
    
    def install(self):
        """Run the complete installation process"""
        print("=== Pose2Sim Installer ===")
        
        # Check for GPU support
        try:
            if self.os_type == "Windows":
                nvidia_smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE)
                if nvidia_smi.returncode != 0:
                    print("WARNING: NVIDIA GPU not detected. GPU acceleration won't be available.")
        except FileNotFoundError:
            print("WARNING: NVIDIA GPU not detected. GPU acceleration won't be available.")
        
        # Install Miniconda if needed
        if not self.miniconda_installed:
            installer_path = self.download_miniconda()
            self.install_miniconda(installer_path)
        else:
            print("Miniconda already installed, skipping installation.")
        
        # Setup Pose2Sim
        self.setup_pose2sim()
        
        # Create launcher
        self.create_launcher()
        
        print("\n=== Installation Complete! ===")
        print("You can now use Pose2Sim by running the created launcher.")


class InstallerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose2Sim Installer")
        self.root.geometry("600x400")
        self.root.resizable(True, True)
        
        # Store original stdout before any redirection
        self.original_stdout = sys.stdout
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add title
        title_label = ttk.Label(main_frame, text="Pose2Sim Installer", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Add description
        desc_text = "This installer will set up Pose2Sim with all required dependencies."
        desc_label = ttk.Label(main_frame, text=desc_text, wraplength=500)
        desc_label.pack(pady=(0, 20))
        
        # Create text box for logs
        self.log_box = tk.Text(main_frame, height=15, width=70, state="disabled")
        self.log_box.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Add scrollbar to text box
        scrollbar = ttk.Scrollbar(self.log_box, command=self.log_box.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_box.config(yscrollcommand=scrollbar.set)
        
        # Add progress bar
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=500, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(0, 20))
        
        # Add buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        self.install_button = ttk.Button(button_frame, text="Install", command=self.start_installation)
        self.install_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = ttk.Button(button_frame, text="Exit", command=self.cleanup_and_exit)
        self.exit_button.pack(side=tk.RIGHT, padx=5)
        
        # Set up output redirection
        sys.stdout = self.RedirectText(self)
    
    def cleanup_and_exit(self):
        """Restore stdout and exit"""
        sys.stdout = self.original_stdout
        self.root.destroy()
    
    class RedirectText:
        def __init__(self, gui):
            self.gui = gui
            
        def write(self, string):
            if self.gui.original_stdout:
                self.gui.original_stdout.write(string)
            self.gui.update_log(string)
            
        def flush(self):
            if self.gui.original_stdout:
                self.gui.original_stdout.flush()
    
    def update_log(self, text):
        """Update the log box with new text"""
        self.root.after(0, self._update_log, text)
    
    def _update_log(self, text):
        """Actually update the log box (must be called from main thread)"""
        self.log_box.config(state="normal")
        self.log_box.insert(tk.END, text)
        self.log_box.see(tk.END)
        self.log_box.config(state="disabled")
    
    def start_installation(self):
        """Start the installation process in a separate thread"""
        self.install_button.config(state="disabled")
        self.progress.start()
        print("Starting installation...\n")
        
        # Create and start installation thread
        install_thread = threading.Thread(target=self.run_installation)
        install_thread.daemon = True
        install_thread.start()
    
    def run_installation(self):
        """Run the actual installation process"""
        try:
            installer = Pose2SimInstaller()
            installer.install()
            
            # Update UI on completion
            self.root.after(0, self.installation_complete, True)
        except Exception as e:
            error_msg = f"ERROR: Installation failed: {str(e)}\n"
            print(error_msg)
            # Update UI on failure
            self.root.after(0, self.installation_complete, False)
    
    def installation_complete(self, success):
        """Handle installation completion"""
        self.progress.stop()
        
        if success:
            print("\nInstallation completed successfully!")
        else:
            print("\nInstallation failed. Please check the log for details.")
        
        self.install_button.config(state="normal")
        self.install_button.config(text="Close")
        self.install_button.config(command=self.cleanup_and_exit)


if __name__ == "__main__":
    # Check if we should use GUI or console mode
    use_gui = True
    
    # Use console mode if --console argument is provided
    if len(sys.argv) > 1 and "--console" in sys.argv:
        use_gui = False
    
    if use_gui:
        # Run with GUI
        root = tk.Tk()
        app = InstallerGUI(root)
        root.mainloop()
    else:
        # Run in console mode
        installer = Pose2SimInstaller()
        try:
            installer.install()
            print("\nInstallation completed successfully!")
        except Exception as e:
            print(f"ERROR: Installation failed: {e}")
            print("Please check the error message and try again.")
            sys.exit(1)
        finally:
            # Keep the window open when double-clicked in Windows
            if platform.system() == "Windows":
                # Only ask for input if it's not being run from a terminal
                if not os.environ.get("PROMPT"):
                    print("\nPress Enter to exit...")
                    input()
            else:
                input("\nPress Enter to exit...")