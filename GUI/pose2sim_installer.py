import os
import sys
import platform
import subprocess
import tempfile
import urllib.request
import shutil
from pathlib import Path

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
        else:  # macOS/Linux
            launcher_path = os.path.join(self.home_dir, "Desktop", "Pose2Sim.sh")
            with open(launcher_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"source {self.home_dir}/miniconda3/bin/activate Pose2Sim\n")
                f.write("echo 'Pose2Sim environment activated!'\n")
                f.write("exec $SHELL\n")
            
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

if __name__ == "__main__":
    installer = Pose2SimInstaller()
    
    try:
        installer.install()
    except Exception as e:
        print(f"ERROR: Installation failed: {e}")
        print("Please check the error message and try again.")
        sys.exit(1)
        
    input("Press Enter to exit...")