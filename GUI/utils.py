import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def generate_checkerboard_image(width, height, square_size):
    """
    Generate a checkerboard image for display.
    
    Args:
        width: Number of internal corners in the checkerboard width
        height: Number of internal corners in the checkerboard height
        square_size: Size of each square in pixels
    
    Returns:
        PIL Image of the checkerboard
    """
    # Add 1 to include outer corners
    num_rows = height + 1
    num_cols = width + 1
    square_size = int(square_size)
    
    # Create checkerboard pattern
    pattern = np.zeros((num_rows * square_size, num_cols * square_size), dtype=np.uint8)
    for row in range(num_rows):
        for col in range(num_cols):
            if (row + col) % 2 == 0:
                pattern[row*square_size:(row+1)*square_size,
                        col*square_size:(col+1)*square_size] = 255
    
    # Convert to PIL Image
    return Image.fromarray(pattern)

def extract_frames_from_video(video_path, output_dir, time_interval):
    """
    Extract frames from a video at regular intervals.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the extracted frames
        time_interval: Time interval between frames in seconds
    
    Returns:
        List of paths to the extracted frames
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default to 30 fps if detection fails
    
    # Calculate frame interval
    frame_interval = int(fps * time_interval)
    
    # Extract frames
    extracted_frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Save frame as image
            frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{frame_count}.png"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
        
        frame_count += 1
    
    cap.release()
    return extracted_frames

def create_point_selection_canvas(parent, image_path, points_callback, max_points=8):
    """
    Create a canvas for selecting points on an image.
    
    Args:
        parent: Parent widget to attach the canvas to
        image_path: Path to the image
        points_callback: Function to call with selected points
        max_points: Maximum number of points to select
    
    Returns:
        Canvas widget
    """
    # Load the image
    if image_path.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(image_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read video frame from: {image_path}")
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        image = plt.imread(image_path)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.set_title(f"Click to select {max_points} points")
    
    # Store selected points
    points = []
    point_markers = []
    
    def onclick(event):
        if len(points) < max_points:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                points.append((x, y))
                # Plot point in red
                point = ax.plot(x, y, 'ro')[0]
                point_markers.append(point)
                # Add point number
                ax.text(x + 5, y + 5, str(len(points)), color='white')
                fig.canvas.draw()
                
                if len(points) == max_points:
                    # Call the callback with the selected points
                    points_callback(points)
    
    # Create canvas and connect click event
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    canvas.mpl_connect('button_press_event', onclick)
    
    return canvas

def activate_pose2sim(participant_path, method='cmd', skip_pose_estimation=False, skip_synchronization=False, analysis_mode='3d'):
    """
    Create scripts to activate Pose2Sim or Sports2D with the specified method.
    
    Args:
        participant_path: Path to the participant directory
        method: Method to use ('cmd', 'conda', or 'powershell')
        skip_pose_estimation: Whether to skip pose estimation
        skip_synchronization: Whether to skip synchronization
        analysis_mode: '2d' or '3d'
    
    Returns:
        Path to the created script
    """
    if analysis_mode == '3d':
        # Generate Python script content for Pose2Sim (3D)
        python_script = f"""
from Pose2Sim import Pose2Sim
Pose2Sim.runAll(do_calibration=True, 
                do_poseEstimation={not skip_pose_estimation}, 
                do_synchronization={not skip_synchronization}, 
                do_personAssociation=True, 
                do_triangulation=True, 
                do_filtering=True, 
                do_markerAugmentation=True, 
                do_kinematics=True)
"""
        script_path = os.path.join(participant_path, 'run_pose2sim.py')
    else:
        # Generate Python script content for Sports2D (2D)
        python_script = """
from Sports2D import Sports2D
Sports2D.process('Config_demo.toml')
"""
        script_path = os.path.join(participant_path, 'run_sports2d.py')
    
    # Save the Python script
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(python_script)
    
    # Create the appropriate conda environment name
    conda_env = "Sports2D" if analysis_mode == '2d' else "Pose2Sim"
    
    # Generate launch script based on method
    if method == 'cmd':
        if analysis_mode == '3d':
            launch_script = f"""
@echo off
setlocal EnableDelayedExpansion

REM Activate Conda environment
call conda activate {conda_env}

REM Change to the specified directory
cd "{os.path.abspath(participant_path)}"

REM Launch the Python script and keep the command prompt open
python {os.path.basename(script_path)}

REM Pause the command prompt to prevent it from closing
pause

endlocal
"""
        else:  # 2D mode
            launch_script = f"""
@echo off
setlocal EnableDelayedExpansion

REM Activate Conda environment
call conda activate {conda_env}

REM Change to the specified directory
cd "{os.path.abspath(participant_path)}"

REM Launch IPython and execute Sports2D command
ipython -c "from Sports2D import Sports2D; Sports2D.process('Config_demo.toml')"

REM Pause the command prompt to prevent it from closing
pause

endlocal
"""
        script_ext = 'cmd'
        
    elif method == 'conda':
        if analysis_mode == '3d':
            launch_script = f"""
@echo off
setlocal EnableDelayedExpansion

REM Change to the specified directory
cd "{os.path.abspath(participant_path)}"

REM Launch the Python script
call conda activate {conda_env} && python {os.path.basename(script_path)}

REM Pause to keep the window open
pause

endlocal
"""
        else:  # 2D mode
            launch_script = f"""
@echo off
setlocal EnableDelayedExpansion

REM Change to the specified directory
cd "{os.path.abspath(participant_path)}"

REM Launch IPython and execute Sports2D command
call conda activate {conda_env} && ipython -c "from Sports2D import Sports2D; Sports2D.process('Config_demo.toml')"

REM Pause to keep the window open
pause

endlocal
"""
        script_ext = 'bat'
        
    else:  # powershell
        if analysis_mode == '3d':
            launch_script = f"""
# Change to the specified directory
cd "{os.path.abspath(participant_path)}"

# Activate Conda environment and run script
conda activate {conda_env}; python {os.path.basename(script_path)}

# Pause to keep the window open
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
"""
        else:  # 2D mode
            launch_script = f"""
# Change to the specified directory
cd "{os.path.abspath(participant_path)}"

# Activate Conda environment and run IPython command
conda activate {conda_env}; ipython -c "from Sports2D import Sports2D; Sports2D.process('Config_demo.toml')"

# Pause to keep the window open
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
"""
        script_ext = 'ps1'
    
    # Save the launch script
    launch_script_path = os.path.join(participant_path, f'activate_{conda_env.lower()}_{method}.{script_ext}')
    with open(launch_script_path, 'w', encoding='utf-8') as f:
        f.write(launch_script)
    
    return launch_script_path