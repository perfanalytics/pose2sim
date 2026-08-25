import os
import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2
from PIL import Image, ImageTk

class VisualizationTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        # Create main frame
        self.frame = ctk.CTkFrame(parent)
        
        # Initialize data variables
        self.trc_data = None
        self.mot_data = None
        self.video_cap = None
        self.video_path = None
        self.current_frame = 0
        self.playing = False
        self.play_after_id = None
        
        # Selected angles/segments for visualization
        self.selected_angles = []
        
        # Stores current time line object in angle plot
        self.time_line = None
        
        # Build the UI
        self.build_ui()
    
    def get_title(self):
        """Return the tab title"""
        return "Data Visualization"
    
    def get_settings(self):
        """Get the visualization settings"""
        return {}  # This tab doesn't add settings to the config file
    
    def build_ui(self):
        # Create main layout with left and right panels
        self.main_paned_window = ctk.CTkFrame(self.frame)
        self.main_paned_window.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Top control panel
        self.control_panel = ctk.CTkFrame(self.main_paned_window)
        self.control_panel.pack(fill='x', pady=(0, 10))
        
        # File control frame
        file_frame = ctk.CTkFrame(self.control_panel)
        file_frame.pack(side='left', fill='y', padx=10, pady=5)
        
        ctk.CTkLabel(file_frame, text="Data Files:", font=("Helvetica", 12, "bold")).pack(side='left', padx=5)
        
        self.auto_detect_btn = ctk.CTkButton(
            file_frame, 
            text="Auto-Detect Files", 
            command=self.auto_detect_files,
            width=120
        )
        self.auto_detect_btn.pack(side='left', padx=5)
        
        self.load_trc_btn = ctk.CTkButton(
            file_frame, 
            text="Load TRC", 
            command=self.load_trc_file,
            width=80
        )
        self.load_trc_btn.pack(side='left', padx=5)
        
        self.load_mot_btn = ctk.CTkButton(
            file_frame, 
            text="Load MOT", 
            command=self.load_mot_file,
            width=80
        )
        self.load_mot_btn.pack(side='left', padx=5)
        
        self.load_video_btn = ctk.CTkButton(
            file_frame, 
            text="Load Video", 
            command=self.load_video_file,
            width=80
        )
        self.load_video_btn.pack(side='left', padx=5)
        
        # Playback controls
        playback_frame = ctk.CTkFrame(self.control_panel)
        playback_frame.pack(side='right', fill='y', padx=10, pady=5)
        
        self.play_btn = ctk.CTkButton(
            playback_frame, 
            text="▶️ Play", 
            command=self.toggle_play,
            width=80
        )
        self.play_btn.pack(side='left', padx=5)
        
        self.speed_var = ctk.DoubleVar(value=1.0)
        speed_frame = ctk.CTkFrame(playback_frame)
        speed_frame.pack(side='left', padx=5)
        
        ctk.CTkLabel(speed_frame, text="Speed:").pack(side='left', padx=2)
        ctk.CTkComboBox(
            speed_frame,
            values=["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"],
            command=self.set_playback_speed,
            width=70
        ).pack(side='left', padx=2)
        
        # Split window contents
        self.content_frame = ctk.CTkFrame(self.main_paned_window)
        self.content_frame.pack(fill='both', expand=True)
        
        # Left panel (70% width) - visualization of markers and/or video
        self.left_panel = ctk.CTkFrame(self.content_frame)
        self.left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Right panel (30% width) - angle plots and selection
        self.right_panel = ctk.CTkFrame(self.content_frame)
        self.right_panel.pack(side='right', fill='both', expand=False, padx=(5, 0), pady=5, ipadx=10)
        self.right_panel.configure(width=350)  # Fixed width
        
        # Add visualization elements
        self.create_visualization_panel()
        self.create_angles_panel()
        
        # Add timeline slider at bottom
        self.timeline_frame = ctk.CTkFrame(self.main_paned_window)
        self.timeline_frame.pack(fill='x', pady=(10, 0))
        
        self.frame_slider = ctk.CTkSlider(
            self.timeline_frame,
            from_=0,
            to=100,
            command=self.on_slider_change
        )
        self.frame_slider.pack(side='left', fill='x', expand=True, padx=5, pady=10)
        
        self.frame_label = ctk.CTkLabel(self.timeline_frame, text="Frame: 0/0")
        self.frame_label.pack(side='right', padx=5)
        
        # Status bar
        self.status_label = ctk.CTkLabel(
            self.main_paned_window, 
            text="Load data files to begin visualization", 
            anchor="w",
            font=("Helvetica", 11),
            text_color="gray"
        )
        self.status_label.pack(fill='x', pady=(5, 0))
    
    def create_visualization_panel(self):
        """Create the left panel for 3D visualization and/or video display"""
        # Top part: 3D markers or video
        self.viz_frame = ctk.CTkFrame(self.left_panel)
        self.viz_frame.pack(fill='both', expand=True, pady=5)
        
        # Notebook for switching between 3D view and video
        self.viz_notebook = ctk.CTkTabview(self.viz_frame)
        self.viz_notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Add tabs
        self.markers_tab = self.viz_notebook.add("3D Markers")
        self.video_tab = self.viz_notebook.add("Video")
        
        # Create marker visualization in markers tab
        self.create_marker_visualization()
        
        # Create video display in video tab
        self.create_video_display()
    
    def create_marker_visualization(self):
        """Create 3D marker visualization with Y-up orientation"""
        self.marker_fig = Figure(figsize=(8, 6), dpi=100)
        self.marker_ax = self.marker_fig.add_subplot(111, projection='3d')
        self.marker_ax.set_title('3D Marker Positions')
        self.marker_ax.set_xlabel('X')
        self.marker_ax.set_ylabel('Y (Up)')
        self.marker_ax.set_zlabel('Z (Depth)')
        
        # Set initial view angle to match Image 1
        self.marker_ax.view_init(elev=20, azim=-35)
        
        # Create canvas widget
        self.marker_canvas = FigureCanvasTkAgg(self.marker_fig, master=self.markers_tab)
        self.marker_canvas.draw()
        self.marker_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initialize empty marker data
        self.scatter = self.marker_ax.scatter([], [], [], s=30)
        
        # Add options for marker display
        self.marker_options_frame = ctk.CTkFrame(self.markers_tab)
        self.marker_options_frame.pack(fill='x', pady=5)
        
        self.connect_joints_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            self.marker_options_frame,
            text="Connect Joints",
            variable=self.connect_joints_var,
            command=self.update_marker_visualization
        ).pack(side='left', padx=10)
        
        self.show_labels_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            self.marker_options_frame,
            text="Show Labels",
            variable=self.show_labels_var,
            command=self.update_marker_visualization
        ).pack(side='left', padx=10)
        
        # Add view angle controls
        angle_frame = ctk.CTkFrame(self.marker_options_frame)
        angle_frame.pack(side='right', padx=10)
        
        ctk.CTkLabel(angle_frame, text="Elev:").pack(side='left', padx=2)
        self.elev_var = ctk.StringVar(value="20")
        elev_entry = ctk.CTkEntry(angle_frame, width=40, textvariable=self.elev_var)
        elev_entry.pack(side='left', padx=2)
        
        ctk.CTkLabel(angle_frame, text="Azim:").pack(side='left', padx=2)
        self.azim_var = ctk.StringVar(value="-35")
        azim_entry = ctk.CTkEntry(angle_frame, width=40, textvariable=self.azim_var)
        azim_entry.pack(side='left', padx=2)
        
        ctk.CTkButton(
            angle_frame,
            text="Apply",
            command=self.apply_view_angle,
            width=60
        ).pack(side='left', padx=2)
    
    def apply_view_angle(self):
        """Apply the specified view angle"""
        try:
            elev = float(self.elev_var.get())
            azim = float(self.azim_var.get())
            self.marker_ax.view_init(elev=elev, azim=azim)
            self.marker_canvas.draw()
        except ValueError:
            pass
                            
    def create_video_display(self):
        """Create video display area"""
        # Frame for video display
        self.video_display_frame = ctk.CTkFrame(self.video_tab)
        self.video_display_frame.pack(fill='both', expand=True)
        
        # Canvas for video
        self.video_canvas = tk.Canvas(self.video_display_frame, bg="black")
        self.video_canvas.pack(fill='both', expand=True)
        
        # Add a label with instructions
        self.video_label = ctk.CTkLabel(
            self.video_display_frame,
            text="Load a video using the 'Load Video' button",
            font=("Helvetica", 14)
        )
        self.video_label.place(relx=0.5, rely=0.5, anchor='center')
    
    def create_angles_panel(self):
        """Create the right panel for angle selection and plots"""
        # Create tabs for different views
        self.angles_notebook = ctk.CTkTabview(self.right_panel)
        self.angles_notebook.pack(fill='both', expand=True)
        
        # Add tabs
        self.plots_tab = self.angles_notebook.add("Plots")
        self.selection_tab = self.angles_notebook.add("Selection")
        
        # Create plots tab
        self.create_angle_plots()
        
        # Create selection tab
        self.create_angle_selection()
    
    def create_angle_plots(self):
        """Create angle plot visualizations"""
        self.angle_fig = Figure(figsize=(5, 7), dpi=100)
        
        # Create a single plot that will show selected angles
        self.angle_ax = self.angle_fig.add_subplot(111)
        self.angle_ax.set_title('Joint Angles')
        self.angle_ax.set_xlabel('Time (s)')
        self.angle_ax.set_ylabel('Angle (degrees)')
        self.angle_ax.grid(True, linestyle='--', alpha=0.7)
        
        self.angle_fig.tight_layout()
        
        # Create canvas widget
        self.angle_canvas = FigureCanvasTkAgg(self.angle_fig, master=self.plots_tab)
        self.angle_canvas.draw()
        self.angle_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Current time indicator
        self.time_line = None
    
    def create_angle_selection(self):
        """Create UI for selecting angles to plot"""
        # Create scrollable frame for angle selection
        self.selection_frame = ctk.CTkScrollableFrame(self.selection_tab)
        self.selection_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Add a message when no data is loaded
        self.no_data_label = ctk.CTkLabel(
            self.selection_frame,
            text="Load a MOT file to view available angles",
            font=("Helvetica", 12),
            text_color="gray"
        )
        self.no_data_label.pack(pady=20)
        
        # Add buttons at the bottom
        self.selection_buttons_frame = ctk.CTkFrame(self.selection_tab)
        self.selection_buttons_frame.pack(fill='x', pady=5)
        
        self.select_all_btn = ctk.CTkButton(
            self.selection_buttons_frame,
            text="Select All",
            command=self.select_all_angles,
            width=90,
            state="disabled"
        )
        self.select_all_btn.pack(side='left', padx=5)
        
        self.deselect_all_btn = ctk.CTkButton(
            self.selection_buttons_frame,
            text="Deselect All",
            command=self.deselect_all_angles,
            width=90,
            state="disabled"
        )
        self.deselect_all_btn.pack(side='left', padx=5)
        
        self.apply_selection_btn = ctk.CTkButton(
            self.selection_buttons_frame,
            text="Apply Selection",
            command=self.apply_angle_selection,
            width=110,
            state="disabled"
        )
        self.apply_selection_btn.pack(side='right', padx=5)
    
    def auto_detect_files(self):
        """Auto-detect TRC and MOT files"""
        try:
            # Determine file paths based on application mode
            self.update_status("Looking for data files...", "blue")
            
            trc_file = None
            mot_file = None
            video_file = None
            
            if self.app.analysis_mode == '2d':
                # For 2D analysis, look for *Sports2D folder
                search_path = self.app.participant_name
                sports2d_folders = []
                
                for root, dirs, _ in os.walk(search_path):
                    for dir_name in dirs:
                        if dir_name.endswith("Sports2D"):
                            sports2d_folders.append(os.path.join(root, dir_name))
                
                if sports2d_folders:
                    folder_path = sports2d_folders[0]
                    
                    # Find TRC files
                    trc_files = [f for f in os.listdir(folder_path) if f.endswith('.trc')]
                    non_lstm_trc = [f for f in trc_files if "LSTM" not in f]
                    
                    if non_lstm_trc:
                        trc_file = os.path.join(folder_path, non_lstm_trc[0])
                    elif trc_files:
                        trc_file = os.path.join(folder_path, trc_files[0])
                    
                    # Find MOT files
                    mot_files = [f for f in os.listdir(folder_path) if f.endswith('.mot')]
                    non_lstm_mot = [f for f in mot_files if "LSTM" not in f]
                    
                    if non_lstm_mot:
                        mot_file = os.path.join(folder_path, non_lstm_mot[0])
                    elif mot_files:
                        mot_file = os.path.join(folder_path, mot_files[0])
                    
                    # Look for video files
                    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
                    if video_files:
                        video_file = os.path.join(folder_path, video_files[0])
            else:
                # For 3D analysis
                search_path = self.app.participant_name
                
                # Look for pose-3d.trc
                potential_trc = os.path.join(search_path, 'pose-3d.trc')
                if os.path.exists(potential_trc):
                    trc_file = potential_trc
                
                # Look in kinematics folder for MOT files
                kinematics_path = os.path.join(search_path, 'kinematics')
                if os.path.exists(kinematics_path):
                    mot_files = [f for f in os.listdir(kinematics_path) if f.endswith('.mot')]
                    if mot_files:
                        mot_file = os.path.join(kinematics_path, mot_files[0])
                
                # Look for videos
                videos_path = os.path.join(search_path, 'videos')
                if os.path.exists(videos_path):
                    video_files = [f for f in os.listdir(videos_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
                    if video_files:
                        video_file = os.path.join(videos_path, video_files[0])
            
            # Load the files if found
            files_found = False
            
            if trc_file:
                self.load_trc_data(trc_file)
                files_found = True
            
            if mot_file:
                self.load_mot_data(mot_file)
                files_found = True
            
            if video_file:
                self.load_video(video_file)
                files_found = True
            
            if not files_found:
                self.update_status("No data files found. Try loading files manually.", "orange")
            else:
                self.update_status("Data files loaded successfully.", "green")
        
        except Exception as e:
            self.update_status(f"Error auto-detecting files: {str(e)}", "red")
    
    def load_trc_file(self):
        """Open file dialog to load TRC file"""
        file_path = filedialog.askopenfilename(
            title="Select TRC File",
            filetypes=[("TRC Files", "*.trc"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.load_trc_data(file_path)
    
    def load_mot_file(self):
        """Open file dialog to load MOT file"""
        file_path = filedialog.askopenfilename(
            title="Select MOT File",
            filetypes=[("MOT Files", "*.mot"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.load_mot_data(file_path)
    
    def load_video_file(self):
        """Open file dialog to load video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_trc_data(self, file_path):
        """Parse and load TRC file"""
        try:
            self.update_status(f"Loading TRC file: {os.path.basename(file_path)}...", "blue")
            
            with open(file_path, 'r') as f:
                content = f.readlines()
            
            # First find the header lines
            data_rate_header_idx = -1
            for i, line in enumerate(content):
                if "DataRate" in line and "CameraRate" in line:
                    data_rate_header_idx = i
                    break
            
            if data_rate_header_idx == -1:
                raise ValueError("Invalid TRC file format: DataRate header line not found")
            
            # Get values from the line after the header
            values_line_idx = data_rate_header_idx + 1
            if values_line_idx >= len(content):
                raise ValueError("Invalid TRC file format: Values line missing")
                
            values_line = content[values_line_idx].strip().split('\t')
            if len(values_line) < 4:
                raise ValueError(f"Invalid values line format: {content[values_line_idx]}")
            
            # Extract values from values line
            frame_rate = float(values_line[0])
            num_frames = int(values_line[2])
            num_markers = int(values_line[3])
            
            # Find column headers (marker names) - usually 2 lines after values line
            marker_line_idx = values_line_idx + 2
            if marker_line_idx >= len(content):
                raise ValueError("Invalid TRC file format: Marker names line not found")
                
            marker_names_line = content[marker_line_idx].strip().split('\t')
            
            # Process marker names (removing duplicates from X/Y/Z components)
            marker_names = []
            i = 2  # Start after Frame# and Time
            while i < len(marker_names_line):
                if marker_names_line[i]:
                    # Remove X/Y/Z suffix if present
                    name = marker_names_line[i].split(':')[0]  # Handle "MarkerName:X" format
                    marker_names.append(name)
                    i += 3  # Skip the X/Y/Z columns for this marker
                else:
                    i += 1  # Skip empty column
            
            # Parse data
            data_start_idx = marker_line_idx + 2  # Skip marker names line and coordinate headers
            
            frames_data = []
            for i in range(data_start_idx, len(content)):
                line = content[i].strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 4:  # Need at least frame, time, and one coordinate
                    continue
                
                try:
                    frame_num = int(float(parts[0]))
                    time_val = float(parts[1])
                    
                    # Process marker data
                    markers = {}
                    marker_idx = 0
                    
                    for j in range(len(marker_names)):
                        # Each marker has 3 values (X, Y, Z)
                        col_offset = 2 + j*3
                        
                        # Check if within bounds
                        if col_offset + 2 < len(parts):
                            x_str = parts[col_offset].strip()
                            y_str = parts[col_offset + 1].strip()
                            z_str = parts[col_offset + 2].strip()
                            
                            try:
                                x = float(x_str) if x_str else float('nan')
                                y = float(y_str) if y_str else float('nan')
                                z = float(z_str) if z_str else float('nan')
                                
                                markers[marker_names[j]] = {'x': x, 'y': y, 'z': z}
                                marker_idx += 1
                            except ValueError:
                                # Skip invalid values
                                pass
                    
                    frames_data.append({
                        'frame': frame_num,
                        'time': time_val,
                        'markers': markers
                    })
                    
                except (ValueError, IndexError) as e:
                    # Skip invalid lines
                    print(f"Error parsing line {i}: {e}")
                    continue
            
            # Store data
            self.trc_data = {
                'file_path': file_path,
                'marker_names': marker_names,
                'num_frames': num_frames,
                'frames': frames_data
            }
            
            # Update slider range
            max_frame = len(frames_data) - 1
            self.frame_slider.configure(to=max_frame)
            self.frame_slider.set(0)
            self.current_frame = 0
            self.frame_label.configure(text=f"Frame: 1/{len(frames_data)}")
            
            # Update visualization
            self.update_marker_visualization()
            
            # Switch to 3D Markers tab
            self.viz_notebook.set("3D Markers")
            
            self.update_status(f"TRC file loaded: {os.path.basename(file_path)} ({len(marker_names)} markers, {len(frames_data)} frames)", "green")
            
        except Exception as e:
            self.update_status(f"Error loading TRC file: {str(e)}", "red")
            import traceback
            traceback.print_exc()
    
    def load_mot_data(self, file_path):
        """Parse and load MOT file"""
        try:
            self.update_status(f"Loading MOT file: {os.path.basename(file_path)}...", "blue")
            
            with open(file_path, 'r') as f:
                content = f.readlines()
            
            # Find endheader line
            header_end_idx = -1
            for i, line in enumerate(content):
                if "endheader" in line.lower():
                    header_end_idx = i
                    break
            
            if header_end_idx == -1:
                # Try alternate format (look for line starting with a number)
                for i, line in enumerate(content):
                    if line.strip() and line[0].isdigit():
                        header_end_idx = i - 1
                        break
            
            if header_end_idx == -1:
                raise ValueError("Could not find header end in MOT file")
            
            # Get column headers
            header_line = content[header_end_idx + 1].strip()
            headers = header_line.split()
            
            # Parse data
            frames_data = []
            for i in range(header_end_idx + 2, len(content)):
                line = content[i].strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:  # Need at least time and one value
                    continue
                
                try:
                    time_val = float(parts[0])
                    
                    # Process angle data
                    angles = {}
                    for j in range(1, min(len(headers), len(parts))):
                        try:
                            value = float(parts[j]) if parts[j].strip() else float('nan')
                            angles[headers[j]] = value
                        except ValueError:
                            # Skip invalid values
                            pass
                    
                    frames_data.append({
                        'time': time_val,
                        'angles': angles
                    })
                    
                except (ValueError, IndexError):
                    # Skip invalid lines
                    continue
            
            # Store data
            self.mot_data = {
                'file_path': file_path,
                'headers': headers[1:],  # Skip 'time' column
                'frames': frames_data
            }
            
            # Update angle selection UI
            self.update_angle_selection()
            
            # Switch to Selection tab in right panel
            self.angles_notebook.set("Selection")
            
            self.update_status(f"MOT file loaded: {os.path.basename(file_path)} ({len(headers)-1} angles, {len(frames_data)} frames)", "green")
            
        except Exception as e:
            self.update_status(f"Error loading MOT file: {str(e)}", "red")
    
    def load_video(self, file_path):
        """Load video file"""
        try:
            # Close any previously open video
            if self.video_cap is not None:
                self.video_cap.release()
            
            # Open the video file
            self.video_cap = cv2.VideoCapture(file_path)
            
            if not self.video_cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.video_path = file_path
            
            # Update UI
            self.video_label.place_forget()  # Hide the instruction label
            
            # Switch to Video tab
            self.viz_notebook.set("Video")
            
            # Show first frame
            self.update_video_frame()
            
            self.update_status(f"Video loaded: {os.path.basename(file_path)} ({width}x{height}, {fps:.1f} fps, {total_frames} frames)", "green")
            
        except Exception as e:
            self.update_status(f"Error loading video: {str(e)}", "red")
    
    def update_marker_visualization(self):
        """Update 3D marker visualization with Y-up coordinate system"""
        if not self.trc_data or self.current_frame >= len(self.trc_data['frames']):
            return
        
        # Clear existing plot
        self.marker_ax.clear()
        
        # Get frame data
        frame_data = self.trc_data['frames'][self.current_frame]
        markers = frame_data['markers']
        
        # Prepare coordinates - SWAPPING Y AND Z CORRECTLY
        xs, ys, zs = [], [], []
        names = []
        
        for name, coords in markers.items():
            if not np.isnan(coords['x']) and not np.isnan(coords['y']) and not np.isnan(coords['z']):
                # Correct mapping with Y and Z swapped
                xs.append(coords['x'])     # X stays as X
                ys.append(coords['z'])     # Z becomes Y (up)
                zs.append(coords['y'])     # Y becomes Z (depth)
                names.append(name)
        
        # Plot markers
        self.marker_ax.scatter(xs, ys, zs, c='blue', s=40)
        
        # Add marker labels if enabled
        if self.show_labels_var.get():
            for i, (x, y, z, name) in enumerate(zip(xs, ys, zs, names)):
                self.marker_ax.text(x, y, z, name, size=8, zorder=1, color='black')
        
        # Connect joints if enabled
        if self.connect_joints_var.get():
            # Define connections between markers
            connections = {
                'Hip': ['RHip', 'LHip', 'Neck'],
                'RHip': ['RKnee'],
                'RKnee': ['RAnkle'],
                'RAnkle': ['RHeel', 'RBigToe'],
                'RBigToe': ['RSmallToe'],
                'LHip': ['LKnee'],
                'LKnee': ['LAnkle'],
                'LAnkle': ['LHeel', 'LBigToe'],
                'LBigToe': ['LSmallToe'],
                'Neck': ['Head', 'RShoulder', 'LShoulder'],
                'Head': ['Nose'],
                'RShoulder': ['RElbow'],
                'RElbow': ['RWrist'],
                'LShoulder': ['LElbow'],
                'LElbow': ['LWrist']
            }
            
            marker_dict = {name: (x, y, z) for name, x, y, z in zip(names, xs, ys, zs)}
            
            for start, ends in connections.items():
                if start in marker_dict:
                    start_coords = marker_dict[start]
                    for end in ends:
                        if end in marker_dict:
                            end_coords = marker_dict[end]
                            self.marker_ax.plot(
                                [start_coords[0], end_coords[0]],
                                [start_coords[1], end_coords[1]],
                                [start_coords[2], end_coords[2]],
                                'k-', linewidth=1
                            )
        
        # Set axis properties
        x_range = max(xs) - min(xs) if xs else 1
        y_range = max(ys) - min(ys) if ys else 1
        z_range = max(zs) - min(zs) if zs else 1
        
        # Find center point
        x_center = (max(xs) + min(xs)) / 2 if xs else 0
        y_center = (max(ys) + min(ys)) / 2 if ys else 0
        z_center = (max(zs) + min(zs)) / 2 if zs else 0
        
        # Set equal aspect ratio
        max_range = max(x_range, y_range, z_range) * 0.6
        
        self.marker_ax.set_xlim(x_center - max_range, x_center + max_range)
        self.marker_ax.set_ylim(y_center - max_range, y_center + max_range)
        self.marker_ax.set_zlim(z_center - max_range, z_center + max_range)
        
        # Set labels
        self.marker_ax.set_xlabel('X')
        self.marker_ax.set_ylabel('Y (Up)')
        self.marker_ax.set_zlabel('Z (Depth)')
        self.marker_ax.set_title(f'3D Markers - Frame {self.current_frame+1}')
        
        # Set view angle to match Image 1
        self.marker_ax.view_init(elev=20, azim=-35)
        
        # Redraw
        self.marker_canvas.draw()
                                
    def update_video_frame(self):
        """Update video display with current frame"""
        if self.video_cap is None:
            return
            
        # Seek to the current frame
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        # Read the frame
        ret, frame = self.video_cap.read()
        
        if not ret:
            self.update_status("Failed to read video frame", "red")
            return
            
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get canvas dimensions
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:  # Canvas not yet realized
            # Set default size
            canvas_width = 640
            canvas_height = 480
        
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
    
    def update_angle_selection(self):
        """Update angle selection UI based on loaded MOT data"""
        if not self.mot_data:
            return
        
        # Clear existing widgets
        for widget in self.selection_frame.winfo_children():
            widget.destroy()
        
        # Get angle headers
        angle_headers = self.mot_data['headers']
        
        if not angle_headers:
            ctk.CTkLabel(
                self.selection_frame, 
                text="No angles found in MOT file",
                font=("Helvetica", 12),
                text_color="gray"
            ).pack(pady=20)
            return
        
        # Create variables for checkboxes
        self.angle_vars = {}
        
        # Group similar angles
        angle_groups = {
            "Lower Limbs": [a for a in angle_headers if any(s in a.lower() for s in 
                           ['ankle', 'knee', 'hip', 'foot', 'toe', 'heel', 'thigh', 'shank'])],
            "Upper Limbs": [a for a in angle_headers if any(s in a.lower() for s in 
                           ['shoulder', 'arm', 'elbow', 'wrist', 'forearm', 'sup'])],
            "Trunk & Spine": [a for a in angle_headers if any(s in a.lower() for s in 
                             ['trunk', 'pelvis', 'lumbar', 'thorax', 'neck', 'head', 'spine', 'l1', 'l2', 'l3', 'l4', 'l5'])],
            "Other": []  # Will catch anything not categorized above
        }
        
        # Add uncategorized angles to "Other"
        for angle in angle_headers:
            if not any(angle in group for group in angle_groups.values()):
                angle_groups["Other"].append(angle)
        
        # Create section for each group
        for group_name, angles in angle_groups.items():
            if not angles:
                continue
                
            # Create group frame
            group_frame = ctk.CTkFrame(self.selection_frame)
            group_frame.pack(fill='x', pady=5, padx=2)
            
            # Group header
            ctk.CTkLabel(
                group_frame,
                text=group_name,
                font=("Helvetica", 12, "bold")
            ).pack(anchor='w', padx=5, pady=5)
            
            # Create checkboxes for all angles in this group
            for angle in angles:
                var = ctk.BooleanVar(value=False)
                self.angle_vars[angle] = var
                
                ctk.CTkCheckBox(
                    group_frame,
                    text=angle,
                    variable=var
                ).pack(anchor='w', padx=20, pady=2)
        
        # Enable selection buttons
        self.select_all_btn.configure(state="normal")
        self.deselect_all_btn.configure(state="normal")
        self.apply_selection_btn.configure(state="normal")
    
    def select_all_angles(self):
        """Select all angles"""
        if hasattr(self, 'angle_vars'):
            for var in self.angle_vars.values():
                var.set(True)
    
    def deselect_all_angles(self):
        """Deselect all angles"""
        if hasattr(self, 'angle_vars'):
            for var in self.angle_vars.values():
                var.set(False)
    
    def apply_angle_selection(self):
        """Apply the current angle selection to the plot"""
        if not hasattr(self, 'angle_vars') or not self.mot_data:
            return
        
        # Get selected angles
        self.selected_angles = [angle for angle, var in self.angle_vars.items() if var.get()]
        
        if not self.selected_angles:
            messagebox.showinfo("Selection Empty", "Please select at least one angle to plot")
            return
        
        # Update angle plot
        self.update_angle_plot()
        
        # Update time indicator if TRC data is loaded
        if self.trc_data and self.current_frame < len(self.trc_data['frames']):
            current_time = self.trc_data['frames'][self.current_frame]['time']
            self.update_time_indicator(current_time)
        
        # Switch to Plots tab
        self.angles_notebook.set("Plots")
    
    def update_angle_plot(self):
        """Update angle plot with selected angles"""
        if not self.mot_data or not self.selected_angles:
            return
        
        # Clear existing plot
        self.angle_ax.clear()
        
        # Get time values
        time_values = [frame['time'] for frame in self.mot_data['frames']]
        
        # Plot selected angles
        for angle in self.selected_angles:
            angle_values = [frame['angles'].get(angle, float('nan')) for frame in self.mot_data['frames']]
            self.angle_ax.plot(time_values, angle_values, label=angle)
        
        # Add vertical line for current time if data available
        if self.trc_data and self.current_frame < len(self.trc_data['frames']):
            current_time = self.trc_data['frames'][self.current_frame]['time']
            # Add or update vertical line to show current time
            self.time_line = self.angle_ax.axvline(x=current_time, color='red', linestyle='--', linewidth=2)
        
        # Set labels and title
        self.angle_ax.set_xlabel('Time (s)')
        self.angle_ax.set_ylabel('Angle (degrees)')
        self.angle_ax.set_title('Joint Angles')
        self.angle_ax.grid(True, linestyle='--', alpha=0.7)
        self.angle_ax.legend(loc='best', fontsize='small')
        
        # Adjust layout
        self.angle_fig.tight_layout()
        
        # Redraw
        self.angle_canvas.draw()
    
    def on_slider_change(self, value):
        """Handle slider position change"""
        if not self.trc_data and not self.video_cap:
            return
        
        # Get the frame index
        frame_index = int(float(value))
        
        # Update current frame
        self.current_frame = frame_index
        
        # Update frame label
        max_frames = 0
        if self.trc_data:
            max_frames = len(self.trc_data['frames'])
        elif self.video_cap:
            max_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.frame_label.configure(text=f"Frame: {frame_index+1}/{max_frames}")
        
        # Update visualizations
        if self.trc_data:
            self.update_marker_visualization()
        
        if self.video_cap:
            self.update_video_frame()
        
        # Update time indicator in angle plot
        if self.trc_data and self.mot_data and self.selected_angles:
            current_time = self.trc_data['frames'][self.current_frame]['time']
            self.update_time_indicator(current_time)
    
    def update_time_indicator(self, current_time):
        """Update the time indicator line in the angle plot"""
        if hasattr(self, 'angle_ax') and self.selected_angles:
            # Remove existing time line if it exists
            if self.time_line:
                try:
                    self.time_line.remove()
                except:
                    pass
            
            # Add new time line
            self.time_line = self.angle_ax.axvline(x=current_time, color='red', linestyle='--', linewidth=2)
            
            # Redraw the canvas
            self.angle_canvas.draw()
    
    def toggle_play(self):
        """Toggle playback of animation"""
        self.playing = not self.playing
        
        if self.playing:
            self.play_btn.configure(text="⏸ Pause")
            self.play_animation()
        else:
            self.play_btn.configure(text="▶️ Play")
            # Cancel scheduled animation
            if self.play_after_id:
                self.frame.after_cancel(self.play_after_id)
                self.play_after_id = None
    
    def play_animation(self):
        """Play animation frame by frame"""
        if not self.playing:
            return
        
        # Determine max frames
        max_frames = 0
        if self.trc_data:
            max_frames = len(self.trc_data['frames'])
        elif self.video_cap:
            max_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames <= 0:
            self.playing = False
            self.play_btn.configure(text="▶️ Play")
            return
        
        # Advance to next frame
        next_frame = (self.current_frame + 1) % max_frames
        
        # Update slider position (will trigger visualization update)
        self.frame_slider.set(next_frame)
        self.on_slider_change(next_frame)
        
        # Calculate frame delay based on speed setting
        speed = self.speed_var.get()
        
        # Determine frame rate
        fps = 30  # Default
        if self.video_cap:
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate delay in milliseconds
        delay = int(1000 / (fps * speed))
        
        # Schedule next frame
        self.play_after_id = self.frame.after(delay, self.play_animation)
    
    def set_playback_speed(self, speed_text):
        """Set playback speed from combo box selection"""
        speed = float(speed_text.replace('x', ''))
        self.speed_var.set(speed)
    
    def update_status(self, message, color="black"):
        """Update status message"""
        self.status_label.configure(text=message, text_color=color)