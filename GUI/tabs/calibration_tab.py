from pathlib import Path
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")  # Ensure we're using TkAgg backend

from GUI.utils import generate_checkerboard_image

class CalibrationTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        # Create main frame
        self.frame = ctk.CTkFrame(parent)
        
        # Initialize variables
        self.calibration_type_var = ctk.StringVar(value='calculate')
        self.num_cameras_var = ctk.StringVar(value='2')
        self.checkerboard_width_var = ctk.StringVar(value='7')
        self.checkerboard_height_var = ctk.StringVar(value='5')
        self.square_size_var = ctk.StringVar(value='30')
        self.video_extension_var = ctk.StringVar(value='mp4')
        self.convert_from_var = ctk.StringVar(value='qualisys')
        self.binning_factor_var = ctk.StringVar(value='1')
        
        # Track configuration state
        self.type_confirmed = False
        self.points_2d = []
        self.point_markers = []
        self.object_coords_3d = []
        self.current_point_index = 0
        
        # Flag to control click handling during zooming
        self.zooming_mode = False
        
        # Build the UI
        self.build_ui()

    def get_settings(self):
        """Get the calibration settings"""
        settings = {
            'calibration': {
                'calibration_type': self.calibration_type_var.get(),
            }
        }
        
        # Add type-specific settings
        if self.calibration_type_var.get() == 'calculate':
            settings['calibration']['calculate'] = {
                'intrinsics': {
                    'intrinsics_corners_nb': [
                        int(self.checkerboard_width_var.get()),
                        int(self.checkerboard_height_var.get())
                    ],
                    'intrinsics_square_size': float(self.square_size_var.get()),
                    'intrinsics_extension': self.video_extension_var.get()
                },
                'extrinsics': {
                    'scene': {
                        'extrinsics_extension': self.video_extension_var.get()
                    }
                }
            }
            
            # Add coordinates if they've been set
            if hasattr(self, 'object_coords_3d') and self.object_coords_3d:
                settings['calibration']['calculate']['extrinsics']['scene']['object_coords_3d'] = self.object_coords_3d
        else:
            settings['calibration']['convert'] = {
                'convert_from': self.convert_from_var.get()
            }
            
            if self.convert_from_var.get() == 'qualisys':
                settings['calibration']['convert']['qualisys'] = {
                    'binning_factor': int(self.binning_factor_var.get())
                }
        
        return settings
    
    def build_ui(self):
        # Create a two-panel layout
        self.main_panel = ctk.CTkFrame(self.frame)
        self.main_panel.pack(fill='both', expand=True, padx=0, pady=0)

        self.title_label = ctk.CTkLabel(
            self.main_panel,
            text=self.app.lang_manager.get_text('calibration_tab'),
            font=("Helvetica", 24, "bold")
        )
        self.title_label.pack(pady=(0, 20))
        
        # Left panel (for inputs)
        self.left_panel = ctk.CTkFrame(self.main_panel, width=600)
        self.left_panel.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Right panel (for scene image)
        self.right_panel = ctk.CTkFrame(self.main_panel)
        self.right_panel.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Add content to the left panel
        self.build_left_panel()
        
        # Right panel will be populated with scene image when needed
        ctk.CTkLabel(
            self.right_panel,
            text="Scene Calibration Image will appear here",
            wraplength=300,
            font=("Helvetica", 14)
        ).pack(expand=True)
    
    def build_left_panel(self):
        # Calibration Type
        type_frame = ctk.CTkFrame(self.left_panel)
        type_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            type_frame, 
            text="Calibration Type:",
            width=150
        ).pack(side='left', padx=10, pady=10)
        
        # Radio buttons for calibration type
        radio_frame = ctk.CTkFrame(type_frame, fg_color="transparent")
        radio_frame.pack(side='left', fill='x', expand=True)
        
        ctk.CTkRadioButton(
            radio_frame, 
            text="Calculate", 
            variable=self.calibration_type_var,
            value='calculate',
            command=self.on_calibration_type_change
        ).pack(side='left', padx=10)
        
        ctk.CTkRadioButton(
            radio_frame,
            text="Convert",
            variable=self.calibration_type_var,
            value='convert',
            command=self.on_calibration_type_change
        ).pack(side='left', padx=10)
        
        # Number of Cameras
        camera_frame = ctk.CTkFrame(self.left_panel)
        camera_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            camera_frame,
            text="Number of Cameras:",
            width=150
        ).pack(side='left', padx=10, pady=10)
        
        self.camera_entry = ctk.CTkEntry(
            camera_frame,
            textvariable=self.num_cameras_var,
            width=100
        )
        self.camera_entry.pack(side='left', padx=10)
        
        # Calculate Options Frame
        self.calculate_frame = ctk.CTkFrame(self.left_panel)
        self.calculate_frame.pack(fill='x', pady=5)
        
        # Checkerboard Width
        width_frame = ctk.CTkFrame(self.calculate_frame)
        width_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            width_frame,
            text="Checkerboard Width:",
            width=150
        ).pack(side='left', padx=10, pady=5)
        
        self.width_entry = ctk.CTkEntry(
            width_frame,
            textvariable=self.checkerboard_width_var,
            width=100
        )
        self.width_entry.pack(side='left', padx=10)
        
        # Checkerboard Height
        height_frame = ctk.CTkFrame(self.calculate_frame)
        height_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            height_frame,
            text="Checkerboard Height:",
            width=150
        ).pack(side='left', padx=10, pady=5)
        
        self.height_entry = ctk.CTkEntry(
            height_frame,
            textvariable=self.checkerboard_height_var,
            width=100
        )
        self.height_entry.pack(side='left', padx=10)
        
        # Square Size
        square_frame = ctk.CTkFrame(self.calculate_frame)
        square_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            square_frame,
            text="Square Size (mm):",
            width=150
        ).pack(side='left', padx=10, pady=5)
        
        self.square_entry = ctk.CTkEntry(
            square_frame,
            textvariable=self.square_size_var,
            width=100
        )
        self.square_entry.pack(side='left', padx=10)
        
        # Video Extension
        extension_frame = ctk.CTkFrame(self.calculate_frame)
        extension_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            extension_frame,
            text="Video/Image Extension:",
            width=150
        ).pack(side='left', padx=10, pady=5)
        
        self.extension_entry = ctk.CTkEntry(
            extension_frame,
            textvariable=self.video_extension_var,
            width=100
        )
        self.extension_entry.pack(side='left', padx=10)
        
        # Checkerboard preview (placed at the bottom of inputs)
        self.checkerboard_frame = ctk.CTkFrame(self.left_panel)
        self.checkerboard_frame.pack(fill='x', pady=10)
        
        # Convert Options Frame (initially hidden)
        self.convert_frame = ctk.CTkFrame(self.left_panel)
        
        # Convert From
        convert_from_frame = ctk.CTkFrame(self.convert_frame)
        convert_from_frame.pack(fill='x', pady=5)
        
        ctk.CTkLabel(
            convert_from_frame,
            text="Convert From:",
            width=150
        ).pack(side='left', padx=10, pady=5)
        
        convert_options = ['qualisys', 'optitrack', 'vicon', 'opencap', 'easymocap', 'biocv', 'anipose', 'freemocap']
        self.convert_menu = ctk.CTkOptionMenu(
            convert_from_frame,
            variable=self.convert_from_var,
            values=convert_options,
            width=150
        )
        self.convert_menu.pack(side='left', padx=10)
        
        # Binning Factor (for Qualisys)
        self.qualisys_frame = ctk.CTkFrame(self.convert_frame)
        
        ctk.CTkLabel(
            self.qualisys_frame,
            text="Binning Factor:",
            width=150
        ).pack(side='left', padx=10, pady=5)
        
        ctk.CTkEntry(
            self.qualisys_frame,
            textvariable=self.binning_factor_var,
            width=100
        ).pack(side='left', padx=10)
        
        # Confirm button
        self.confirm_button = ctk.CTkButton(
            self.left_panel,
            text="Confirm Configuration",
            command=self.confirm_calibration_type,
            height=40,
            width=200,
            font=("Helvetica", 14),
            fg_color=("#4CAF50", "#2E7D32")
        )
        self.confirm_button.pack(side='bottom', pady=10)
        
        # Proceed button (initially hidden)
        self.proceed_button = ctk.CTkButton(
            self.left_panel,
            text="Proceed with Calibration",
            command=self.proceed_calibration,
            height=40,
            width=200
        )
        
        # Points selection frame (for scene calibration) - will be shown in right panel
        self.points_frame = ctk.CTkFrame(self.right_panel)
        
        # Apply the current calibration type
        self.on_calibration_type_change()
    
    def on_calibration_type_change(self):
        """Handle changes to calibration type"""
        # If already confirmed, ask for reconfirmation
        if self.type_confirmed:
            response = messagebox.askyesno(
                "Confirm Changes",
                "Do you want to modify the configuration? This will require reconfirmation."
            )
            if response:
                # Re-enable inputs for modification
                self.camera_entry.configure(state='normal')
                if self.calibration_type_var.get() == 'calculate':
                    self.width_entry.configure(state='normal')
                    self.height_entry.configure(state='normal')
                    self.square_entry.configure(state='normal')
                    self.extension_entry.configure(state='normal')
                else:
                    self.convert_menu.configure(state='normal')
                
                # Show confirm button, hide proceed button
                self.confirm_button.pack(pady=10)
                self.proceed_button.pack_forget()
                
                # Reset confirmation flag
                self.type_confirmed = False
            else:
                # Revert radio button selection
                self.calibration_type_var.set('convert' if self.calibration_type_var.get() == 'calculate' else 'calculate')
                return
        
        # Show/hide appropriate frames
        if self.calibration_type_var.get() == 'calculate':
            self.calculate_frame.pack(fill='x', pady=5)
            self.convert_frame.pack_forget()
            self.qualisys_frame.pack_forget()
        else:
            self.calculate_frame.pack_forget()
            self.convert_frame.pack(fill='x', pady=5)
            
            # Show/hide Qualisys-specific settings
            if self.convert_from_var.get() == 'qualisys':
                self.qualisys_frame.pack(fill='x', pady=5)
            else:
                self.qualisys_frame.pack_forget()
    
    def confirm_calibration_type(self):
        """Confirm the calibration type configuration"""
        try:
            # Validate number of cameras
            num_cameras = int(self.num_cameras_var.get())
            if num_cameras < 2:
                messagebox.showerror(
                    "Error",
                    "Number of cameras must be at least 2"
                )
                return
            
            # Validate calculate-specific inputs
            if self.calibration_type_var.get() == 'calculate':
                if not all([
                    self.checkerboard_width_var.get(),
                    self.checkerboard_height_var.get(),
                    self.square_size_var.get(),
                    self.video_extension_var.get()
                ]):
                    messagebox.showerror("Error", "All fields must be filled")
                    return
                
                # Generate and display checkerboard preview
                checkerboard_width = int(self.checkerboard_width_var.get())
                checkerboard_height = int(self.checkerboard_height_var.get())
                square_size = float(self.square_size_var.get())
                
                self.display_checkerboard(checkerboard_width, checkerboard_height, square_size)
            
            # Disable inputs
            self.camera_entry.configure(state='disabled')
            if self.calibration_type_var.get() == 'calculate':
                self.width_entry.configure(state='disabled')
                self.height_entry.configure(state='disabled')
                self.square_entry.configure(state='disabled')
                self.extension_entry.configure(state='disabled')
            else:
                self.convert_menu.configure(state='disabled')
            
            # Update buttons
            self.confirm_button.pack_forget()
            self.proceed_button.pack(pady=10)
            
            # Set confirmed flag
            self.type_confirmed = True
            
            messagebox.showinfo(
                "Configuration Confirmed",
                "Calibration configuration confirmed. Click 'Proceed with Calibration' when ready."
            )
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")
    
    def display_checkerboard(self, width, height, square_size):
        """Display a checkerboard preview"""
        # Clear existing widgets
        for widget in self.checkerboard_frame.winfo_children():
            widget.destroy()
        
        # Generate checkerboard image
        checkerboard_image = generate_checkerboard_image(width, height, square_size)
        
        # Checkerboard preview title
        ctk.CTkLabel(
            self.checkerboard_frame,
            text="Checkerboard Preview:",
            font=("Helvetica", 16, "bold")
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        # Resize for display
        max_size = 200
        img_width, img_height = checkerboard_image.size
        scale = min(max_size / img_width, max_size / img_height, 1)
        display_image = checkerboard_image.resize(
            (int(img_width * scale), int(img_height * scale)),
            Image.Resampling.LANCZOS
        )
        
        # Convert to CTkImage
        ctk_img = ctk.CTkImage(
            light_image=display_image,
            dark_image=display_image,
            size=(int(img_width * scale), int(img_height * scale))
        )
        
        # Display checkerboard
        image_label = ctk.CTkLabel(self.checkerboard_frame, image=ctk_img, text="")
        image_label.ctk_image = ctk_img  # Keep a reference
        image_label.pack(padx=10, pady=10)
        
        # Save button
        ctk.CTkButton(
            self.checkerboard_frame,
            text="Save as PDF",
            command=lambda: self.save_checkerboard_as_pdf(checkerboard_image)
        ).pack(pady=(0, 10))
    
    def save_checkerboard_as_pdf(self, image):
        """Save the checkerboard image as a PDF file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        if file_path:
            image.save(file_path, "PDF")
            messagebox.showinfo(
                "Saved",
                f"Checkerboard image saved as {file_path}"
            )
    
    def proceed_calibration(self):
        """Proceed with calibration setup"""
        if not self.type_confirmed:
            messagebox.showerror("Error", "Please confirm your configuration first")
            return
        
        # Get number of cameras
        try:
            num_cameras = int(self.num_cameras_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number of cameras")
            return
        
        # Process based on calibration type
        if self.calibration_type_var.get() == 'calculate':
            # Create calibration folders
            self.create_calibration_folders(num_cameras)
            
            # Input checkerboard videos
            if not self.input_checkerboard_videos(num_cameras):
                return
            
            # Input scene videos
            if not self.input_scene_videos(num_cameras):
                return
            
            # Input scene coordinates
            if not self.input_scene_coordinates():
                return
            
        else:  # convert
            # Input calibration file to convert
            if not self.input_calibration_file():
                return
        
        # Update the progress only after all coordinates are entered in the input_scene_coordinates method
    
    def create_calibration_folders(self, num_cameras):
        """Create the necessary calibration folders"""
        # Define base path based on analysis mode
        base_path = Path(self.app.participant_name) / 'calibration'
        
        # Create folders for each camera
        for cam in range(1, num_cameras + 1):
            intrinsics_folder = base_path / 'intrinsics' / f'int_cam{cam}_img'
            extrinsics_folder = base_path / 'extrinsics' / f'ext_cam{cam}_img'
            
            # Create directories
            intrinsics_folder.mkdir(parents=True, exist_ok=True)
            extrinsics_folder.mkdir(parents=True, exist_ok=True)
    
    def input_checkerboard_videos(self, num_cameras):
        """Input checkerboard videos/images for each camera"""
        messagebox.showinfo(
            "Input Checkerboard Videos",
            "Please select the checkerboard videos/images for each camera."
        )
        
        base_path = Path(self.app.participant_name) / 'calibration'
        
        for cam in range(1, num_cameras + 1):
            file_path = filedialog.askopenfilename(
                title=f"Select Checkerboard Video/Image for Camera {cam}",
                filetypes=[
                    ("Video/Image files", f"*.{self.video_extension_var.get()}"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                messagebox.showerror("Error", f"No file selected for camera {cam}")
                return False
            
            # Copy to appropriate folder
            dest_folder = base_path / 'intrinsics' / f'int_cam{cam}_img'
            dest_path = dest_folder / Path(file_path).name
            if dest_path.exists(): dest_path.unlink()
            dest_path.symlink_to(file_path)
        
        return True
    
    def input_scene_videos(self, num_cameras):
        """Input scene videos/images for each camera"""
        messagebox.showinfo(
            "Input Scene Videos",
            "Please select the scene videos/images for each camera."
        )
        
        base_path = Path(self.app.participant_name) / 'calibration'
        
        for cam in range(1, num_cameras + 1):
            file_path = filedialog.askopenfilename(
                title=f"Select Scene Video/Image for Camera {cam}",
                filetypes=[
                    ("Video/Image files", f"*.{self.video_extension_var.get()}"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                messagebox.showerror("Error", f"No file selected for camera {cam}")
                return False
            
            # Copy to appropriate folder
            dest_folder = base_path / 'extrinsics' / f'ext_cam{cam}_img'
            dest_path = dest_folder / Path(file_path).name
            if dest_path.exists(): dest_path.unlink()
            dest_path.symlink_to(file_path)
        
        return True
    
    def input_scene_coordinates(self):
        """Input scene coordinates for calibration with zoomable image"""
        # Clear any existing content in right panel
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        
        # Show points frame in the right panel
        self.points_frame = ctk.CTkFrame(self.right_panel)
        self.points_frame.pack(fill='both', expand=True)
        
        # Choose a scene image/video for reference
        file_path = filedialog.askopenfilename(
            title="Select a Scene Image/Video for Point Selection",
            filetypes=[
                ("Video/Image files", f"*.{self.video_extension_var.get()}"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            messagebox.showerror("Error", "No file selected for point selection")
            return False
        
        # Load image from video if video file
        if Path(file_path).suffix.lower() in ('.mp4', '.avi', '.mov'):
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                messagebox.showerror("Error", "Failed to read video frame")
                return False
            scene_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            scene_image = plt.imread(file_path)
        
        # Create matplotlib figure for point selection with zoom capability
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(scene_image)
        self.ax.set_title("Click to select 8 points for calibration (use mouse wheel to zoom, right-click to remove last point)")
        
        # Store selected points
        self.points_2d = []
        self.point_markers = []
        
        # Add navigation toolbar for zoom functionality
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        
        # Create canvas widget
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.points_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add toolbar with zoom capabilities
        self.toolbar_frame = tk.Frame(self.points_frame)
        self.toolbar_frame.pack(fill='x')
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        
        # Connect the toolbar events to track zoom state
        self.toolbar.pan()  # Start in pan mode
        self.toolbar.mode = ""  # Reset mode
        original_update = self.toolbar.update
        
        def custom_update():
            # Track if we're in zoom or pan mode
            self.zooming_mode = self.toolbar.mode in ('zoom rect', 'pan/zoom')
            original_update()
        
        self.toolbar.update = custom_update
        
        # Click handler for selecting points
        def onclick(event):
            # Only handle clicks if we're not in zoom or pan mode
            if event.inaxes == self.ax and event.button == 1 and not self.zooming_mode:
                if len(self.points_2d) < 8:
                    x, y = event.xdata, event.ydata
                    if x is not None and y is not None:
                        self.points_2d.append((x, y))
                        # Plot point in red
                        point = self.ax.plot(x, y, 'ro')[0]
                        self.point_markers.append(point)
                        # Add point number
                        self.ax.text(x + 10, y + 10, str(len(self.points_2d)), color='white', 
                                fontsize=14, fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))
                        self.fig.canvas.draw()
                        
                        if len(self.points_2d) == 8:
                            # Process the points
                            self.process_coordinate_input()
        
        # Right-click handler to remove the last point
        def on_right_click(event):
            if event.inaxes == self.ax and event.button == 3 and not self.zooming_mode:
                if len(self.points_2d) > 0:
                    # Remove the last point
                    self.points_2d.pop()
                    # Remove the marker
                    last_marker = self.point_markers.pop()
                    last_marker.remove()
                    # Remove any text annotations for this point (approximate by finding last added)
                    for text in self.ax.texts:
                        if text.get_text() == str(len(self.points_2d) + 1):
                            text.remove()
                            break
                    self.fig.canvas.draw()
        
        # Connect click events
        self.canvas.mpl_connect('button_press_event', onclick)
        self.canvas.mpl_connect('button_press_event', on_right_click)
        
        # Instructions label
        ctk.CTkLabel(
            self.points_frame,
            text="Click to select 8 points for calibration. Use mouse wheel or toolbar to zoom. Right-click to remove last point.",
            wraplength=600,
            font=("Helvetica", 12),
            text_color="gray"
        ).pack(pady=10)
        
        return True
    
    def process_coordinate_input(self):
        """Process the 8 selected points and input their 3D coordinates"""
        # Predefined coordinates layout with origin at first point
        predefined_coords = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]
        
        def create_coordinate_window(point_idx):
            if point_idx >= len(self.points_2d):
                # All points processed
                self.save_coordinates_to_config()
                return
            
            # Change current point to yellow
            if point_idx < len(self.point_markers):
                self.point_markers[point_idx].set_color('yellow')
                self.fig.canvas.draw()
            
            # Create window for coordinate input
            coord_win = ctk.CTkToplevel(self.app.root)
            coord_win.title(f"Point {point_idx + 1} Coordinates")
            coord_win.geometry("400x300")
            coord_win.transient(self.app.root)
            coord_win.grab_set()
            
            # Main frame
            main_frame = ctk.CTkFrame(coord_win)
            main_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            # Title
            ctk.CTkLabel(
                main_frame,
                text=f"Enter 3D Coordinates for Point {point_idx + 1}",
                font=("Helvetica", 16, "bold")
            ).pack(pady=(0, 20))
            
            # For first point, use [0,0,0] and disable editing
            x_var = ctk.StringVar(value=str(predefined_coords[point_idx][0]))
            y_var = ctk.StringVar(value=str(predefined_coords[point_idx][1]))
            z_var = ctk.StringVar(value=str(predefined_coords[point_idx][2]))
            
            # X coordinate
            x_frame = ctk.CTkFrame(main_frame)
            x_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(x_frame, text="X (meters):", width=100).pack(side='left', padx=5)
            x_entry = ctk.CTkEntry(x_frame, textvariable=x_var, width=150)
            x_entry.pack(side='left', padx=5)
            
            # Y coordinate
            y_frame = ctk.CTkFrame(main_frame)
            y_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(y_frame, text="Y (meters):", width=100).pack(side='left', padx=5)
            y_entry = ctk.CTkEntry(y_frame, textvariable=y_var, width=150)
            y_entry.pack(side='left', padx=5)
            
            # Z coordinate
            z_frame = ctk.CTkFrame(main_frame)
            z_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(z_frame, text="Z (meters):", width=100).pack(side='left', padx=5)
            z_entry = ctk.CTkEntry(z_frame, textvariable=z_var, width=150)
            z_entry.pack(side='left', padx=5)
            
            # Disable entries for first point
            if point_idx == 0:
                x_entry.configure(state='disabled')
                y_entry.configure(state='disabled')
                z_entry.configure(state='disabled')
            
            # Submit function
            def submit_coords():
                try:
                    x = float(x_var.get())
                    y = float(y_var.get())
                    z = float(z_var.get())
                    
                    # Save coordinates
                    self.object_coords_3d.append([x, y, z])
                    
                    # Change point color to green
                    if point_idx < len(self.point_markers):
                        self.point_markers[point_idx].set_color('green')
                        self.fig.canvas.draw()
                    
                    # Close window
                    coord_win.destroy()
                    
                    # Process next point
                    self.app.root.after(100, lambda: create_coordinate_window(point_idx + 1))
                    
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numbers for coordinates")
            
            # Submit button
            ctk.CTkButton(
                main_frame,
                text="Next Point",
                command=submit_coords,
                height=40,
                width=150
            ).pack(pady=20)
        
        # Start with first point
        create_coordinate_window(0)
    
    def save_coordinates_to_config(self):
        """Save the 3D coordinates to the config file"""
        # Only update progress after all coordinates are entered
        messagebox.showinfo(
            "Calibration Complete",
            "The 3D coordinates have been saved. You will need to click on these same points in order when running Pose2Sim After activation."
        )
        
        # Update the progress bar and tab indicator now that all steps are complete
        if hasattr(self.app, 'update_tab_indicator'):
            self.app.update_tab_indicator('calibration', True)
        if hasattr(self.app, 'update_progress_bar') and hasattr(self.app, 'progress_steps'):
            progress_value = self.app.progress_steps.get('calibration', 15)
            self.app.update_progress_bar(progress_value)
    
    def input_calibration_file(self):
        """Input a calibration file for conversion"""
        file_path = filedialog.askopenfilename(
            title="Select Calibration File to Convert",
            filetypes=[
                ("All files", "*.*"),
                ("QTM files", "*.qtm"),
                ("CSV files", "*.csv"),
                ("XML files", "*.xml")
            ]
        )
        
        if not file_path:
            messagebox.showerror("Error", "No calibration file selected")
            return False
        
        # Create calibration folder
        calibration_path = Path(self.app.participant_name) / 'calibration'
        calibration_path.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        dest_path = calibration_path / Path(file_path).name
        if dest_path.exists(): dest_path.unlink()
        dest_path.symlink_to(file_path)
        
        # Update progress now that conversion is complete
        if hasattr(self.app, 'update_tab_indicator'):
            self.app.update_tab_indicator('calibration', True)
        if hasattr(self.app, 'update_progress_bar') and hasattr(self.app, 'progress_steps'):
            progress_value = self.app.progress_steps.get('calibration', 15)
            self.app.update_progress_bar(progress_value)
            
        # Show success message
        messagebox.showinfo(
            "Calibration Complete",
            "Calibration file has been imported successfully."
        )
        
        return True