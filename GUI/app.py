import os
from pathlib import Path
import customtkinter as ctk

# Import language manager
from GUI.language_manager import LanguageManager

# Import tabs
from GUI.tabs.welcome_tab import WelcomeTab
from GUI.tabs.calibration_tab import CalibrationTab
from GUI.tabs.prepare_video_tab import PrepareVideoTab
from GUI.tabs.pose_model_tab import PoseModelTab
from GUI.tabs.synchronization_tab import SynchronizationTab
from GUI.tabs.activation_tab import ActivationTab
from GUI.tabs.advanced_tab import AdvancedTab
from GUI.tabs.batch_tab import BatchTab
from GUI.tabs.visualization_tab import VisualizationTab
from GUI.tabs.tutorial_tab import TutorialTab
from GUI.tabs.about_tab import AboutTab


# Import config generator
from GUI.config_generator import ConfigGenerator

class Pose2SimApp:
    def __init__(self, root):
        self.root = root

        # Initialize language manager
        self.lang_manager = LanguageManager()

        self.root.title(self.lang_manager.get_text('app_title'))
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size
        self.window_width = 1300
        self.window_height = 800
        
        # Calculate position for center of screen
        x = (screen_width - self.window_width) // 2
        y = (screen_height - self.window_height) // 2
        
        # Set window size and position
        self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
        
        # Initialize variables
        self.language = None  # Will be 'en' or 'fr'
        self.analysis_mode = None  # Will be '2d' or '3d'
        self.process_mode = None  # Will be 'single' or 'batch'
        self.participant_name = None
        self.num_trials = 0  # For batch mode
        
        # Check tutorial status
        self.tutorial_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tutorial_completed")
        self.tutorial_completed = os.path.exists(self.tutorial_marker)
        
        self.top_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.top_frame.pack(fill="x", padx=10, pady=5)

        # Configure light/dark mode selector in top-left corner
        self.setup_darkmode_selector(self.top_frame)

        # Configure language selector in top-right corner
        self.setup_language_selector(self.top_frame)

        # Create config generator
        self.config_generator = ConfigGenerator()

        # Start with welcome screen for initial setup
        self.welcome_tab = WelcomeTab(self.root, self)
        
    def setup_darkmode_selector(self, frame):
        """Creates a light/dark mode switch in the top-left corner"""
        self.darkmode_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.darkmode_frame.pack(side="left")

        self.current_appearance_mode = ctk.get_appearance_mode().lower()
        
        darkmode_button = ctk.CTkButton(
            self.darkmode_frame, 
            text="☀️⏾", 
            width=60, 
            command=lambda: self.change_darkmode(), 
            fg_color=("blue" if self.current_appearance_mode == "light" else "SkyBlue1")
        )
        darkmode_button.pack(side="left", padx=2)
        
    def change_darkmode(self):
        """Toggles between light and dark mode"""
        if self.current_appearance_mode == "dark":
            ctk.set_appearance_mode("light")
            self.current_appearance_mode = "light"
        else:
            ctk.set_appearance_mode("dark")
            self.current_appearance_mode = "dark"    
    
    def setup_language_selector(self, frame):
        """Creates a language selector in the top-right corner"""
        self.lang_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.lang_frame.pack(side="right")
        
        self.lang_var = ctk.StringVar(value="EN")
        
        self.en_button = ctk.CTkButton(
            self.lang_frame, 
            text="EN", 
            width=40, 
            command=lambda: self.change_language("en"), 
            fg_color=("blue" if self.lang_var.get() == "EN" else "SkyBlue1")
        )
        self.en_button.pack(side="left", padx=2)
        
        self.fr_button = ctk.CTkButton(
            self.lang_frame, 
            text="FR", 
            width=40, 
            command=lambda: self.change_language("fr"),
            fg_color=("blue" if self.lang_var.get() == "FR" else "SkyBlue1")
        )
        self.fr_button.pack(side="left", padx=2)
    
    def change_language(self, lang_code):
        """Changes the application language"""
        if lang_code == "en":
            self.lang_var.set("EN")
            self.language = "en"
            self.en_button.configure(fg_color="blue")
            self.fr_button.configure(fg_color="SkyBlue1")
        else:
            self.lang_var.set("FR")
            self.language = "fr"
            self.fr_button.configure(fg_color="blue")
            self.en_button.configure(fg_color="SkyBlue1")

        # Update the LanguageManager
        self.lang_manager.set_language(self.language)

        # Update all text elements if the main UI is already built
        self.update_ui_language()
    
    def update_ui_language(self):
        """Updates all UI text elements with the selected language"""
        # Update window title
        self.root.title(self.lang_manager.get_text('app_title'))
        
        self._update_widget_text(self.root)

    def _update_widget_text(self, widget):
        """Recursively updates text for all CTkLabel and CTkButton widgets"""
        if isinstance(widget, (ctk.CTkLabel, ctk.CTkButton)):
            if hasattr(widget, 'translation_key'):
                new_text = self.lang_manager.get_text(widget.translation_key)
                widget.configure(text=new_text)

        # Recursively check all child widgets
        try:
            for child in widget.winfo_children():
                self._update_widget_text(child)
        except AttributeError:
            pass

    def start_configuration(self, analysis_mode, process_mode, participant_name, num_trials=0):
        """Starts the configuration process after welcome screen"""
        self.analysis_mode = analysis_mode
        self.process_mode = process_mode
        self.participant_name = participant_name
        
        if process_mode == 'batch':
            self.num_trials = num_trials
        
        # Clear welcome screen
        self.welcome_tab.clear()
        
        # Create folder structure based on analysis mode and process mode
        self.create_folder_structure()
        
        # Set up the main interface with tabs
        self.setup_main_interface()
    
    def create_folder_structure(self):
        """Creates the folder structure based on analysis mode and process mode"""
        if self.analysis_mode == '3d':
            # 3D analysis needs the full folder structure
            if self.process_mode == 'single':
                # Create participant directory
                participant_path = os.path.join(self.participant_name)
                
                # Create calibration and videos subdirectories
                calibration_path = os.path.join(participant_path, 'calibration')
                videos_path = os.path.join(participant_path, 'videos')
                
                # Create all directories
                os.makedirs(calibration_path, exist_ok=True)
                os.makedirs(videos_path, exist_ok=True)
            else:
                # Batch mode needs a parent directory with calibration folder
                # and separate trial directories
                participant_path = os.path.join(self.participant_name)
                calibration_path = os.path.join(participant_path, 'calibration')
                
                os.makedirs(participant_path, exist_ok=True)
                os.makedirs(calibration_path, exist_ok=True)
                
                for i in range(1, self.num_trials + 1):
                    trial_path = os.path.join(participant_path, f'Trial_{i}')
                    videos_path = os.path.join(trial_path, 'videos')
                    
                    os.makedirs(trial_path, exist_ok=True)
                    os.makedirs(videos_path, exist_ok=True)
        else:
            # 2D analysis just needs a single directory for the participant
            participant_path = os.path.join(self.participant_name)
            os.makedirs(participant_path, exist_ok=True)
    
    def setup_main_interface(self):
        """Sets up the main interface with sidebar navigation and content area"""
        # Clear any existing content
        for widget in self.root.winfo_children():
            if widget != self.lang_frame:  # Keep the language selector
                widget.destroy()
        
        # Create main container frame
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create sidebar frame (left)
        self.sidebar = ctk.CTkFrame(self.main_container, width=220)
        self.sidebar.pack(side='left', fill='both', padx=5, pady=5)
        self.sidebar.pack_propagate(False)  # Prevent shrinking
        
        self.top_sidebar = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.top_sidebar.pack(fill="x", padx=10, pady=5)

        # Configure light/dark mode selector in top-left corner
        self.setup_darkmode_selector(self.top_sidebar)

        # Configure language selector in top-right corner
        self.setup_language_selector(self.top_sidebar)

        # Add logo above the title
        favicon_path = Path(__file__).parent/"assets/Pose2Sim_logo.png"
        self.top_image = Image.open(favicon_path)
        self.top_photo = ctk.CTkImage(light_image=self.top_image, dark_image=self.top_image, size=(120,120))
        image_label = ctk.CTkLabel(self.sidebar, image=self.top_photo, text="")
        image_label.pack(pady=(0, 0))

        # App title in sidebar
        app_title_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        app_title_frame.pack(fill='x', pady=(10, 30))
        
        ctk.CTkLabel(
            app_title_frame,
            text="Pose2Sim",
            font=("Helvetica", 22, "bold")
        ).pack()
        
        mode_text = "2D Analysis" if self.analysis_mode == '2d' else "3D Analysis"
        ctk.CTkLabel(
            app_title_frame,
            text=mode_text,
            font=("Helvetica", 14)
        ).pack()
        
        # Create content area frame (right)
        self.content_area = ctk.CTkFrame(self.main_container)
        self.content_area.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Initialize progress tracking
        self.setup_progress_bar()
        
        # Initialize tabs dictionary
        self.tabs = {}
        self.tab_buttons = {}
        self.tab_info = {}
        
        # Create tabs based on analysis mode
        if self.analysis_mode == '3d':
            # 3D mode tabs
            tab_classes = [
                ('tutorial', TutorialTab, "Tutorial", "📚"),
                ('calibration', CalibrationTab, "Calibration", "📏"),
                ('prepare_video', PrepareVideoTab, "Prepare Video", "🎥"),
                ('pose_model', PoseModelTab, "Pose Estimation", "👤"),
                ('synchronization', SynchronizationTab, "Synchronization", "⏱️"),
                ('advanced', AdvancedTab, "Advanced Settings", "⚙️"),
                ('activation', ActivationTab, "Run Analysis", "▶️")
            ]
            
            if self.process_mode == 'batch':
                tab_classes.append(('batch', BatchTab, "Batch Configuration", "📚"))
            
            # Add visualization tab
            tab_classes.append(('visualization', VisualizationTab, "Data Visualization", "📊"))
            
            # Add about tab at the end
            tab_classes.append(('about', AboutTab, "About Us", "ℹ️"))
        else:
            # 2D mode tabs (simplified)
            tab_classes = [
                ('tutorial', TutorialTab, "Tutorial", "📚"),
                ('pose_model', PoseModelTab, "Pose Estimation", "👤"),
                ('advanced', AdvancedTab, "Advanced Settings", "⚙️"),
                ('activation', ActivationTab, "Run Analysis", "▶️"),
                ('visualization', VisualizationTab, "Data Visualization", "📊"),
                ('about', AboutTab, "About Us", "ℹ️")  # Add about tab at the end
            ]
        
        # Create tab instances and sidebar buttons
        for i, (tab_id, tab_class, tab_title, tab_icon) in enumerate(tab_classes):
            # Store tab class and metadata for lazy instantiation
            self.tab_info[tab_id] = {
                'class': tab_class,
                'title': tab_title,
                'icon': tab_icon,
                'needs_simplified': tab_id in ['pose_model', 'advanced', 'activation']
            }

            # Create sidebar button for this tab
            button = ctk.CTkButton(
                self.sidebar,
                text=f"{tab_icon} {tab_title}",
                anchor="w",
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                command=lambda t=tab_id: self.show_tab(t),
                height=40,
                width=200
            )
            button.pack(pady=5, padx=10)
            self.tab_buttons[tab_id] = button
        
        # Determine first tab to show
        if not self.tutorial_completed and 'tutorial' in self.tab_info:
            first_tab_id = 'tutorial'
        else:
            first_tab_id = list(self.tab_info.keys())[0]
            if first_tab_id == 'tutorial':  # Skip tutorial if completed
                first_tab_id = list(self.tab_info.keys())[1] if len(self.tab_info) > 1 else first_tab_id
        
        # Show first tab (this will trigger lazy instantiation)
        self.show_tab(first_tab_id)
    
    def show_tab(self, tab_id):
        """Show the selected tab and hide others"""
        if tab_id not in self.tabs:
            # Optional: Show loading indicator for heavy tabs
            if tab_id in ['visualization', 'tutorial']:
                loading_label = ctk.CTkLabel(
                    self.content_area,
                    text=f"Loading {self.tab_info[tab_id]['title']}...",
                    font=("Helvetica", 14)
                )
                loading_label.place(relx=0.5, rely=0.5, anchor="center")
                self.root.update()  # Force UI to show loading message
            
            # Instantiate the tab
            tab_class = self.tab_info[tab_id]['class']
            
            if self.tab_info[tab_id]['needs_simplified']:
                self.tabs[tab_id] = tab_class(
                    self.content_area, 
                    self, 
                    simplified=(self.analysis_mode == '2d')
                )
            else:
                self.tabs[tab_id] = tab_class(self.content_area, self)
            
            # Remove loading indicator if it was shown
            if tab_id in ['visualization', 'tutorial']:
                loading_label.destroy()
        
        # Hide all tab frames
        for tid, tab in self.tabs.items():
            tab.frame.pack_forget()
            
            # Reset button colors
            self.tab_buttons[tid].configure(
                fg_color="transparent",
                text_color=("gray10", "gray90")
            )
        
        # Show selected tab frame
        self.tabs[tab_id].frame.pack(fill='both', expand=True)
        
        # Highlight selected tab button
        self.tab_buttons[tab_id].configure(
            fg_color=("#3a7ebf", "#1f538d"),
            text_color=("white", "white")
        )
    
    def setup_progress_bar(self):
        """Create and configure the progress bar at the bottom of the window."""
        # Create a frame for the progress bar
        self.progress_frame = ctk.CTkFrame(self.root, height=50)
        self.progress_frame.pack(side="bottom", fill="x", padx=10, pady=5)
        
        # Progress label
        self.progress_label = ctk.CTkLabel(
            self.progress_frame, 
            text="Overall Progress: 0%",
            font=("Helvetica", 12)
        )
        self.progress_label.pack(pady=(5, 2))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, height=15)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 5))
        self.progress_bar.set(0)  # Initialize to 0%
        
        # Define progress steps based on analysis mode
        if self.analysis_mode == '3d':
            self.progress_steps = {
                'calibration': 15,
                'prepare_video': 30,
                'pose_model': 50,
                'synchronization': 70,
                'advanced': 85,
                'activation': 100
            }
        else:  # 2D mode
            self.progress_steps = {
                'pose_model': 40,
                'advanced': 70,
                'activation': 100
            }
    
    def update_progress_bar(self, value):
        """Update the progress bar to a specific value (0-100)."""
        progress = value / 100
        self.progress_bar.set(progress)
        self.progress_label.configure(text=f"Overall Progress: {value}%")
    
    def update_tab_indicator(self, tab_name, completed=True):
        """Updates the tab indicator to show completion status"""
        if tab_name in self.tabs:
            # Get the current tab title and icon
            tab_title = self.tabs[tab_name].get_title()
            tab_icon = self.tab_buttons[tab_name].cget("text").split(" ")[0]
            
            # Update the tab button text
            indicator = "✅" if completed else "❌"
            self.tab_buttons[tab_name].configure(
                text=f"{tab_icon} {tab_title} {indicator}"
            )
    
    def generate_config(self):
        """Generates the configuration file based on the settings"""
        # Collect all settings from tabs
        settings = {}
        for name, tab in self.tabs.items():
            if hasattr(tab, 'get_settings'):
                tab_settings = tab.get_settings()
                settings.update(tab_settings)
        
        # Generate config file
        config_path = os.path.join(self.participant_name, 'Config.toml')
        if self.analysis_mode == '2d':
            self.config_generator.generate_2d_config(config_path, settings)
        else:
            self.config_generator.generate_3d_config(config_path, settings)
            
            # For batch mode, also generate configs for each trial
            if self.process_mode == 'batch':
                for i in range(1, self.num_trials + 1):
                    trial_config_path = os.path.join(self.participant_name, f'Trial_{i}', 'Config.toml')
                    self.config_generator.generate_3d_config(trial_config_path, settings)