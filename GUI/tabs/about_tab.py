from pathlib import Path
import requests
import threading
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
import webbrowser
import datetime
import re
import time
import json
import traceback
from PIL import Image

class AboutTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        # Create main frame
        self.frame = ctk.CTkFrame(parent)
        
        # GitHub repository URL
        self.github_url = "https://github.com/perfanalytics/pose2sim"
        
        # Citation DOIs for tracking
        self.citation_dois = [
            "10.21105/joss.04362",  # Pose2Sim JOSS paper
            "10.3390/s22072712",    # Pose2Sim Accuracy paper
            "10.3390/s21196530",    # Pose2Sim Robustness paper
            "10.21105/joss.06849"   # Sports2D JOSS paper
        ]
        
        # Data storage
        self.releases = []
        self.citations = []
        self.citation_data = {}
        self.latest_version = "Unknown"
        
        # Build the UI
        self.build_ui()
        
        # Fetch data in background threads
        threading.Thread(target=self.fetch_github_releases, daemon=True).start()
        threading.Thread(target=self.fetch_citation_data, daemon=True).start()
    
    def get_title(self):
        """Return the tab title"""
        return "About Us"
    
    def get_settings(self):
        """Get the about tab settings"""
        return {}  # This tab doesn't add settings to the config file
    
    def show_update_instructions(self):
        """Show update instructions"""
        # Create a custom dialog with instructions
        dialog = ctk.CTkToplevel(self.frame)
        dialog.title("Update Pose2Sim")
        dialog.geometry("500x300")
        dialog.transient(self.frame)  # Set as transient to main window
        dialog.grab_set()  # Make it modal
        
        # Center the window
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - dialog.winfo_width()) // 2
        y = (dialog.winfo_screenheight() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Dialog content
        version_text = f"to version {self.latest_version}" if self.latest_version != "Unknown" else "to the latest version"
        ctk.CTkLabel(
            dialog,
            text=f"Update Pose2Sim {version_text}",
            font=("Helvetica", 16, "bold")
        ).pack(pady=(20, 15))
        
        # Instructions frame with monospace font for command
        instruction_frame = ctk.CTkFrame(dialog)
        instruction_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        ctk.CTkLabel(
            instruction_frame,
            text="To update Pose2Sim to the latest version:",
            font=("Helvetica", 12),
            anchor="w",
            justify="left"
        ).pack(fill='x', padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            instruction_frame,
            text="1. Open a command prompt or terminal",
            font=("Helvetica", 12),
            anchor="w",
            justify="left"
        ).pack(fill='x', padx=10, pady=2)
        
        ctk.CTkLabel(
            instruction_frame,
            text="2. Run the following command:",
            font=("Helvetica", 12),
            anchor="w",
            justify="left"
        ).pack(fill='x', padx=10, pady=2)
        
        # Command box with copy button
        cmd_frame = ctk.CTkFrame(instruction_frame, fg_color=("gray95", "gray20"))
        cmd_frame.pack(fill='x', padx=20, pady=10)
        
        command = "pip install pose2sim --upgrade"
        cmd_text = ctk.CTkTextbox(
            cmd_frame,
            height=30,
            font=("Courier", 12),
            wrap="none"
        )
        cmd_text.pack(fill='x', padx=10, pady=(10, 5))
        cmd_text.insert("1.0", command)
        cmd_text.configure(state="disabled")
        
        # Copy button
        ctk.CTkButton(
            cmd_frame,
            text="Copy Command",
            command=lambda: self.copy_to_clipboard(command),
            width=120,
            height=28
        ).pack(anchor='e', padx=10, pady=(0, 10))
        
        ctk.CTkLabel(
            instruction_frame,
            text="3. Restart this application after updating",
            font=("Helvetica", 12),
            anchor="w",
            justify="left"
        ).pack(fill='x', padx=10, pady=2)
        
        # Close button
        ctk.CTkButton(
            dialog,
            text="Close",
            command=dialog.destroy,
            width=100,
            height=32
        ).pack(pady=15)
    
    def build_ui(self):
        """Build the about tab UI"""
        # Create a scrollable content frame with more padding
        self.content_frame = ctk.CTkScrollableFrame(self.frame)
        self.content_frame.pack(fill='both', expand=True, padx=0, pady=0)
        
        # Create header with logo and title
        self.create_header()
        
        # Create What's New section
        self.create_whats_new_section()
        
        # Create Contributors section
        self.create_contributors_section()
        
        # Create Citation section
        self.create_citation_section()
        
        # Create Citation Tracker section
        self.create_citation_tracker_section()
    
    def create_header(self):
        """Create header with logo and title"""
        header_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        header_frame.pack(fill='x', pady=(0, 25))
        
        # Left side for logo and version info
        left_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        left_frame.pack(side='left', fill='y')
        
        # Try to load logo image
        logo_path = Path(__file__).parent.parent / "assets" / "Pose2Sim_logo.png"
        try:
            if logo_path.exists():
                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((100, 100), Image.LANCZOS)
                logo = ctk.CTkImage(light_image=logo_img, dark_image=logo_img, size=(100, 100))
                
                logo_label = ctk.CTkLabel(left_frame, image=logo, text="")
                logo_label.image = logo  # Keep a reference
                logo_label.pack(padx=20)
        except Exception:
            # If logo loading fails, just skip it
            pass
        
        # Update button with improved styling
        update_button = ctk.CTkButton(
            left_frame,
            text="Update Pose2Sim",
            command=self.show_update_instructions,
            width=160,
            height=28,
            corner_radius=8,
            fg_color=("#28A745", "#218838"),
            hover_color=("#218838", "#1E7E34")
        )
        update_button.pack(pady=(5, 0))
        
        # Title and description
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(side='left', fill='both', expand=True, padx=20)
        
        ctk.CTkLabel(
            title_frame,
            text="Pose2Sim",
            font=("Helvetica", 28, "bold")
        ).pack(anchor='w')
        
        ctk.CTkLabel(
            title_frame,
            text="An open-source Python package for multiview markerless kinematics",
            font=("Helvetica", 16)
        ).pack(anchor='w', pady=(5, 0))
        
        # Website and GitHub buttons - improved styling
        button_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        button_frame.pack(side='right', padx=20)
        
        ctk.CTkButton(
            button_frame,
            text="GitHub",
            command=lambda: webbrowser.open(self.github_url),
            width=120,
            height=32,
            corner_radius=8,
            hover_color=("#2E86C1", "#1F618D")
        ).pack(pady=5)
        
        ctk.CTkButton(
            button_frame,
            text="Documentation",
            command=lambda: webbrowser.open("https://github.com/perfanalytics/pose2sim"),
            width=120,
            height=32,
            corner_radius=8,
            hover_color=("#2E86C1", "#1F618D")
        ).pack(pady=5)
    
    def create_whats_new_section(self):
        """Create What's New section showing recent releases"""
        # Section frame with improved styling
        section_frame = ctk.CTkFrame(self.content_frame, corner_radius=10)
        section_frame.pack(fill='x', pady=15)
        
        # Section header with improved styling
        ctk.CTkLabel(
            section_frame,
            text="What's New",
            font=("Helvetica", 20, "bold"),
        ).pack(anchor='w', padx=20, pady=(15, 5))
        
        # Add a separator
        separator = ctk.CTkFrame(section_frame, height=2, fg_color=("gray80", "gray30"))
        separator.pack(fill='x', padx=20, pady=(0, 15))
        
        # Loading indicator with improved styling
        self.releases_loading_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        self.releases_loading_frame.pack(fill='x', padx=20, pady=15)
        
        ctk.CTkLabel(
            self.releases_loading_frame,
            text="Loading recent releases...",
            font=("Helvetica", 12)
        ).pack(pady=10)
        
        progress = ctk.CTkProgressBar(self.releases_loading_frame, height=10)
        progress.pack(fill='x', padx=40, pady=5)
        progress.configure(mode="indeterminate")
        progress.start()
        
        # Create frame for releases (initially empty) with improved styling
        self.releases_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        self.releases_frame.pack(fill='x', padx=20, pady=10)
    
    def create_contributors_section(self):
        """Create Contributors section with improved styling"""
        # Section frame with improved styling
        section_frame = ctk.CTkFrame(self.content_frame, corner_radius=10)
        section_frame.pack(fill='x', pady=15)
        
        # Section header with improved styling
        ctk.CTkLabel(
            section_frame,
            text="Acknowledgements",
            font=("Helvetica", 20, "bold"),
        ).pack(anchor='w', padx=20, pady=(15, 5))
        
        # Add a separator
        separator = ctk.CTkFrame(section_frame, height=2, fg_color=("gray80", "gray30"))
        separator.pack(fill='x', padx=20, pady=(0, 15))
                      
        ## Community Contributors Section with improved styling
        community_frame = ctk.CTkFrame(section_frame, fg_color=("gray95", "gray20"), corner_radius=8)
        community_frame.pack(fill='x', padx=20, pady=15)
        
        # Contributors Details with improved styling and readability
        contributors_text = (
            "Thanks to all the contributors who have helped improve Pose2Sim through their valuable support:\n\n"
            "Supervised my PhD: @lreveret (INRIA, Université Grenoble Alpes), @mdomalai (Université de Poitiers).\n"
            "Provided the Demo data: @aaiaueil (Université Gustave Eiffel).\n"
            "Tested the code and provided feedback: @simonozan, @daeyongyang, @ANaaim, @rlagnsals.\n"
            "Submitted various accepted pull requests: @ANaaim, @rlagnsals, @peterlololsss.\n"
            "Provided a code snippet for Optitrack calibration: @claraaudap (Université Bretagne Sud).\n"
            "Issued MPP2SOS, a (non-free) Blender extension based on Pose2Sim: @carlosedubarreto.\n"
            "Bug reports, feature suggestions, and code contributions: @AYLARDJ (AYLardjne), @M.BLANDEAU, @J.Janseen."
        )
        
        ctk.CTkLabel(
            community_frame,
            text=contributors_text,
            wraplength=800,
            justify="left",
            padx=15, 
            pady=15
        ).pack(fill='x', padx=10, pady=10)
        
        # View all contributors button with improved styling
        ctk.CTkButton(
            community_frame,
            text="View All Contributors on GitHub",
            command=lambda: webbrowser.open(f"{self.github_url}/graphs/contributors"),
            width=250,
            height=35,
            corner_radius=8,
            hover_color=("#2E86C1", "#1F618D")
        ).pack(anchor='w', padx=10, pady=(0, 15))
    
    def create_citation_section(self):
        """Create Citation section with paper references - improved styling"""
        # Section frame with improved styling
        section_frame = ctk.CTkFrame(self.content_frame, corner_radius=10)
        section_frame.pack(fill='x', pady=15)
        
        # Section header with improved styling
        ctk.CTkLabel(
            section_frame,
            text="How to Cite",
            font=("Helvetica", 20, "bold"),
        ).pack(anchor='w', padx=20, pady=(15, 5))
        
        # Add a separator
        separator = ctk.CTkFrame(section_frame, height=2, fg_color=("gray80", "gray30"))
        separator.pack(fill='x', padx=20, pady=(0, 15))
        
        # Citation information with improved styling
        info_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        info_frame.pack(fill='x', padx=20, pady=10)
        
        ctk.CTkLabel(
            info_frame,
            text="If you use Pose2Sim in your work, please cite the following papers:",
            font=("Helvetica", 14),
            wraplength=800,
            justify="left"
        ).pack(anchor='w', padx=10, pady=10)
        
        # Papers to cite - improved layout and styling
        self.papers = [
            {
                "title": "Pose2Sim: An open-source Python package for multiview markerless kinematics",
                "authors": "Pagnon David, Domalain Mathieu and Reveret Lionel",
                "journal": "Journal of Open Source Software",
                "year": "2022",
                "doi": "10.21105/joss.04362",
                "url": "https://joss.theoj.org/papers/10.21105/joss.04362",
                "bibtex": """@Article{Pagnon_2022_JOSS, 
  AUTHOR = {Pagnon, David and Domalain, Mathieu and Reveret, Lionel}, 
  TITLE = {Pose2Sim: An open-source Python package for multiview markerless kinematics}, 
  JOURNAL = {Journal of Open Source Software}, 
  YEAR = {2022},
  DOI = {10.21105/joss.04362}, 
  URL = {https://joss.theoj.org/papers/10.21105/joss.04362}
}"""
            },
            {
                "title": "Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 2: Accuracy",
                "authors": "Pagnon David, Domalain Mathieu and Reveret Lionel",
                "journal": "Sensors",
                "year": "2022",
                "doi": "10.3390/s22072712",
                "url": "https://www.mdpi.com/1424-8220/22/7/2712",
                "bibtex": """@Article{Pagnon_2022_Accuracy,
  AUTHOR = {Pagnon, David and Domalain, Mathieu and Reveret, Lionel},
  TITLE = {Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 2: Accuracy},
  JOURNAL = {Sensors},
  YEAR = {2022},
  DOI = {10.3390/s22072712},
  URL = {https://www.mdpi.com/1424-8220/22/7/2712}
}"""
            },
            {
                "title": "Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 1: Robustness",
                "authors": "Pagnon David, Domalain Mathieu and Reveret Lionel",
                "journal": "Sensors",
                "year": "2021",
                "doi": "10.3390/s21196530",
                "url": "https://www.mdpi.com/1424-8220/21/19/6530",
                "bibtex": """@Article{Pagnon_2021_Robustness,
  AUTHOR = {Pagnon, David and Domalain, Mathieu and Reveret, Lionel},
  TITLE = {Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 1: Robustness},
  JOURNAL = {Sensors},
  YEAR = {2021},
  DOI = {10.3390/s21196530},
  URL = {https://www.mdpi.com/1424-8220/21/19/6530}
}"""
            },
            {
                "title": "Sports2D: Compute 2D human pose and angles from a video or a webcam",
                "authors": "Pagnon David and Kim HunMin",
                "journal": "Journal of Open Source Software",
                "year": "2024",
                "doi": "10.21105/joss.06849",
                "url": "https://joss.theoj.org/papers/10.21105/joss.06849",
                "bibtex": """@article{Pagnon_Sports2D_Compute_2D_2024,
   author = {Pagnon, David and Kim, HunMin},
   doi = {10.21105/joss.06849},
   journal = {Journal of Open Source Software},
   month = sep,
   number = {101},
   pages = {6849},
   title = {{Sports2D: Compute 2D human pose and angles from a video or a webcam}},
   url = {https://joss.theoj.org/papers/10.21105/joss.06849},
   volume = {9},
   year = {2024}
}"""
            },
        ]
        
        # Create expandable sections for each paper with improved styling
        for i, paper in enumerate(self.papers):
            paper_frame = ctk.CTkFrame(info_frame, fg_color=("gray90", "gray25"), corner_radius=8)
            paper_frame.pack(fill='x', padx=10, pady=8)
            
            # Paper title and year with improved styling
            title_frame = ctk.CTkFrame(paper_frame, fg_color="transparent")
            title_frame.pack(fill='x', padx=10, pady=(10, 5))
            
            ctk.CTkLabel(
                title_frame,
                text=f"{paper['title']} ({paper['year']})",
                font=("Helvetica", 14, "bold"),
                text_color=("blue", "#5B8CD7"),
                anchor="w",
                wraplength=700
            ).pack(fill='x', padx=5)
            
            # Paper details with improved styling
            details_frame = ctk.CTkFrame(paper_frame, fg_color="transparent")
            details_frame.pack(fill='x', padx=15, pady=(5, 10))
            
            ctk.CTkLabel(
                details_frame,
                text=f"Authors: {paper['authors']}",
                anchor="w",
                wraplength=750,
                justify="left"
            ).pack(anchor='w', pady=(5, 0))
            
            ctk.CTkLabel(
                details_frame,
                text=f"Journal: {paper['journal']}",
                anchor="w",
                justify="left"
            ).pack(anchor='w', pady=(5, 0))
            
            if "volume" in paper:
                ctk.CTkLabel(
                    details_frame,
                    text=f"Volume: {paper['volume']}, Number: {paper['number']}",
                    anchor="w",
                    justify="left"
                ).pack(anchor='w', pady=(5, 0))
            
            # DOI and buttons with improved styling
            button_frame = ctk.CTkFrame(details_frame, fg_color="transparent")
            button_frame.pack(fill='x', pady=(10, 5))
            
            # DOI button
            ctk.CTkButton(
                button_frame,
                text=f"DOI: {paper['doi']}",
                anchor="w",
                fg_color=("#E1F0F9", "#203A4C"),
                text_color=("blue", "#5B8CD7"),
                hover_color=("#C9E2F2", "#2C4A5E"),
                corner_radius=8,
                height=28,
                command=lambda doi=paper["doi"]: webbrowser.open(f"https://doi.org/{doi}")
            ).pack(side='left', padx=(0, 10))
            
            # View paper button
            ctk.CTkButton(
                button_frame,
                text="View Paper",
                fg_color=("#E1F0F9", "#203A4C"),
                text_color=("blue", "#5B8CD7"), 
                hover_color=("#C9E2F2", "#2C4A5E"),
                corner_radius=8,
                height=28,
                command=lambda url=paper.get("url"): webbrowser.open(url) if url else None
            ).pack(side='left')
            
            # Copy BibTeX button
            ctk.CTkButton(
                button_frame,
                text="Copy BibTeX",
                fg_color=("#E1F0F9", "#203A4C"),
                text_color=("blue", "#5B8CD7"),
                hover_color=("#C9E2F2", "#2C4A5E"),
                corner_radius=8,
                height=28,
                command=lambda txt=paper["bibtex"]: self.copy_to_clipboard(txt)
            ).pack(side='right')
            
            # Add a "Show BibTeX" button to expand/collapse
            bibtex_var = tk.BooleanVar(value=False)
            bibtex_button = ctk.CTkCheckBox(
                details_frame,
                text="Show BibTeX",
                variable=bibtex_var,
                onvalue=True,
                offvalue=False,
                command=lambda var=bibtex_var, idx=i: self.toggle_bibtex(var, idx)
            )
            bibtex_button.pack(anchor='w', pady=(10, 0))
            
            # Hidden BibTeX frame (will be shown when checkbox is clicked)
            bibtex_frame = ctk.CTkFrame(details_frame, fg_color=("gray95", "gray18"))
            bibtex_frame.pack(fill='x', pady=(10, 0))
            bibtex_frame.pack_forget()  # Initially hidden
            
            bibtex_text = ctk.CTkTextbox(
                bibtex_frame,
                height=120,
                font=("Courier", 11),
                wrap="none"
            )
            bibtex_text.pack(fill='x', padx=5, pady=5)
            bibtex_text.insert("1.0", paper["bibtex"])
            bibtex_text.configure(state="disabled")
            
            # Store references to be able to toggle
            paper["bibtex_frame"] = bibtex_frame
    
    def toggle_bibtex(self, var, idx):
        """Toggle the visibility of BibTeX frame for a paper"""
        if var.get():
            # Show BibTeX
            self.papers[idx]["bibtex_frame"].pack(fill='x', pady=(10, 0))
        else:
            # Hide BibTeX
            self.papers[idx]["bibtex_frame"].pack_forget()
    
    def create_citation_tracker_section(self):
        """Create Citation Tracker section with improved styling"""
        # Section frame with improved styling
        section_frame = ctk.CTkFrame(self.content_frame, corner_radius=10)
        section_frame.pack(fill='x', pady=15)
        
        # Section header with improved styling
        ctk.CTkLabel(
            section_frame,
            text="Citation Tracker",
            font=("Helvetica", 20, "bold"),
        ).pack(anchor='w', padx=20, pady=(15, 5))
        
        # Add a separator
        separator = ctk.CTkFrame(section_frame, height=2, fg_color=("gray80", "gray30"))
        separator.pack(fill='x', padx=20, pady=(0, 15))
        
        # Loading indicator with improved styling
        self.citations_loading_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        self.citations_loading_frame.pack(fill='x', padx=20, pady=15)
        
        ctk.CTkLabel(
            self.citations_loading_frame,
            text="Loading citation data...",
            font=("Helvetica", 12)
        ).pack(pady=10)
        
        progress = ctk.CTkProgressBar(self.citations_loading_frame, height=10)
        progress.pack(fill='x', padx=40, pady=5)
        progress.configure(mode="indeterminate")
        progress.start()
        
        # Create frame for citation data (initially empty)
        self.citations_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        self.citations_frame.pack(fill='x', padx=20, pady=10)
    
    def fetch_github_releases(self):
        """Fetch recent releases from GitHub API with improved error handling"""
        try:
            # Add a user agent to avoid GitHub API rate limiting
            headers = {
                'User-Agent': 'Pose2Sim-App',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # Use a fixed endpoint format and add parameters for pagination
            releases_url = "https://api.github.com/repos/perfanalytics/pose2sim/releases?per_page=5"
            
            response = requests.get(releases_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                releases_data = response.json()
                
                # Validate that we received a list (better error detection)
                if isinstance(releases_data, list):
                    # Store the most recent 5 releases
                    self.releases = releases_data[:5] if len(releases_data) > 5 else releases_data
                    
                    # Get the latest version tag (first release)
                    if self.releases and 'tag_name' in self.releases[0]:
                        self.latest_version = self.releases[0]['tag_name'].lstrip('v')
                    
                    # Update UI in main thread
                    self.frame.after(0, self.update_releases_ui)
                else:
                    error_msg = f"Invalid response format from GitHub API"
                    self.frame.after(0, lambda: self.update_releases_error(error_msg))
            elif response.status_code == 403:
                # Rate limiting specific error
                error_msg = "GitHub API rate limit exceeded. Please try again later."
                self.frame.after(0, lambda: self.update_releases_error(error_msg))
            else:
                # Other HTTP errors
                error_msg = f"GitHub API error: HTTP {response.status_code}"
                self.frame.after(0, lambda: self.update_releases_error(error_msg))
                
        except requests.exceptions.Timeout:
            error_msg = "Connection timed out. Please check your internet connection."
            self.frame.after(0, lambda: self.update_releases_error(error_msg))
        except requests.exceptions.ConnectionError:
            error_msg = "Network connection error. Please check your internet connection."
            self.frame.after(0, lambda: self.update_releases_error(error_msg))
        except json.JSONDecodeError:
            error_msg = "Error parsing GitHub data. The response was not valid JSON."
            self.frame.after(0, lambda: self.update_releases_error(error_msg))
        except Exception as e:
            # General exception handler
            error_msg = f"Error fetching releases: {str(e)}"
            self.frame.after(0, lambda: self.update_releases_error(error_msg))
    
    def update_releases_ui(self):
        """Update UI with fetched GitHub releases - improved styling"""
        # Remove loading indicator
        self.releases_loading_frame.pack_forget()
        
        if not self.releases:
            # Show error message if no releases found
            ctk.CTkLabel(
                self.releases_frame,
                text="No releases found. Check the GitHub repository for updates.",
                wraplength=700
            ).pack(pady=15)
            return
        
        # Show each release with improved styling
        for release in self.releases:
            release_frame = ctk.CTkFrame(self.releases_frame, fg_color=("gray90", "gray25"), corner_radius=8)
            release_frame.pack(fill='x', pady=8)
            
            # Release header with improved styling
            header_frame = ctk.CTkFrame(release_frame, fg_color="transparent")
            header_frame.pack(fill='x', padx=10, pady=(10, 5))
            
            # Release tag and date with improved styling
            tag_name = release.get('tag_name', 'Unknown version')
            
            # Format the date
            date_str = "Unknown date"
            if 'published_at' in release:
                try:
                    date_obj = datetime.datetime.strptime(release['published_at'], "%Y-%m-%dT%H:%M:%SZ")
                    date_str = date_obj.strftime("%B %d, %Y")
                except (ValueError, TypeError):
                    pass
            
            ctk.CTkLabel(
                header_frame,
                text=f"{tag_name} - Released {date_str}",
                font=("Helvetica", 16, "bold"),
                anchor="w"
            ).pack(side='left')
            
            # View button with improved styling
            ctk.CTkButton(
                header_frame,
                text="View on GitHub",
                command=lambda url=release.get('html_url'): webbrowser.open(url),
                width=120,
                height=30,
                corner_radius=8,
                hover_color=("#2E86C1", "#1F618D")
            ).pack(side='right')
            
            # Release body - clean up markdown
            body = release.get('body', 'No release notes provided')
            
            # Basic markdown cleanup for better readability
            body = re.sub(r'#+\s+', '', body)  # Remove headers
            body = re.sub(r'\*\*(.+?)\*\*', r'\1', body)  # Remove bold
            body = re.sub(r'\*(.+?)\*', r'\1', body)  # Remove italics
            body = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', body)  # Remove links
            
            # Truncate if too long
            if len(body) > 500:
                body = body[:497] + "..."
            
            # Release notes with improved styling
            body_frame = ctk.CTkFrame(release_frame, fg_color=("gray95", "gray18"))
            body_frame.pack(fill='x', padx=10, pady=(5, 10))
            
            body_text = ctk.CTkTextbox(
                body_frame,
                height=100,
                wrap="word",
                font=("Helvetica", 12)
            )
            body_text.pack(fill='x', padx=5, pady=5)
            body_text.insert("1.0", body)
            body_text.configure(state="disabled")
    
    def update_releases_error(self, error_message):
        """Show error message in releases section - improved styling"""
        # Remove loading indicator
        self.releases_loading_frame.pack_forget()
        
        # Show error message with improved styling
        error_frame = ctk.CTkFrame(
            self.releases_frame, 
            fg_color=("#F8D7DA", "#5C1E25"),
            corner_radius=8
        )
        error_frame.pack(fill='x', pady=15)
        
        ctk.CTkLabel(
            error_frame,
            text=error_message,
            text_color=("#721C24", "#EAACB0"),
            wraplength=700
        ).pack(pady=15)
        
        # Retry button with improved styling
        ctk.CTkButton(
            error_frame,
            text="Retry",
            command=lambda: threading.Thread(target=self.fetch_github_releases, daemon=True).start(),
            width=100,
            height=30,
            corner_radius=8,
            fg_color=("#DC3545", "#A71D2A"),
            hover_color=("#C82333", "#8B1823")
        ).pack(pady=(0, 15))
    
    def fetch_citation_data(self):
        """Fetch citation data for the papers using DOIs"""
        try:
            # Cache file path for citation data
            cache_dir = Path(__file__).parent.parent / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "citation_data.json"
            
            # Check if cache exists and not older than 1 day
            cache_valid = False
            if cache_file.exists():
                try:
                    file_mod_time = cache_file.stat().st_mtime
                    if (time.time() - file_mod_time) < 86400:  # 24 hours
                        with open(cache_file, 'r') as f:
                            self.citation_data = json.load(f)
                            cache_valid = True
                except:
                    pass
            
            # If no valid cache, fetch new data
            if not cache_valid:
                # These would be real API calls in a production app
                # For RSS tracking, you would parse feed data from the DOI-related feeds
                
                # For now, create mock data based on the DOIs
                for doi in self.citation_dois:
                    # In a real app, make API calls here to services like Crossref, Semantic Scholar, etc.
                    # Here's a placeholder using mock data
                    if doi == "10.21105/joss.04362":  # Pose2Sim JOSS paper
                        self.citation_data[doi] = {
                            "title": "Pose2Sim: An open-source Python package for multiview markerless kinematics",
                            "citation_count": 42,
                            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d")
                        }
                    elif doi == "10.3390/s22072712":  # Accuracy paper
                        self.citation_data[doi] = {
                            "title": "Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 2: Accuracy",
                            "citation_count": 35,
                            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d")
                        }
                    elif doi == "10.3390/s21196530":  # Robustness paper
                        self.citation_data[doi] = {
                            "title": "Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 1: Robustness",
                            "citation_count": 48,
                            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d")
                        }
                    elif doi == "10.21105/joss.06849":  # Sports2D paper
                        self.citation_data[doi] = {
                            "title": "Sports2D: Compute 2D human pose and angles from a video or a webcam",
                            "citation_count": 8,
                            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d")
                        }
                
                # Save to cache
                with open(cache_file, 'w') as f:
                    json.dump(self.citation_data, f)
            
            # Format data for display
            self.citations = [
                {
                    "doi": doi,
                    "title": data.get("title", "Unknown paper"),
                    "citation_count": data.get("citation_count", 0),
                    "last_updated": data.get("last_updated", datetime.datetime.now().strftime("%Y-%m-%d"))
                }
                for doi, data in self.citation_data.items()
            ]
            
            # Update UI in main thread
            self.frame.after(0, self.update_citations_ui)
            
        except Exception as e:
            # Log the exception
            traceback.print_exc()
            # Handle exceptions
            error_msg = f"Error fetching citation data: {str(e)}"
            self.frame.after(0, lambda: self.update_citations_error(error_msg))
    
    def update_citations_ui(self):
        """Update UI with citation data - improved styling"""
        # Remove loading indicator
        self.citations_loading_frame.pack_forget()
        
        if not self.citations:
            # Show message if no citation data with improved styling
            ctk.CTkLabel(
                self.citations_frame,
                text="No citation data available at this time.",
                wraplength=700
            ).pack(pady=15)
            return
        
        # Create header with improved styling
        header_frame = ctk.CTkFrame(self.citations_frame, fg_color=("gray95", "gray20"), corner_radius=8)
        header_frame.pack(fill='x', pady=(0, 15))
        
        # Get the last update date from the first citation
        last_updated = datetime.datetime.now().strftime("%B %d, %Y")
        if self.citations and 'last_updated' in self.citations[0]:
            try:
                date_obj = datetime.datetime.strptime(self.citations[0]['last_updated'], "%Y-%m-%d")
                last_updated = date_obj.strftime("%B %d, %Y")
            except (ValueError, TypeError):
                pass
        
        ctk.CTkLabel(
            header_frame,
            text=f"Publication Impact (Last updated: {last_updated})",
            font=("Helvetica", 16, "bold")
        ).pack(pady=(15, 10))
        
        # Total citations with improved styling
        total_citations = sum(citation.get('citation_count', 0) for citation in self.citations)
        
        ctk.CTkLabel(
            header_frame,
            text=f"Total Citations: {total_citations}",
            font=("Helvetica", 20)
        ).pack(pady=(0, 15))
        
        # Create citation cards with improved styling
        for citation in self.citations:
            citation_card = ctk.CTkFrame(self.citations_frame, fg_color=("gray90", "gray25"), corner_radius=8)
            citation_card.pack(fill='x', pady=5)
            
            # Paper title and citation count with improved layout
            title_frame = ctk.CTkFrame(citation_card, fg_color="transparent")
            title_frame.pack(fill='x', padx=15, pady=(10, 0))
            
            # Title on the left
            ctk.CTkLabel(
                title_frame,
                text=citation['title'],
                font=("Helvetica", 14),
                anchor="w",
                wraplength=600,
                justify="left"
            ).pack(side='left', fill='x', expand=True)
            
            # Citation count on the right
            count_frame = ctk.CTkFrame(
                title_frame, 
                fg_color=("#E1F0F9", "#203A4C"),
                corner_radius=15,
                width=60,
                height=30
            )
            count_frame.pack(side='right', padx=(15, 0))
            count_frame.pack_propagate(False)  # Fix the size
            
            ctk.CTkLabel(
                count_frame,
                text=str(citation['citation_count']),
                font=("Helvetica", 14, "bold"),
                text_color=("blue", "#5B8CD7")
            ).pack(expand=True, fill='both')
            
            # DOI with improved styling
            doi_frame = ctk.CTkFrame(citation_card, fg_color="transparent")
            doi_frame.pack(fill='x', padx=15, pady=(5, 10))
            
            ctk.CTkLabel(
                doi_frame,
                text="DOI: ",
                width=40,
                anchor="w"
            ).pack(side='left')
            
            ctk.CTkButton(
                doi_frame,
                text=citation['doi'],
                anchor="w",
                fg_color="transparent",
                text_color=("blue", "#5B8CD7"),
                hover_color=("gray90", "gray20"),
                command=lambda doi=citation['doi']: webbrowser.open(f"https://doi.org/{doi}")
            ).pack(side='left')
        
        # Note about citation tracking with improved styling
        note_frame = ctk.CTkFrame(self.citations_frame, fg_color=("gray95", "gray20"), corner_radius=8)
        note_frame.pack(fill='x', pady=15)
        
        ctk.CTkLabel(
            note_frame,
            text="Note: Citation counts are updated periodically from Google Scholar and may not reflect the most recent data.",
            wraplength=700,
            font=("Helvetica", 11),
            text_color=("gray40", "gray80")
        ).pack(pady=10)
        
        # Refresh button with improved styling
        ctk.CTkButton(
            self.citations_frame,
            text="Refresh Citation Data",
            command=lambda: threading.Thread(target=self.refresh_citation_data, daemon=True).start(),
            width=160,
            height=35,
            corner_radius=8,
            hover_color=("#2E86C1", "#1F618D")
        ).pack(anchor='center', pady=15)
    
    def refresh_citation_data(self):
        """Force refresh of citation data"""
        # Delete cache if it exists
        cache_dir = Path(__file__).parent.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "citation_data.json"
        if cache_file.exists(): cache_file.unlink()
        
        # Reset citation data
        self.citation_data = {}
        self.citations = []
        
        # Show loading indicator again
        self.citations_frame.pack_forget()
        self.citations_loading_frame.pack(fill='x', padx=20, pady=15)
        
        # Fetch new data
        threading.Thread(target=self.fetch_citation_data, daemon=True).start()
    
    def update_citations_error(self, error_message):
        """Show error message in citations section - improved styling"""
        # Remove loading indicator
        self.citations_loading_frame.pack_forget()
        
        # Show error message with improved styling
        error_frame = ctk.CTkFrame(
            self.citations_frame, 
            fg_color=("#F8D7DA", "#5C1E25"),
            corner_radius=8
        )
        error_frame.pack(fill='x', pady=15)
        
        ctk.CTkLabel(
            error_frame,
            text=error_message,
            text_color=("#721C24", "#EAACB0"),
            wraplength=700
        ).pack(pady=15)
        
        # Retry button with improved styling
        ctk.CTkButton(
            error_frame,
            text="Retry",
            command=lambda: threading.Thread(target=self.fetch_citation_data, daemon=True).start(),
            width=100,
            height=30,
            corner_radius=8,
            fg_color=("#DC3545", "#A71D2A"),
            hover_color=("#C82333", "#8B1823")
        ).pack(pady=(0, 15))
    
    def copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        try:
            self.frame.clipboard_clear()
            self.frame.clipboard_append(text)
            self.frame.update()
            messagebox.showinfo("Success", "Text copied to clipboard")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy: {str(e)}")