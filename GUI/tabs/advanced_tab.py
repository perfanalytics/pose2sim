import customtkinter as ctk
from tkinter import messagebox
import toml
from pathlib import Path
from collections import OrderedDict

class ToolTip:
    """
    Create a tooltip for a given widget with word wrapping
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        
        self.tooltip_window = tw = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        # Bind hover events to the tooltip window itself
        tw.bind("<Enter>", self.on_tooltip_enter)
        tw.bind("<Leave>", self.hide_tooltip)
        
        # Wrap text to max 60 characters per line
        wrapped_text = self.wrap_text(self.text, 60)
        
        label = ctk.CTkLabel(
            tw, 
            text=wrapped_text,
            justify='left',
            fg_color=("#ffffe0", "#363636"),
            corner_radius=6,
            padx=10,
            pady=8
        )
        label.pack()
    
    def on_tooltip_enter(self, event=None):
        """When mouse enters tooltip, keep it visible"""
        pass
    
    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
    
    @staticmethod
    def wrap_text(text, width):
        """Wrap text to specified width"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)


class AdvancedTab:
    def __init__(self, parent, app, config_path=None, simplified=False):
        """
        Initialize the Advanced Configuration tab.
        
        Args:
            parent: Parent widget
            app: Main application instance
            config_path: Path to the TOML config file (optional, will auto-select if not provided)
            simplified: Whether to show simplified interface (for 2D analysis)
        """
        self.parent = parent
        self.app = app
        self.simplified = simplified
        
        # Auto-select config path based on simplified flag if not provided
        if config_path is None:
            if simplified:
                # 2D analysis config
                try:
                    import Sports2D
                    config_path = Path(Sports2D.__file__).parent / 'Demo' / 'Config_demo.toml'
                except ImportError:
                    raise ImportError("Sports2D module not found. Cannot load 2D config.")
            else:
                # 3D analysis config
                try:
                    import Pose2Sim
                    config_path = Path(Pose2Sim.__file__).parent / 'Demo_SinglePerson' / 'Config.toml'
                except ImportError:
                    raise ImportError("Pose2Sim module not found. Cannot load 3D config.")
        
        self.config_path = config_path
        
        # Create main frame
        self.frame = ctk.CTkFrame(parent)
        
        # Storage for widgets and variables
        self.config_data = OrderedDict()
        self.config_vars = {}
        self.widgets = {}
        
        # Load configuration
        if self.config_path:
            self.load_config(self.config_path)
        
        # Build the UI
        self.build_ui()
    
    def get_title(self):
        """Return the tab title"""
        return "Advanced Configuration"
    
    def load_config(self, filepath):
        """Load TOML configuration with comments"""
        self.config_data = self.parse_toml_with_comments(filepath)
    
    def parse_toml_with_comments(self, filepath):
        """
        Read config file, retain docstrings, preserve order
        Returns value, comment, and is_section for each key
        """
        def extract_items(table, lines, prefix=''):
            for key, value in table.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    # It's a section - mark it
                    result[full_key] = {
                        'value': OrderedDict(),
                        'comment': '',
                        'is_section': True,
                        'section_level': full_key.count('.')
                    }
                    # Recursively process nested items
                    extract_items(value, lines, full_key)
                else:
                    # It's a value - get inline and subsequent comments
                    comment_parts = []
                    
                    # Try to find multi-line comments in raw file
                    found_key = False
                    for i, line in enumerate(lines):
                        # Look for the key assignment
                        if not found_key and f'{key} =' in line:
                            found_key = True
                            
                            # Get the inline comment from this line
                            if '#' in line:
                                inline_comment = line.split('#', 1)[1].strip()
                                comment_parts.append(inline_comment)
                            
                            # Check subsequent lines for continuation comments
                            j = i + 1
                            while j < len(lines):
                                next_line = lines[j]
                                stripped = next_line.strip()
                                
                                # Stop at first blank line
                                if stripped == '':
                                    break
                                # Check if it's a comment-only line (continuation)
                                if stripped.startswith('#'):
                                    comment_parts.append(stripped.strip('# ').strip())
                                    j += 1
                                else:
                                    # Found non-comment line - stop
                                    break
                            break
                    
                    comment = '\n'.join(comment_parts) if comment_parts else ''
                    
                    result[full_key] = {
                        'value': value,
                        'comment': comment,
                        'is_section': False,
                        'parent_section': prefix
                    }
        
        # Parse TOML for values
        doc = toml.load(filepath)
        # Read raw lines to capture comments
        with open(filepath, 'r') as f:
            lines = f.readlines()
        result = OrderedDict()
        
        extract_items(doc, lines)
        return result
    
    def format_key_name(self, key):
        """
        Convert key like 'synchronization.likelihood_threshold' to 'Likelihood Threshold'
        """
        # Get the last part after the last dot
        if '.' in key:
            key = key.split('.')[-1]
        
        # Replace underscores with spaces and title case
        formatted = key.replace('_', ' ').title()
        
        return formatted
    
    def create_field_widget(self, parent, key, data):
        """Create appropriate widget based on value type"""
        value = data['value']
        comment = data['comment']
        
        # Create a frame for this field
        field_frame = ctk.CTkFrame(parent, fg_color="transparent")
        field_frame.pack(fill='x', pady=3, padx=5)
        
        # Create label with info icon
        label_frame = ctk.CTkFrame(field_frame, fg_color="transparent")
        label_frame.pack(side='left', fill='y')
        
        formatted_key = self.format_key_name(key)
        
        # Label
        label = ctk.CTkLabel(
            label_frame, 
            text=formatted_key + ":",
            width=200,
            anchor='w'
        )
        label.pack(side='left')
        
        # Info icon (if there's a comment)
        if comment:
            info_label = ctk.CTkLabel(
                label_frame,
                text=" ⓘ",
                width=20,
                text_color=("#2196F3", "#64B5F6"),
                cursor="question_arrow"
            )
            info_label.pack(side='left')
            ToolTip(info_label, comment)
        
        # Create appropriate input widget based on type
        if isinstance(value, bool):
            # Boolean -> Checkbox
            var = ctk.BooleanVar(value=value)
            widget = ctk.CTkCheckBox(
                field_frame,
                text="",
                variable=var,
                width=30
            )
            widget.pack(side='right', padx=5)
            
        elif isinstance(value, (int, float)):
            # Numeric -> Entry
            var = ctk.StringVar(value=str(value))
            widget = ctk.CTkEntry(
                field_frame,
                textvariable=var,
                width=150
            )
            widget.pack(side='right', padx=5)
            
        elif isinstance(value, str):
            # Check if it's a list of predefined values (could be enhanced)
            var = ctk.StringVar(value=value)
            widget = ctk.CTkEntry(
                field_frame,
                textvariable=var,
                width=150
            )
            widget.pack(side='right', padx=5)
            
        elif isinstance(value, list):
            # List -> Entry with string representation
            var = ctk.StringVar(value=str(value))
            widget = ctk.CTkEntry(
                field_frame,
                textvariable=var,
                width=150
            )
            widget.pack(side='right', padx=5)
            
        else:
            # Default -> Entry
            var = ctk.StringVar(value=str(value))
            widget = ctk.CTkEntry(
                field_frame,
                textvariable=var,
                width=150
            )
            widget.pack(side='right', padx=5)
        
        # Store references
        self.config_vars[key] = var
        self.widgets[key] = widget
        
        return field_frame
    
    def build_ui(self):
        """Build the user interface from config data"""
        # Create scrollable frame for content
        scrollable_frame = ctk.CTkScrollableFrame(self.frame)
        scrollable_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Header
        header_label = ctk.CTkLabel(
            scrollable_frame,
            text=self.get_title(),
            font=('Helvetica', 24, 'bold')
        )
        header_label.pack(pady=(0, 20))
        
        # If no config loaded, show message
        if not self.config_data:
            ctk.CTkLabel(
                scrollable_frame,
                text="No configuration file loaded",
                font=('Helvetica', 14)
            ).pack(pady=20)
            return
        
        # Organize items by section hierarchy
        sections = {}  # Store section frames by key
        
        for key, data in self.config_data.items():
            if data['is_section']:
                section_level = data['section_level']
                
                if section_level == 0:
                    # Main section (no dots in key)
                    section_frame = self.create_section_frame(
                        scrollable_frame,
                        self.format_key_name(key)
                    )
                    sections[key] = section_frame
                    
                elif section_level == 1:
                    # Subsection (one dot in key)
                    parent_key = key.rsplit('.', 1)[0]
                    parent_frame = sections.get(parent_key)
                    
                    if parent_frame:
                        subsection_frame = self.create_subsection_frame(
                            parent_frame,
                            self.format_key_name(key)
                        )
                        sections[key] = subsection_frame
                    else:
                        # Parent not found, create as main section
                        section_frame = self.create_section_frame(
                            scrollable_frame,
                            self.format_key_name(key)
                        )
                        sections[key] = section_frame
                        
                else:
                    # Deeper nesting - treat as subsection under parent
                    parent_key = key.rsplit('.', 1)[0]
                    parent_frame = sections.get(parent_key)
                    
                    if parent_frame:
                        subsection_frame = self.create_subsection_frame(
                            parent_frame,
                            self.format_key_name(key)
                        )
                        sections[key] = subsection_frame
        
        # Now add all fields to their respective sections
        for key, data in self.config_data.items():
            if not data['is_section']:
                # Find the appropriate parent section
                parent_section = data.get('parent_section', '')
                target_frame = sections.get(parent_section)
                
                # If parent not found, try grandparent
                if not target_frame and '.' in parent_section:
                    grandparent = parent_section.rsplit('.', 1)[0]
                    target_frame = sections.get(grandparent)
                
                # Last resort - use scrollable frame
                if not target_frame:
                    target_frame = scrollable_frame
                
                self.create_field_widget(target_frame, key, data)
        
        # Save Button
        save_button = ctk.CTkButton(
            self.frame,
            text="Save Configuration",
            command=self.save_settings,
            height=40,
            font=("Helvetica", 14),
            fg_color=("#4CAF50", "#2E7D32")
        )
        save_button.pack(side='bottom', pady=20)
    
    def create_section_frame(self, parent, title):
        """Create a main section frame with title"""
        section_frame = ctk.CTkFrame(parent)
        section_frame.pack(fill='x', pady=10, padx=5)
        
        # Title
        title_label = ctk.CTkLabel(
            section_frame,
            text=title,
            font=('Helvetica', 18, 'bold'),
            anchor='w'
        )
        title_label.pack(fill='x', padx=15, pady=(10, 5))
        
        # Separator
        separator = ctk.CTkFrame(section_frame, height=2, fg_color=("#CCCCCC", "#333333"))
        separator.pack(fill='x', padx=15, pady=(0, 10))
        
        # Content frame
        content_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        content_frame.pack(fill='x', padx=20, pady=(0, 10))
        
        return content_frame
    
    def create_subsection_frame(self, parent, title):
        """Create a subsection frame with subtitle"""
        subsection_frame = ctk.CTkFrame(parent, fg_color="transparent")
        subsection_frame.pack(fill='x', pady=8)
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            subsection_frame,
            text=title,
            font=('Helvetica', 14, 'bold'),
            anchor='w'
        )
        subtitle_label.pack(fill='x', pady=(5, 5))
        
        # Content frame
        content_frame = ctk.CTkFrame(subsection_frame, fg_color="transparent")
        content_frame.pack(fill='x', padx=10)
        
        return content_frame
    
    def get_settings(self):
        """Get the current configuration as a dictionary"""
        config_dict = {}
        
        for key, var in self.config_vars.items():
            # Reconstruct nested structure
            keys = key.split('.')
            current = config_dict
            
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            
            # Get value with type conversion
            value = var.get()
            original_value = self.config_data[key]['value']
            
            # Convert back to original type
            if isinstance(original_value, bool):
                current[keys[-1]] = value
            elif isinstance(original_value, int):
                try:
                    current[keys[-1]] = int(value)
                except ValueError:
                    current[keys[-1]] = value
            elif isinstance(original_value, float):
                try:
                    current[keys[-1]] = float(value)
                except ValueError:
                    current[keys[-1]] = value
            elif isinstance(original_value, list):
                try:
                    # Try to parse as Python literal
                    import ast
                    current[keys[-1]] = ast.literal_eval(value)
                except:
                    current[keys[-1]] = value
            else:
                current[keys[-1]] = value
        
        return config_dict
    
    def save_settings(self):
        """Save the configuration to file"""
        try:
            config_dict = self.get_settings()
            
            # Save to file if path provided
            if self.config_path:
                with open(self.config_path, 'w') as f:
                    toml.dump(config_dict, f)
                
                messagebox.showinfo(
                    "Success",
                    f"Configuration saved to {self.config_path}"
                )
            else:
                messagebox.showinfo(
                    "Success",
                    "Configuration updated successfully"
                )
            
            # Update app if needed
            if hasattr(self.app, 'update_tab_indicator'):
                self.app.update_tab_indicator('advanced', True)
            if hasattr(self.app, 'update_progress_bar'):
                self.app.update_progress_bar(85)
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to save configuration:\n{str(e)}"
            )
