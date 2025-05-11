import tkinter as tk
import customtkinter as ctk

class IntroWindow:
    """Displays an animated introduction window for the Pose2Sim GUI."""
    def __init__(self, color='dark'):
        """Initializes the IntroWindow.

        Args:
            color (str, optional): The color theme for the window. 
                                   Can be 'light' or 'dark'. Defaults to 'dark'.
        """
        # Set color parameters based on choice
        if color.lower() == 'light':
            self.main_color = 'black'
            self.shadow_color = '#404040'  # Dark gray
            self.main_color_value = 0      
            self.shadow_color_value = 64   
            self.bg_color = '#F0F0F0'      # Light gray
        elif color.lower() == 'dark':
            self.main_color = 'white'
            self.shadow_color = '#AAAAAA'  # Light gray 
            self.main_color_value = 255    
            self.shadow_color_value = 170  
            self.bg_color = '#1A1A1A'      # Very dark gray


        # Create the intro window
        self.root = ctk.CTk()
        self.root.title("Welcome to Pose2Sim")
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size (80% of screen size)
        # window_width = int(screen_width * 0.7)
        # window_height = int(screen_height * 0.7)
        
        # Size should be same as app.py
        window_width = 1300 
        window_height = 800
        
        # Calculate position for center of screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Set background color
        self.root.configure(fg_color=self.bg_color)
        
        # Create canvas for animation
        self.canvas = tk.Canvas(self.root, bg=self.bg_color, highlightthickness=0)
        self.canvas.pack(expand=True, fill='both')
        
        # Create individual letters with initial opacity
        letters = ['P', 'o', 's', 'e', '2', 'S', 'i', 'm']
        self.text_ids = []
        self.shadow_ids = []  # Add shadow text IDs
        spacing = 50  # Adjust spacing between letters
        total_width = len(letters) * spacing
        start_x = window_width/2 - total_width/2
        
        for i, letter in enumerate(letters):
            # Adjust font size for P and S
            font_size = 78 if letter in ['P', '2', 'S'] else 70
            
            if letter == 'i' or letter == 'm':
                spacing = 49
            elif letter == 'i':
                spacing = 55
            elif letter == 'S':
                spacing = 51
            elif letter == 's':
                spacing = 52
            elif letter == 'o':
                spacing = 54
            # Create shadow text (slightly offset)
            shadow_id = self.canvas.create_text(
                start_x + i * spacing + 2,  # Offset by 2 pixels right
                window_height/2 + 2,        # Offset by 2 pixels down
                text=letter,
                font=('Helvetica', font_size, 'bold'),
                fill=self.shadow_color,
                state='hidden'
            )
            self.shadow_ids.append(shadow_id)
            
            # Create main text
            text_id = self.canvas.create_text(
                start_x + i * spacing,
                window_height/2,
                text=letter,
                font=('Helvetica', font_size, 'bold'),
                fill=self.main_color,
                state='hidden'
            )

            self.text_ids.append(text_id)
            spacing = 50  # Reset spacing for other letters
        
        # Store animation parameters
        self.opacity = 0
        self.fadein_step = 0.008 # Time step for fade-in/out
        self.fadeout_step = 0.0018
        self.current_group = 0  # Track current group (0: Pose, 1: 2, 2: Sim)
        self.animation_done = False
        self.after_id = None
        
        # Create subtitle text
        subtitle = "markerless motion capture solution"
        subtitle_font_size = 26
        
        self.subtitle_shadow_id = self.canvas.create_text(
            window_width/2 - 30 + 1,
            window_height/2 + 60 + 1,
            text=subtitle,
            font=('Helvetica', subtitle_font_size),
            fill=self.shadow_color,
            state='hidden'
        )
        
        self.subtitle_id = self.canvas.create_text(
            window_width/2 - 30,  
            window_height/2 + 60, 
            text=subtitle,
            font=('Helvetica', subtitle_font_size),
            fill=self.main_color,
            state='hidden'
        )
        
        # Define letter groups (including shadows)
        self.groups = [
            list(zip(self.text_ids[:4], self.shadow_ids[:4])),  # Pose
            list(zip([self.text_ids[4]], [self.shadow_ids[4]])),  # 2
            list(zip(self.text_ids[5:], self.shadow_ids[5:]))   # Sim
        ]
        
        # Add subtitle as the 4th group
        self.groups.append([(self.subtitle_id, self.subtitle_shadow_id)])
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start the fade-in animation after a short delay
        self.after_id = self.root.after(150, self.fade_in)

    def on_closing(self):
        """Handles the window closing event."""
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.animation_done = True
        self.root.destroy()

    def fade_in(self):
        """Animates the fade-in effect for the text elements."""
        if not self.root.winfo_exists():
            return
        if self.current_group < len(self.groups):
            if self.opacity < 1:
                self.opacity += self.fadein_step
                # Make current group visible and set opacity
                for text_id, shadow_id in self.groups[self.current_group]:
                    self.canvas.itemconfig(shadow_id, state='normal')
                    self.canvas.itemconfig(text_id, state='normal')
                    
                    # Calculate color values based on mode
                    if self.main_color == 'black':
                        # Light mode (black text on #F0F0F0 background)
                        main_r = int(240 * (1 - self.opacity) + 0 * self.opacity)  # Fade from bg color (240) to black (0)
                        shadow_r = int(240 * (1 - self.opacity) + 64 * self.opacity)  # Fade from bg color to shadow
                        hex_color = f'#{main_r:02x}{main_r:02x}{main_r:02x}'
                        shadow_color = f'#{shadow_r:02x}{shadow_r:02x}{shadow_r:02x}'
                    elif self.main_color == 'white':
                        # Dark mode (white text on #1A1A1A background)
                        main_r = int(26 * (1 - self.opacity) + 255 * self.opacity)  # Fade from bg color (26) to white (255)
                        shadow_r = int(26 * (1 - self.opacity) + self.shadow_color_value * self.opacity)  # Fade from bg to shadow
                        hex_color = f'#{main_r:02x}{main_r:02x}{main_r:02x}'
                        shadow_color = f'#{shadow_r:02x}{shadow_r:02x}{shadow_r:02x}'
                        
                    self.canvas.itemconfig(shadow_id, fill=shadow_color)
                    self.canvas.itemconfig(text_id, fill=hex_color)
                self.after_id = self.root.after(1, self.fade_in)
            else:
                self.opacity = 0
                self.current_group += 1
                self.after_id = self.root.after(1, self.fade_in)
        else:
            self.opacity = 1
            self.fade_out()

    def fade_out(self):
        """Animates the fade-out effect for the text elements and closes the window."""
        if not self.root.winfo_exists():
            return
        if self.opacity > 0:
            self.opacity -= self.fadeout_step
            # Update all letters opacity together
            
            # Calculate color values based on mode
            if self.main_color == 'black':
                # Light mode (black text on #F0F0F0 background)
                main_r = int(240 * (1 - self.opacity) + 0 * self.opacity)  # Fade from bg color (240) to black (0)
                shadow_r = int(240 * (1 - self.opacity) + 64 * self.opacity)  # Fade from bg color to shadow
                hex_color = f'#{main_r:02x}{main_r:02x}{main_r:02x}'
                shadow_color = f'#{shadow_r:02x}{shadow_r:02x}{shadow_r:02x}'
            elif self.main_color == 'white':
                # Dark mode (white text on #1A1A1A background)
                main_r = int(26 * (1 - self.opacity) + 255 * self.opacity)  # Fade from bg color (26) to white (255)
                shadow_r = int(26 * (1 - self.opacity) + self.shadow_color_value * self.opacity)  # Fade from bg to shadow
                hex_color = f'#{main_r:02x}{main_r:02x}{main_r:02x}'
                shadow_color = f'#{shadow_r:02x}{shadow_r:02x}{shadow_r:02x}'

                
            for text_id, shadow_id in zip(self.text_ids, self.shadow_ids):
                self.canvas.itemconfig(shadow_id, fill=shadow_color)
                self.canvas.itemconfig(text_id, fill=hex_color)
            self.canvas.itemconfig(self.subtitle_shadow_id, fill=shadow_color)
            self.canvas.itemconfig(self.subtitle_id, fill=hex_color)
            self.after_id = self.root.after(1, self.fade_out)
        else:
            self.animation_done = True
            if self.root.winfo_exists():
                self.on_closing()

    def run(self):
        """Runs the main event loop for the intro window.

        Returns:
            bool: True when the animation is complete and the window is closed.
        """
        self.root.mainloop()

        if self.after_id:
            self.root.after_cancel(self.after_id)

        self.animation_done = True
        return self.animation_done

if __name__ == "__main__":

    intro = IntroWindow('dark')
    intro.run()
