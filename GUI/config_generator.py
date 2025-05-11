from pathlib import Path
import toml


class ConfigGenerator:
    def __init__(self):
        # Load templates from files to avoid comment issues
        self.config_3d_template_path = Path('templates') / '3d_config_template.toml'
        self.config_2d_template_path = Path('templates') /'2d_config_template.toml'
        
        # Create templates directory if it doesn't exist
        Path('templates').mkdir(parents=True, exist_ok=True)
        
        # Write the template files if they don't exist
        self.create_template_files()
    
    def create_template_files(self):
        """Create template files if they don't exist"""
        # Create 3D template file
        if not self.config_3d_template_path.exists():
            with open(self.config_3d_template_path, 'w', encoding='utf-8') as f:
                toml.dump(self.get_3d_template(), f)
        
        # Create 2D template file
        if not self.config_2d_template_path.exists():
            with open(self.config_2d_template_path, 'w', encoding='utf-8') as f:
                toml.dump(self.get_2d_template(), f)
    
    def get_3d_template(self):
        """Return the 3D configuration template"""
        from Pose2Sim import Pose2Sim
        config_template_3d = toml.load(Path(Pose2Sim.__file__).parent / 'Demo_SinglePerson' / 'Config.toml')
        return config_template_3d
    
    def get_2d_template(self):
         """Return the 2D configuration template"""
         from Sports2D import Sports2D
         config_template_2d = toml.load(Path(Sports2D.__file__).parent / 'Demo/Config_demo.toml')
         return config_template_2d
        
    
    def generate_2d_config(self, config_path, settings):
        """Generate configuration file for 2D analysis"""
        try:
            # Load the template
            config = toml.load(self.config_2d_template_path)
            
            # Debug print to check settings
            print("2D Settings being applied:", settings)
            
            # Update sections recursively
            for section_name, section_data in settings.items():
                if section_name not in config:
                    config[section_name] = {}
                
                self.update_nested_section(config[section_name], section_data)
            
            # Write the updated config with pretty formatting
            with open(config_path, 'w', encoding='utf-8') as f:
                toml.dump(config, f)
            
            print(f"2D Config file saved successfully to {config_path}")
            return True
        except Exception as e:
            print(f"Error generating 2D config: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_3d_config(self, config_path, settings):
        """Generate configuration file for 3D analysis"""
        try:
            # Parse the template
            config = toml.load(self.config_3d_template_path)
            
            # Debug print to check settings
            print("3D Settings being applied:", settings)
            
            # Update sections recursively
            for section_name, section_data in settings.items():
                if section_name not in config:
                    config[section_name] = {}
                
                self.update_nested_section(config[section_name], section_data)
            
            # Write the updated config with pretty formatting
            with open(config_path, 'w', encoding='utf-8') as f:
                toml.dump(config, f)
            
            print(f"3D Config file saved successfully to {config_path}")
            return True
        except Exception as e:
            print(f"Error generating 3D config: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_nested_section(self, config_section, settings_section):
        """Recursively update nested sections of the configuration file"""
        if not isinstance(settings_section, dict):
            return
        
        for key, value in settings_section.items():
            if isinstance(value, dict):
                # If the key doesn't exist in the config section, create it
                if key not in config_section:
                    config_section[key] = {}
                
                # Recursively update the subsection
                self.update_nested_section(config_section[key], value)
            else:
                # Update the value
                config_section[key] = value