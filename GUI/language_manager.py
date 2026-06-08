class LanguageManager:
    def __init__(self):
        # Define dictionaries for each language
        self.translations = {
            'en': {
                # General UI elements
                'app_title': "Pose2Sim Configuration Tool",
                'next': "Next",
                'previous': "Previous",
                'save': "Save",
                'cancel': "Cancel",
                'confirm': "Confirm",
                'error': "Error",
                'warning': "Warning",
                'info': "Information",
                'success': "Success",
                'select': "Select",
                
                # Welcome screen
                'welcome_title': "Welcome to Pose2Sim",
                'welcome_subtitle': "3D Pose Estimation Configuration Tool",
                'select_language': "Select Language",
                'select_analysis_mode': "Select Analysis Mode",
                '2d_analysis': "2D Analysis",
                '3d_analysis': "3D Analysis",
                "single_camera": "Single camera",
                "multi_camera": "Two cameras or more",
                '2d_description': "Track subjects in 2D space from a single camera view.",
                '3d_description': "Reconstruct 3D motion using multiple synchronized cameras.",
                'single_mode': "Single Mode",
                'batch_mode': "Batch Mode",
                'enter_participant_name': "Enter Participant Name:",
                'enter_trials_number': "Enter Number of Trials:",
                
                # Calibration tab
                'calibration_tab': "Calibration",
                'calibration_type': "Calibration Type:",
                'calculate': "Calculate",
                'convert': "Convert",
                'num_cameras': "Number of Cameras:",
                'checkerboard_width': "Checkerboard Width:",
                'checkerboard_height': "Checkerboard Height:",
                'square_size': "Square Size (mm):",
                'video_extension': "Video/Image Extension:",
                'proceed_calibration': "Proceed with Calibration",
                
                # Prepare Video tab
                'prepare_video_tab': "Prepare Video",
                'only_checkerboard': "Do your videos contain only checkerboard images?",
                'time_interval': "Enter time interval in seconds for image extraction:",
                'image_format': "Enter the image format (e.g., png, jpg):",
                'proceed_prepare_video': "Proceed with Prepare Video",
                
                # Pose Model tab
                'pose_model_tab': "Pose Estimation",
                'multiple_persons': "Multiple Persons:",
                'single_person': "Single Person:",
                'participant_height': "Participant Height (m):",
                'participant_mass': "Participant Mass (kg):",
                'pose_model_selection': "Pose Model Selection:",
                'mode': "Mode:",
                'proceed_pose_estimation': "Proceed with Pose Estimation",
                
                # Synchronization tab
                'synchronization_tab': "Synchronization",
                'skip_sync': "Skip synchronization part? (Videos are already synchronized)",
                'select_keypoints': "Select keypoints to consider for synchronization:",
                'approx_time': "Do you want to specify approximate times of movement?",
                'time_range': "Time interval around max speed (seconds):",
                'likelihood_threshold': "Likelihood Threshold:",
                'filter_cutoff': "Filter Cutoff (Hz):",
                'filter_order': "Filter Order:",
                'save_sync_settings': "Save Synchronization Settings",
                
                # Advanced tab
                'advanced_tab': "Advanced Configuration",
                'frame_rate': "Frame Rate (fps):",
                'frame_range': "Frame Range (e.g., [10, 300]):",
                'person_association': "Person Association",
                'triangulation': "Triangulation",
                'filtering': "Filtering",
                'marker_augmentation': "Marker Augmentation",
                'kinematics': "Kinematics",
                'save_advanced_settings': "Save Advanced Settings",
                
                # Activation tab
                'activation_tab': "Activation",
                'launch_options': "Choose how you want to launch Pose2Sim:",
                'launch_cmd': "Launch with CMD",
                'launch_conda': "Run analysis", #"Launch with Anaconda Prompt",
                'launch_powershell': "Launch with PowerShell",
                
                # Batch tab
                'batch_tab': "Batch Configuration",
                'trial_config': "Trial-Specific Configuration",
                'batch_info': "Configure trial-specific parameters. Other settings will be inherited from the main configuration.",
                'save_trial_config': "Save Trial Configuration",
            },
            'fr': {
                # General UI elements
                'app_title': "Outil de Configuration Pose2Sim",
                'next': "Suivant",
                'previous': "Précédent",
                'save': "Sauvegarder",
                'cancel': "Annuler",
                'confirm': "Confirmer",
                'error': "Erreur",
                'warning': "Avertissement",
                'info': "Information",
                'success': "Succès",
                'select': "Sélectionner",
                
                # Welcome screen
                'welcome_title': "Bienvenue sur Pose2Sim",
                'welcome_subtitle': "Outil de Configuration de l'Estimation de Pose 3D",
                'select_language': "Sélectionnez la Langue",
                'select_analysis_mode': "Sélectionnez le Mode d'Analyse",
                '2d_analysis': "Analyse 2D",
                '3d_analysis': "Analyse 3D",
                "single_camera": "Une seule caméra",
                "multi_camera": "Au moins deux caméras",
                '2d_description': "Suivez des sujets en 2D à partir d'une seule caméra.",
                '3d_description': "Reconstruisez des mouvements en 3D avec plusieurs caméras synchronisées.",
                'single_mode': "Mode Simple",
                'batch_mode': "Mode Batch",
                'enter_participant_name': "Entrez le Nom du Participant :",
                'enter_trials_number': "Entrez le Nombre d'Essais :",
                
                # Calibration tab
                'calibration_tab': "Calibration",
                'calibration_type': "Type de Calibration :",
                'calculate': "Calculer",
                'convert': "Convertir",
                'num_cameras': "Nombre de Caméras :",
                'checkerboard_width': "Largeur de l'Échiquier :",
                'checkerboard_height': "Hauteur de l'Échiquier :",
                'square_size': "Taille du Carré (mm) :",
                'video_extension': "Extension Vidéo/Image :",
                'proceed_calibration': "Procéder à la Calibration",
                
                # Prepare Video tab
                'prepare_video_tab': "Préparer la Vidéo",
                'only_checkerboard': "Vos vidéos contiennent-elles uniquement des images d'échiquier ?",
                'time_interval': "Entrez l'intervalle de temps en secondes pour l'extraction d'images :",
                'image_format': "Entrez le format d'image (ex : png, jpg) :",
                'proceed_prepare_video': "Procéder à la Préparation Vidéo",
                
                # Pose Model tab
                'pose_model_tab': "Estimation de Pose",
                'multiple_persons': "Plusieurs Personnes :",
                'single_person': "Personne Unique :",
                'participant_height': "Taille du Participant (m) :",
                'participant_mass': "Masse du Participant (kg) :",
                'pose_model_selection': "Sélection du Modèle de Pose :",
                'mode': "Mode :",
                'proceed_pose_estimation': "Procéder à l'Estimation de Pose",
                
                # Synchronization tab
                'synchronization_tab': "Synchronisation",
                'skip_sync': "Passer la synchronisation ? (Les vidéos sont déjà synchronisées)",
                'select_keypoints': "Sélectionnez les points clés à considérer pour la synchronisation :",
                'approx_time': "Voulez-vous spécifier des temps approximatifs de mouvement ?",
                'time_range': "Intervalle de temps autour de la vitesse max (secondes) :",
                'likelihood_threshold': "Seuil de Vraisemblance :",
                'filter_cutoff': "Fréquence de Coupure du Filtre (Hz) :",
                'filter_order': "Ordre du Filtre :",
                'save_sync_settings': "Sauvegarder les Paramètres de Synchronisation",
                
                # Advanced tab
                'advanced_tab': "Configuration Avancée",
                'frame_rate': "Fréquence d'Images (fps) :",
                'frame_range': "Plage d'Images (ex : [10, 300]) :",
                'person_association': "Association de Personne",
                'triangulation': "Triangulation",
                'filtering': "Filtrage",
                'marker_augmentation': "Augmentation de Marqueurs",
                'kinematics': "Cinématique",
                'save_advanced_settings': "Sauvegarder les Paramètres Avancés",
                
                # Activation tab
                'activation_tab': "Activation",
                'launch_options': "Choisissez comment lancer Pose2Sim :",
                'launch_cmd': "Lancer avec CMD",
                'launch_conda': "Lancer l'analyse", #"Lancer avec Anaconda Prompt",
                'launch_powershell': "Lancer avec PowerShell",
                
                # Batch tab
                'batch_tab': "Configuration Batch",
                'trial_config': "Configuration Spécifique à l'Essai",
                'batch_info': "Configurez les paramètres spécifiques à l'essai. Les autres paramètres seront hérités de la configuration principale.",
                'save_trial_config': "Sauvegarder la Configuration de l'Essai",
            }
        }
        
        # Default language is English
        self.current_language = 'en'
    
    def set_language(self, lang_code):
        """Sets the current language"""
        if lang_code in self.translations:
            self.current_language = lang_code
    
    def get_text(self, key):
        """Gets the text for a given key in the current language"""
        if key in self.translations[self.current_language]:
            return self.translations[self.current_language][key]
        else:
            # Return the key itself if translation not found
            return key