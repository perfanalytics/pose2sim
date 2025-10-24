import customtkinter as ctk
from tkinter import messagebox
import ast

class AdvancedTab:
    def __init__(self, parent, app, simplified=False):
        """
        Initialize the Advanced Configuration tab.
        
        Args:
            parent: Parent widget
            app: Main application instance
            simplified: Whether to show a simplified interface for 2D analysis
        """
        self.parent = parent
        self.app = app
        self.simplified = simplified
        
        # Create main frame
        self.frame = ctk.CTkFrame(parent)
        
        # Initialize variables
        self.init_variables()
        
        # Build the UI
        self.build_ui()
    
    def get_title(self):
        """Return the tab title"""
        return self.app.lang_manager.get_text('advanced_tab')
    
    def init_variables(self):
        """Initialize all configuration variables"""
        # Basic settings
        self.frame_rate_var = ctk.StringVar(value='auto')
        self.frame_range_var = ctk.StringVar(value='auto')
        
        # Person Association Variables (3D only)
        self.likelihood_threshold_association_var = ctk.StringVar(value='0.3')
        self.reproj_error_threshold_association_var = ctk.StringVar(value='20')
        self.tracked_keypoint_var = ctk.StringVar(value='Neck')
        self.reconstruction_error_threshold_var = ctk.StringVar(value='0.1')
        self.min_affinity_var = ctk.StringVar(value='0.2')
        
        # Triangulation Variables (for 3D)
        self.reproj_error_threshold_triangulation_var = ctk.StringVar(value='15')
        self.likelihood_threshold_triangulation_var = ctk.StringVar(value='0.3')
        self.min_cameras_for_triangulation_var = ctk.StringVar(value='2')
        self.interp_if_gap_smaller_than_var = ctk.StringVar(value='20')
        self.interpolation_type_var = ctk.StringVar(value='linear')
        self.remove_incomplete_frames_var = ctk.BooleanVar(value=False)
        self.sections_to_keep_var = ctk.StringVar(value='all')
        self.fill_large_gaps_with_var = ctk.StringVar(value='last_value')
        self.show_interp_indices_var = ctk.BooleanVar(value=True)
        self.triangulation_make_c3d_var = ctk.BooleanVar(value=True)
        
        # Filtering Variables
        self.reject_outliers_var = ctk.BooleanVar(value=True)
        self.filter_var = ctk.BooleanVar(value=True)
        self.filter_type_var = ctk.StringVar(value='butterworth')
        self.display_figures_var = ctk.BooleanVar(value=True)
        self.save_filt_plots_var = ctk.BooleanVar(value=True)
        self.filtering_make_c3d_var = ctk.BooleanVar(value=True)
        
        # Butterworth Variables
        self.filter_cutoff_var = ctk.StringVar(value='6')
        self.filter_order_var = ctk.StringVar(value='4')
        
        # Kalman Variables
        self.kalman_trust_ratio_var = ctk.StringVar(value='500')
        self.kalman_smooth_var = ctk.BooleanVar(value=True)
        
        # GCV Spline Variables
        self.gcv_cut_off_frequency_var = ctk.StringVar(value='auto')
        self.gcv_smoothing_factor_var = ctk.StringVar(value='1.0')
        
        # Butterworth on Speed Variables
        self.butterworth_on_speed_order_var = ctk.StringVar(value='4')
        self.butterworth_on_speed_cut_off_frequency_var = ctk.StringVar(value='10')
        
        # Gaussian Variables
        self.gaussian_sigma_kernel_var = ctk.StringVar(value='1')
        
        # LOESS Variables
        self.LOESS_nb_values_used_var = ctk.StringVar(value='5')
        
        # Median Variables
        self.median_kernel_size_var = ctk.StringVar(value='3')
        
        # Marker Augmentation Variables (for 3D)
        self.feet_on_floor_var = ctk.BooleanVar(value=False)
        self.augmentation_make_c3d_var = ctk.BooleanVar(value=True)
        
        # Kinematics Variables
        self.use_augmentation_var = ctk.BooleanVar(value=True)
        self.use_simple_model_var = ctk.BooleanVar(value=False)
        self.use_contacts_muscles_var = ctk.BooleanVar(value=True)
        self.right_left_symmetry_var = ctk.BooleanVar(value=True)
        self.default_height_var = ctk.StringVar(value='1.7')
        self.remove_individual_scaling_setup_var = ctk.BooleanVar(value=True)
        self.remove_individual_IK_setup_var = ctk.BooleanVar(value=True)
        self.fastest_frames_to_remove_percent_var = ctk.StringVar(value='0.1')
        self.close_to_zero_speed_m_var = ctk.StringVar(value='0.2')
        self.large_hip_knee_angles_var = ctk.StringVar(value='45')
        self.trimmed_extrema_percent_var = ctk.StringVar(value='0.5')

        # 2D specific variables
        if self.simplified:
            # For 2D analysis
            self.slowmo_factor_var = ctk.StringVar(value='1')
            self.keypoint_likelihood_threshold_var = ctk.StringVar(value='0.3')
            self.average_likelihood_threshold_var = ctk.StringVar(value='0.5')
            self.keypoint_number_threshold_var = ctk.StringVar(value='0.3')
            self.interpolate_var = ctk.BooleanVar(value=True)
            self.interp_gap_smaller_than_var = ctk.StringVar(value='10')
            self.fill_large_gaps_with_2d_var = ctk.StringVar(value='last_value')
            self.show_graphs_var = ctk.BooleanVar(value=True)
            self.filter_type_2d_var = ctk.StringVar(value='butterworth')
    
    def get_settings(self):
        """Get the advanced configuration settings"""
        if self.simplified:
            # 2D mode settings
            settings = {
                'project': {
                    'frame_rate': self.frame_rate_var.get()
                },
                'pose': {
                    'slowmo_factor': int(self.slowmo_factor_var.get()),
                    'keypoint_likelihood_threshold': float(self.keypoint_likelihood_threshold_var.get()),
                    'average_likelihood_threshold': float(self.average_likelihood_threshold_var.get()),
                    'keypoint_number_threshold': float(self.keypoint_number_threshold_var.get())
                },
                'post-processing': {
                    'interpolate': self.interpolate_var.get(),
                    'interp_gap_smaller_than': int(self.interp_gap_smaller_than_var.get()),
                    'fill_large_gaps_with': self.fill_large_gaps_with_2d_var.get(),
                    'filter': self.filter_var.get(),
                    'show_graphs': self.show_graphs_var.get(),
                    'filter_type': self.filter_type_2d_var.get()
                },
                'logging': {
                    'use_custom_logging': False
                }
            }
            
            # Add filter-specific parameters
            filter_type = self.filter_type_2d_var.get()
            if filter_type == 'butterworth':
                settings['post-processing']['butterworth'] = {
                    'order': int(self.filter_order_var.get()),
                    'cut_off_frequency': float(self.filter_cutoff_var.get())
                }
            elif filter_type == 'gaussian':
                settings['post-processing']['gaussian'] = {
                    'sigma_kernel': float(self.gaussian_sigma_kernel_var.get())
                }
            elif filter_type == 'loess':
                settings['post-processing']['loess'] = {
                    'nb_values_used': int(self.LOESS_nb_values_used_var.get())
                }
            elif filter_type == 'median':
                settings['post-processing']['median'] = {
                    'kernel_size': int(self.median_kernel_size_var.get())
                }
            
            # Kinematics settings for 2D
            settings['kinematics'] = {
                'use_augmentation': self.use_augmentation_var.get(),
                'use_contacts_muscles': self.use_contacts_muscles_var.get(),
                'right_left_symmetry': self.right_left_symmetry_var.get(),
                'remove_individual_scaling_setup': self.remove_individual_scaling_setup_var.get(),
                'remove_individual_ik_setup': self.remove_individual_IK_setup_var.get()
            }
        else:
            # 3D mode settings
            settings = {
                'project': {
                    'frame_rate': self.frame_rate_var.get()
                },
                'personAssociation': {
                    'single_person': {
                        'likelihood_threshold_association': float(self.likelihood_threshold_association_var.get()),
                        'reproj_error_threshold_association': float(self.reproj_error_threshold_association_var.get()),
                        'tracked_keypoint': self.tracked_keypoint_var.get()
                    },
                    'multi_person': {
                        'reconstruction_error_threshold': float(self.reconstruction_error_threshold_var.get()),
                        'min_affinity': float(self.min_affinity_var.get())
                    }
                },
                'triangulation': {
                    'reproj_error_threshold_triangulation': float(self.reproj_error_threshold_triangulation_var.get()),
                    'likelihood_threshold_triangulation': float(self.likelihood_threshold_triangulation_var.get()),
                    'min_cameras_for_triangulation': int(self.min_cameras_for_triangulation_var.get()),
                    'interp_if_gap_smaller_than': int(self.interp_if_gap_smaller_than_var.get()),
                    'interpolation': self.interpolation_type_var.get(),
                    'remove_incomplete_frames': self.remove_incomplete_frames_var.get(),
                    'sections_to_keep': self.sections_to_keep_var.get(),
                    'fill_large_gaps_with': self.fill_large_gaps_with_var.get(),
                    'show_interp_indices': self.show_interp_indices_var.get(),
                    'make_c3d': self.triangulation_make_c3d_var.get()
                },
                'filtering': {
                    'reject_outliers': self.reject_outliers_var.get(),
                    'filter': self.filter_var.get(),
                    'type': self.filter_type_var.get(),
                    'display_figures': self.display_figures_var.get(),
                    'save_filt_plots': self.save_filt_plots_var.get(),
                    'make_c3d': self.filtering_make_c3d_var.get()
                },
                'markerAugmentation': {
                    'feet_on_floor': self.feet_on_floor_var.get(),
                    'make_c3d': self.augmentation_make_c3d_var.get()
                },
                'kinematics': {
                    'use_augmentation': self.use_augmentation_var.get(),
                    'use_simple_model': self.use_simple_model_var.get(),
                    'use_contacts_muscles': self.use_contacts_muscles_var.get(),
                    'right_left_symmetry': self.right_left_symmetry_var.get(),
                    'default_height': float(self.default_height_var.get()),
                    'remove_individual_scaling_setup': self.remove_individual_scaling_setup_var.get(),
                    'remove_individual_ik_setup': self.remove_individual_IK_setup_var.get(),
                    'fastest_frames_to_remove_percent': float(self.fastest_frames_to_remove_percent_var.get()),
                    'close_to_zero_speed_m': float(self.close_to_zero_speed_m_var.get()),
                    'large_hip_knee_angles': float(self.large_hip_knee_angles_var.get()),
                    'trimmed_extrema_percent': float(self.trimmed_extrema_percent_var.get())
                },
                'logging': {
                    'use_custom_logging': False
                }
            }
            
            # Try to parse frame range if it's not empty
            try:
                frame_range = self.frame_range_var.get()
                if frame_range and frame_range != 'auto' and frame_range != 'all':
                    parsed_range = ast.literal_eval(frame_range)
                    if isinstance(parsed_range, list):
                        settings['project']['frame_range'] = parsed_range
                    else:
                        settings['project']['frame_range'] = frame_range
                else:
                    settings['project']['frame_range'] = frame_range
            except:
                settings['project']['frame_range'] = 'auto'
            
            # Add filter-specific parameters
            filter_type = self.filter_type_var.get()
            if filter_type == 'butterworth':
                settings['filtering']['butterworth'] = {
                    'order': int(self.filter_order_var.get()),
                    'cut_off_frequency': float(self.filter_cutoff_var.get())
                }
            elif filter_type == 'kalman':
                settings['filtering']['kalman'] = {
                    'trust_ratio': float(self.kalman_trust_ratio_var.get()),
                    'smooth': self.kalman_smooth_var.get()
                }
            elif filter_type == 'gcv_spline':
                settings['filtering']['gcv_spline'] = {
                    'cut_off_frequency': self.gcv_cut_off_frequency_var.get(),
                    'smoothing_factor': float(self.gcv_smoothing_factor_var.get())
                }
            elif filter_type == 'butterworth_on_speed':
                settings['filtering']['butterworth_on_speed'] = {
                    'order': int(self.butterworth_on_speed_order_var.get()),
                    'cut_off_frequency': float(self.butterworth_on_speed_cut_off_frequency_var.get())
                }
            elif filter_type == 'gaussian':
                settings['filtering']['gaussian'] = {
                    'sigma_kernel': float(self.gaussian_sigma_kernel_var.get())
                }
            elif filter_type == 'loess':
                settings['filtering']['loess'] = {
                    'nb_values_used': int(self.LOESS_nb_values_used_var.get())
                }
            elif filter_type == 'median':
                settings['filtering']['median'] = {
                    'kernel_size': int(self.median_kernel_size_var.get())
                }
        
        return settings
    
    def build_ui(self):
        """Build the user interface"""
        # Create header
        header_frame = ctk.CTkScrollableFrame(self.frame)
        header_frame.pack(fill='both', expand=True, padx=0, pady=(0, 0))
        
        ctk.CTkLabel(
            header_frame, 
            text=self.get_title(), 
            font=('Helvetica', 24, 'bold')
        ).pack(fill='both', expand=True, padx=0, pady=0)

        if self.simplified:
            # Build simplified 2D interface
            self.build_2d_interface(header_frame)
        else:
            # Build full 3D interface
            self.build_3d_interface(header_frame)
        
        # Save Button
        save_button = ctk.CTkButton(
            self.frame,
            text=self.app.lang_manager.get_text('save_advanced_settings'),
            command=self.save_settings,
            height=40,
            font=("Helvetica", 14),
            fg_color=("#4CAF50", "#2E7D32")
        )
        save_button.pack(side='bottom', pady=20)
    
    def build_2d_interface(self, parent):
        """Build simplified interface for 2D analysis"""
        # Basic Settings Section
        basic_frame = self.create_section_frame(parent, "Basic Settings")
        
        # Frame Rate
        frame_rate_frame = ctk.CTkFrame(basic_frame, fg_color="transparent")
        frame_rate_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(frame_rate_frame, text=self.app.lang_manager.get_text('frame_rate'), width=200).pack(side='left')
        ctk.CTkEntry(frame_rate_frame, textvariable=self.frame_rate_var, width=150).pack(side='left', padx=5)
        
        # Slow Motion Factor
        slowmo_frame = ctk.CTkFrame(basic_frame, fg_color="transparent")
        slowmo_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(slowmo_frame, text="Slow Motion Factor:", width=200).pack(side='left')
        ctk.CTkEntry(slowmo_frame, textvariable=self.slowmo_factor_var, width=150).pack(side='left', padx=5)
        
        # Pose Processing Section
        pose_frame = self.create_section_frame(parent, "Pose Processing")
        
        # Keypoint Likelihood Threshold
        kp_thresh_frame = ctk.CTkFrame(pose_frame, fg_color="transparent")
        kp_thresh_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(kp_thresh_frame, text="Keypoint Likelihood Threshold:", width=200).pack(side='left')
        ctk.CTkEntry(kp_thresh_frame, textvariable=self.keypoint_likelihood_threshold_var, width=150).pack(side='left', padx=5)
        
        # Average Likelihood Threshold
        avg_thresh_frame = ctk.CTkFrame(pose_frame, fg_color="transparent")
        avg_thresh_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(avg_thresh_frame, text="Average Likelihood Threshold:", width=200).pack(side='left')
        ctk.CTkEntry(avg_thresh_frame, textvariable=self.average_likelihood_threshold_var, width=150).pack(side='left', padx=5)
        
        # Keypoint Number Threshold
        num_thresh_frame = ctk.CTkFrame(pose_frame, fg_color="transparent")
        num_thresh_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(num_thresh_frame, text="Keypoint Number Threshold:", width=200).pack(side='left')
        ctk.CTkEntry(num_thresh_frame, textvariable=self.keypoint_number_threshold_var, width=150).pack(side='left', padx=5)
        
        # Post-Processing Section
        post_frame = self.create_section_frame(parent, "Post-Processing")
        
        # Interpolation
        interp_frame = ctk.CTkFrame(post_frame, fg_color="transparent")
        interp_frame.pack(fill='x', pady=5)
        ctk.CTkCheckBox(interp_frame, text="Interpolate", variable=self.interpolate_var).pack(side='left')
        
        # Gap Size
        gap_frame = ctk.CTkFrame(post_frame, fg_color="transparent")
        gap_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(gap_frame, text="Interpolate Gaps Smaller Than:", width=200).pack(side='left')
        ctk.CTkEntry(gap_frame, textvariable=self.interp_gap_smaller_than_var, width=150).pack(side='left', padx=5)
        
        # Large Gaps
        large_gap_frame = ctk.CTkFrame(post_frame, fg_color="transparent")
        large_gap_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(large_gap_frame, text="Fill Large Gaps With:", width=200).pack(side='left')
        ctk.CTkOptionMenu(large_gap_frame, variable=self.fill_large_gaps_with_2d_var, 
                         values=['last_value', 'nan', 'zeros'], width=150).pack(side='left', padx=5)
        
        # Filtering
        filter_frame = ctk.CTkFrame(post_frame, fg_color="transparent")
        filter_frame.pack(fill='x', pady=5)
        ctk.CTkCheckBox(filter_frame, text="Filter", variable=self.filter_var).pack(side='left')
        ctk.CTkCheckBox(filter_frame, text="Show Graphs", variable=self.show_graphs_var).pack(side='left', padx=20)
        
        # Filter Type
        filter_type_frame = ctk.CTkFrame(post_frame, fg_color="transparent")
        filter_type_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(filter_type_frame, text="Filter Type:", width=200).pack(side='left')
        ctk.CTkOptionMenu(filter_type_frame, variable=self.filter_type_2d_var, 
                         values=['butterworth', 'gaussian', 'loess', 'median'], 
                         command=self.on_filter_type_change_2d, width=150).pack(side='left', padx=5)
        
        # Filter Parameters Frame
        self.filter_params_2d_frame = ctk.CTkFrame(post_frame)
        self.filter_params_2d_frame.pack(fill='x', pady=10)
        
        # Initialize with current filter type
        self.on_filter_type_change_2d(self.filter_type_2d_var.get())
        
        # Kinematics Section
        kin_frame = self.create_section_frame(parent, "Kinematics")
        
        ctk.CTkCheckBox(kin_frame, text="Use Augmentation", variable=self.use_augmentation_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(kin_frame, text="Use Contacts & Muscles", variable=self.use_contacts_muscles_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(kin_frame, text="Right-Left Symmetry", variable=self.right_left_symmetry_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(kin_frame, text="Remove Individual Scaling Setup", variable=self.remove_individual_scaling_setup_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(kin_frame, text="Remove Individual IK Setup", variable=self.remove_individual_IK_setup_var).pack(pady=5, anchor='w')
    
    def build_3d_interface(self, parent):
        """Build full interface for 3D analysis"""
        # Basic Settings Section
        basic_frame = self.create_section_frame(parent, "Basic Settings")
        
        # Frame Rate
        frame_rate_frame = ctk.CTkFrame(basic_frame, fg_color="transparent")
        frame_rate_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(frame_rate_frame, text=self.app.lang_manager.get_text('frame_rate'), width=200).pack(side='left')
        ctk.CTkEntry(frame_rate_frame, textvariable=self.frame_rate_var, width=150).pack(side='left', padx=5)
        
        # Frame Range
        frame_range_frame = ctk.CTkFrame(basic_frame, fg_color="transparent")
        frame_range_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(frame_range_frame, text=self.app.lang_manager.get_text('frame_range'), width=200).pack(side='left')
        ctk.CTkEntry(frame_range_frame, textvariable=self.frame_range_var, width=150).pack(side='left', padx=5)
        
        # Person Association Section
        pa_frame = self.create_section_frame(parent, self.app.lang_manager.get_text('person_association'))
        
        # Single Person subsection
        ctk.CTkLabel(pa_frame, text="Single Person Settings:", font=("Helvetica", 14, "bold")).pack(anchor='w', pady=(10, 5))
        
        likelihood_frame = ctk.CTkFrame(pa_frame, fg_color="transparent")
        likelihood_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(likelihood_frame, text="Likelihood Threshold:", width=200).pack(side='left')
        ctk.CTkEntry(likelihood_frame, textvariable=self.likelihood_threshold_association_var, width=150).pack(side='left', padx=5)
        
        reproj_frame = ctk.CTkFrame(pa_frame, fg_color="transparent")
        reproj_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(reproj_frame, text="Reprojection Error Threshold:", width=200).pack(side='left')
        ctk.CTkEntry(reproj_frame, textvariable=self.reproj_error_threshold_association_var, width=150).pack(side='left', padx=5)
        
        tracked_frame = ctk.CTkFrame(pa_frame, fg_color="transparent")
        tracked_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(tracked_frame, text="Tracked Keypoint:", width=200).pack(side='left')
        ctk.CTkEntry(tracked_frame, textvariable=self.tracked_keypoint_var, width=150).pack(side='left', padx=5)
        
        # Multi Person subsection
        ctk.CTkLabel(pa_frame, text="Multi Person Settings:", font=("Helvetica", 14, "bold")).pack(anchor='w', pady=(10, 5))
        
        recon_error_frame = ctk.CTkFrame(pa_frame, fg_color="transparent")
        recon_error_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(recon_error_frame, text="Reconstruction Error Threshold:", width=200).pack(side='left')
        ctk.CTkEntry(recon_error_frame, textvariable=self.reconstruction_error_threshold_var, width=150).pack(side='left', padx=5)
        
        min_affinity_frame = ctk.CTkFrame(pa_frame, fg_color="transparent")
        min_affinity_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(min_affinity_frame, text="Minimum Affinity:", width=200).pack(side='left')
        ctk.CTkEntry(min_affinity_frame, textvariable=self.min_affinity_var, width=150).pack(side='left', padx=5)
        
        # Triangulation Section
        tri_frame = self.create_section_frame(parent, self.app.lang_manager.get_text('triangulation'))
        
        tri_reproj_frame = ctk.CTkFrame(tri_frame, fg_color="transparent")
        tri_reproj_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(tri_reproj_frame, text="Reprojection Error Threshold:", width=200).pack(side='left')
        ctk.CTkEntry(tri_reproj_frame, textvariable=self.reproj_error_threshold_triangulation_var, width=150).pack(side='left', padx=5)
        
        tri_likelihood_frame = ctk.CTkFrame(tri_frame, fg_color="transparent")
        tri_likelihood_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(tri_likelihood_frame, text="Likelihood Threshold:", width=200).pack(side='left')
        ctk.CTkEntry(tri_likelihood_frame, textvariable=self.likelihood_threshold_triangulation_var, width=150).pack(side='left', padx=5)
        
        min_cameras_frame = ctk.CTkFrame(tri_frame, fg_color="transparent")
        min_cameras_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(min_cameras_frame, text="Minimum Cameras:", width=200).pack(side='left')
        ctk.CTkEntry(min_cameras_frame, textvariable=self.min_cameras_for_triangulation_var, width=150).pack(side='left', padx=5)
        
        interp_gap_frame = ctk.CTkFrame(tri_frame, fg_color="transparent")
        interp_gap_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(interp_gap_frame, text="Interpolate if Gap Smaller Than:", width=200).pack(side='left')
        ctk.CTkEntry(interp_gap_frame, textvariable=self.interp_if_gap_smaller_than_var, width=150).pack(side='left', padx=5)
        
        interp_type_frame = ctk.CTkFrame(tri_frame, fg_color="transparent")
        interp_type_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(interp_type_frame, text="Interpolation Type:", width=200).pack(side='left')
        ctk.CTkOptionMenu(interp_type_frame, variable=self.interpolation_type_var,
                         values=['linear', 'slinear', 'quadratic', 'cubic', 'none'], width=150).pack(side='left', padx=5)
        
        ctk.CTkCheckBox(tri_frame, text="Remove Incomplete Frames", variable=self.remove_incomplete_frames_var).pack(pady=5, anchor='w')
        
        sections_frame = ctk.CTkFrame(tri_frame, fg_color="transparent")
        sections_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(sections_frame, text="Sections to Keep:", width=200).pack(side='left')
        ctk.CTkOptionMenu(sections_frame, variable=self.sections_to_keep_var,
                         values=['all', 'largest', 'first', 'last'], width=150).pack(side='left', padx=5)
        
        fill_gaps_frame = ctk.CTkFrame(tri_frame, fg_color="transparent")
        fill_gaps_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(fill_gaps_frame, text="Fill Large Gaps With:", width=200).pack(side='left')
        ctk.CTkOptionMenu(fill_gaps_frame, variable=self.fill_large_gaps_with_var,
                         values=['last_value', 'nan', 'zeros'], width=150).pack(side='left', padx=5)
        
        ctk.CTkCheckBox(tri_frame, text="Show Interpolation Indices", variable=self.show_interp_indices_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(tri_frame, text="Make C3D", variable=self.triangulation_make_c3d_var).pack(pady=5, anchor='w')
        
        # Filtering Section
        filter_frame = self.create_section_frame(parent, self.app.lang_manager.get_text('filtering'))
        
        ctk.CTkCheckBox(filter_frame, text="Reject Outliers (Hampel Filter)", variable=self.reject_outliers_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(filter_frame, text="Apply Filter", variable=self.filter_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(filter_frame, text="Display Figures", variable=self.display_figures_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(filter_frame, text="Save Filtering Plots", variable=self.save_filt_plots_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(filter_frame, text="Make C3D", variable=self.filtering_make_c3d_var).pack(pady=5, anchor='w')
        
        filter_type_frame = ctk.CTkFrame(filter_frame, fg_color="transparent")
        filter_type_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(filter_type_frame, text="Filter Type:", width=200).pack(side='left')
        filter_options = ['butterworth', 'kalman', 'gcv_spline', 'gaussian', 'loess', 'median', 'butterworth_on_speed']
        ctk.CTkOptionMenu(filter_type_frame, variable=self.filter_type_var, 
                         values=filter_options, 
                         command=self.on_filter_type_change, width=150).pack(side='left', padx=5)
        
        # Filter Parameters Frame
        self.filter_params_frame = ctk.CTkFrame(filter_frame)
        self.filter_params_frame.pack(fill='x', pady=10)
        
        # Initialize with current filter type
        self.on_filter_type_change(self.filter_type_var.get())
        
        # Marker Augmentation Section
        marker_frame = self.create_section_frame(parent, self.app.lang_manager.get_text('marker_augmentation'))
        
        ctk.CTkCheckBox(marker_frame, text="Feet on Floor", variable=self.feet_on_floor_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(marker_frame, text="Make C3D", variable=self.augmentation_make_c3d_var).pack(pady=5, anchor='w')
        
        # Kinematics Section
        kin_frame = self.create_section_frame(parent, self.app.lang_manager.get_text('kinematics'))
        
        ctk.CTkCheckBox(kin_frame, text="Use Augmentation", variable=self.use_augmentation_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(kin_frame, text="Use Simple Model (>10x faster)", variable=self.use_simple_model_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(kin_frame, text="Use Contacts & Muscles", variable=self.use_contacts_muscles_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(kin_frame, text="Right-Left Symmetry", variable=self.right_left_symmetry_var).pack(pady=5, anchor='w')
        
        default_height_frame = ctk.CTkFrame(kin_frame, fg_color="transparent")
        default_height_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(default_height_frame, text="Default Height (m):", width=200).pack(side='left')
        ctk.CTkEntry(default_height_frame, textvariable=self.default_height_var, width=150).pack(side='left', padx=5)
        
        ctk.CTkCheckBox(kin_frame, text="Remove Individual Scaling Setup", variable=self.remove_individual_scaling_setup_var).pack(pady=5, anchor='w')
        ctk.CTkCheckBox(kin_frame, text="Remove Individual IK Setup", variable=self.remove_individual_IK_setup_var).pack(pady=5, anchor='w')
        
        fastest_frames_frame = ctk.CTkFrame(kin_frame, fg_color="transparent")
        fastest_frames_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(fastest_frames_frame, text="Fastest Frames to Remove (%):", width=200).pack(side='left')
        ctk.CTkEntry(fastest_frames_frame, textvariable=self.fastest_frames_to_remove_percent_var, width=150).pack(side='left', padx=5)
        
        close_to_zero_frame = ctk.CTkFrame(kin_frame, fg_color="transparent")
        close_to_zero_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(close_to_zero_frame, text="Close to Zero Speed (m):", width=200).pack(side='left')
        ctk.CTkEntry(close_to_zero_frame, textvariable=self.close_to_zero_speed_m_var, width=150).pack(side='left', padx=5)
        
        large_angles_frame = ctk.CTkFrame(kin_frame, fg_color="transparent")
        large_angles_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(large_angles_frame, text="Large Hip/Knee Angles (deg):", width=200).pack(side='left')
        ctk.CTkEntry(large_angles_frame, textvariable=self.large_hip_knee_angles_var, width=150).pack(side='left', padx=5)
        
        trimmed_extrema_frame = ctk.CTkFrame(kin_frame, fg_color="transparent")
        trimmed_extrema_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(trimmed_extrema_frame, text="Trimmed Extrema Percent:", width=200).pack(side='left')
        ctk.CTkEntry(trimmed_extrema_frame, textvariable=self.trimmed_extrema_percent_var, width=150).pack(side='left', padx=5)
    
    def create_section_frame(self, parent, title):
        """Create a section frame with a title"""
        section_frame = ctk.CTkFrame(parent)
        section_frame.pack(fill='x', pady=10, padx=5)
        
        # Title
        title_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        title_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(title_frame, text=title, font=('Helvetica', 16, 'bold')).pack(anchor='w', padx=10)
        
        # Content frame
        content_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        content_frame.pack(fill='x', pady=5, padx=20)
        
        return content_frame
    
    def on_filter_type_change(self, selected_filter):
        """Update filter parameters when filter type changes"""
        # Clear existing widgets
        for widget in self.filter_params_frame.winfo_children():
            widget.destroy()
        
        if selected_filter == 'butterworth':
            cutoff_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            cutoff_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(cutoff_frame, text="Cutoff Frequency (Hz):", width=200).pack(side='left')
            ctk.CTkEntry(cutoff_frame, textvariable=self.filter_cutoff_var, width=150).pack(side='left', padx=5)
            
            order_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            order_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(order_frame, text="Filter Order:", width=200).pack(side='left')
            ctk.CTkEntry(order_frame, textvariable=self.filter_order_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'kalman':
            trust_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            trust_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(trust_frame, text="Trust Ratio:", width=200).pack(side='left')
            ctk.CTkEntry(trust_frame, textvariable=self.kalman_trust_ratio_var, width=150).pack(side='left', padx=5)
            
            smooth_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            smooth_frame.pack(fill='x', pady=5)
            ctk.CTkCheckBox(smooth_frame, text="Smooth", variable=self.kalman_smooth_var).pack(side='left')
            
        elif selected_filter == 'gcv_spline':
            cutoff_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            cutoff_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(cutoff_frame, text="Cutoff Frequency ('auto' or Hz):", width=200).pack(side='left')
            ctk.CTkEntry(cutoff_frame, textvariable=self.gcv_cut_off_frequency_var, width=150).pack(side='left', padx=5)
            
            smoothing_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            smoothing_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(smoothing_frame, text="Smoothing Factor:", width=200).pack(side='left')
            ctk.CTkEntry(smoothing_frame, textvariable=self.gcv_smoothing_factor_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'butterworth_on_speed':
            cutoff_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            cutoff_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(cutoff_frame, text="Cutoff Frequency (Hz):", width=200).pack(side='left')
            ctk.CTkEntry(cutoff_frame, textvariable=self.butterworth_on_speed_cut_off_frequency_var, width=150).pack(side='left', padx=5)
            
            order_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            order_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(order_frame, text="Filter Order:", width=200).pack(side='left')
            ctk.CTkEntry(order_frame, textvariable=self.butterworth_on_speed_order_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'gaussian':
            sigma_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            sigma_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(sigma_frame, text="Sigma Kernel (px):", width=200).pack(side='left')
            ctk.CTkEntry(sigma_frame, textvariable=self.gaussian_sigma_kernel_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'loess':
            values_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            values_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(values_frame, text="Number of Values Used:", width=200).pack(side='left')
            ctk.CTkEntry(values_frame, textvariable=self.LOESS_nb_values_used_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'median':
            kernel_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            kernel_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(kernel_frame, text="Kernel Size:", width=200).pack(side='left')
            ctk.CTkEntry(kernel_frame, textvariable=self.median_kernel_size_var, width=150).pack(side='left', padx=5)
    
    def on_filter_type_change_2d(self, selected_filter):
        """Update filter parameters when filter type changes in 2D mode"""
        # Clear existing widgets
        for widget in self.filter_params_2d_frame.winfo_children():
            widget.destroy()
        
        if selected_filter == 'butterworth':
            cutoff_frame = ctk.CTkFrame(self.filter_params_2d_frame, fg_color="transparent")
            cutoff_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(cutoff_frame, text="Cutoff Frequency (Hz):", width=200).pack(side='left')
            ctk.CTkEntry(cutoff_frame, textvariable=self.filter_cutoff_var, width=150).pack(side='left', padx=5)
            
            order_frame = ctk.CTkFrame(self.filter_params_2d_frame, fg_color="transparent")
            order_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(order_frame, text="Filter Order:", width=200).pack(side='left')
            ctk.CTkEntry(order_frame, textvariable=self.filter_order_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'gaussian':
            sigma_frame = ctk.CTkFrame(self.filter_params_2d_frame, fg_color="transparent")
            sigma_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(sigma_frame, text="Sigma Kernel (px):", width=200).pack(side='left')
            ctk.CTkEntry(sigma_frame, textvariable=self.gaussian_sigma_kernel_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'loess':
            values_frame = ctk.CTkFrame(self.filter_params_2d_frame, fg_color="transparent")
            values_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(values_frame, text="Number of Values Used:", width=200).pack(side='left')
            ctk.CTkEntry(values_frame, textvariable=self.LOESS_nb_values_used_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'median':
            kernel_frame = ctk.CTkFrame(self.filter_params_2d_frame, fg_color="transparent")
            kernel_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(kernel_frame, text="Kernel Size:", width=200).pack(side='left')
            ctk.CTkEntry(kernel_frame, textvariable=self.median_kernel_size_var, width=150).pack(side='left', padx=5)
    
    def save_settings(self):
        """Save the advanced settings"""
        try:
            # Validate inputs
            self.validate_inputs()
            
            # Update the app with our settings
            if hasattr(self.app, 'update_tab_indicator'):
                self.app.update_tab_indicator('advanced', True)
            if hasattr(self.app, 'update_progress_bar'):
                progress_value = 85  # Based on progress_steps
                self.app.update_progress_bar(progress_value)
            
            # Show success message
            messagebox.showinfo(
                self.app.lang_manager.get_text('success'),
                "Advanced settings saved successfully"
            )
            
        except ValueError as e:
            messagebox.showerror(
                self.app.lang_manager.get_text('error'),
                str(e)
            )
    
    def validate_inputs(self):
        """Validate all input values"""
        errors = []
        
        # Frame rate
        frame_rate = self.frame_rate_var.get()
        if frame_rate != 'auto':
            try:
                float(frame_rate)
            except ValueError:
                errors.append("Frame Rate must be 'auto' or a number")
        
        # Validate numeric inputs
        if not self.simplified:
            try:
                float(self.likelihood_threshold_association_var.get())
            except ValueError:
                errors.append("Likelihood Threshold must be a number")
            
            try:
                float(self.reproj_error_threshold_association_var.get())
            except ValueError:
                errors.append("Reprojection Error Threshold must be a number")
            
            try:
                float(self.default_height_var.get())
            except ValueError:
                errors.append("Default Height must be a number")
        
        if errors:
            raise ValueError("\n".join(errors))
        
        return True