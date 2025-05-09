import os
import tkinter as tk
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
        self.frame_range_var = ctk.StringVar(value='[]')
        
        # Person Association Variables
        self.likelihood_threshold_association_var = ctk.StringVar(value='0.3')
        self.reproj_error_threshold_association_var = ctk.StringVar(value='20')
        self.tracked_keypoint_var = ctk.StringVar(value='Neck')
        
        # Triangulation Variables (for 3D)
        self.reproj_error_threshold_triangulation_var = ctk.StringVar(value='15')
        self.likelihood_threshold_triangulation_var = ctk.StringVar(value='0.3')
        self.min_cameras_for_triangulation_var = ctk.StringVar(value='2')
        
        # Filtering Variables
        self.filter_type_var = ctk.StringVar(value='butterworth')
        
        # Butterworth Variables
        self.filter_cutoff_var = ctk.StringVar(value='6')
        self.filter_order_var = ctk.StringVar(value='4')
        
        # Kalman Variables
        self.kalman_trust_ratio_var = ctk.StringVar(value='100')
        self.kalman_smooth_var = ctk.BooleanVar(value=True)
        
        # Butterworth on Speed Variables
        self.butterworth_on_speed_order_var = ctk.StringVar(value='4')
        self.butterworth_on_speed_cut_off_frequency_var = ctk.StringVar(value='10')
        
        # Gaussian Variables
        self.gaussian_sigma_kernel_var = ctk.StringVar(value='2')
        
        # LOESS Variables
        self.LOESS_nb_values_used_var = ctk.StringVar(value='30')
        
        # Median Variables
        self.median_kernel_size_var = ctk.StringVar(value='9')
        
        # Marker Augmentation Variables (for 3D)
        self.make_c3d_var = ctk.BooleanVar(value=True)
        
        # Kinematics Variables
        self.use_augmentation_var = ctk.BooleanVar(value=True)
        self.use_contacts_muscles_var = ctk.BooleanVar(value=True)
        self.right_left_symmetry_var = ctk.BooleanVar(value=True)
        self.remove_individual_scaling_setup_var = ctk.BooleanVar(value=True)
        self.remove_individual_IK_setup_var = ctk.BooleanVar(value=True)

        # 2D specific variables
        if self.simplified:
            # For 2D analysis
            self.slowmo_factor_var = ctk.StringVar(value='1')
            self.keypoint_likelihood_threshold_var = ctk.StringVar(value='0.3')
            self.average_likelihood_threshold_var = ctk.StringVar(value='0.5')
            self.keypoint_number_threshold_var = ctk.StringVar(value='0.3')
            self.interpolate_var = ctk.BooleanVar(value=True)
            self.interp_gap_smaller_than_var = ctk.StringVar(value='10')
            self.fill_large_gaps_with_var = ctk.StringVar(value='last_value')
            self.filter_var = ctk.BooleanVar(value=True)
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
                    'fill_large_gaps_with': self.fill_large_gaps_with_var.get(),
                    'filter': self.filter_var.get(),
                    'show_graphs': self.show_graphs_var.get(),
                    'filter_type': self.filter_type_2d_var.get()
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
                    'likelihood_threshold_association': float(self.likelihood_threshold_association_var.get()),
                    'single_person': {
                        'reproj_error_threshold_association': float(self.reproj_error_threshold_association_var.get()),
                        'tracked_keypoint': self.tracked_keypoint_var.get()
                    }
                },
                'triangulation': {
                    'reproj_error_threshold_triangulation': float(self.reproj_error_threshold_triangulation_var.get()),
                    'likelihood_threshold_triangulation': float(self.likelihood_threshold_triangulation_var.get()),
                    'min_cameras_for_triangulation': int(self.min_cameras_for_triangulation_var.get())
                },
                'filtering': {
                    'type': self.filter_type_var.get(),
                    'make_c3d': self.make_c3d_var.get()
                },
                'markerAugmentation': {
                    'make_c3d': self.make_c3d_var.get()
                },
                'kinematics': {
                    'use_augmentation': self.use_augmentation_var.get(),
                    'use_contacts_muscles': self.use_contacts_muscles_var.get(),
                    'right_left_symmetry': self.right_left_symmetry_var.get(),
                    'remove_individual_scaling_setup': self.remove_individual_scaling_setup_var.get(),
                    'remove_individual_IK_setup': self.remove_individual_IK_setup_var.get()
                }
            }
            
            # Try to parse frame range if it's not empty
            try:
                frame_range = ast.literal_eval(self.frame_range_var.get())
                if isinstance(frame_range, list):
                    settings['project']['frame_range'] = frame_range
            except:
                settings['project']['frame_range'] = []
            
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
            elif filter_type == 'butterworth_on_speed':
                settings['filtering']['butterworth_on_speed'] = {
                    'order': int(self.butterworth_on_speed_order_var.get()),
                    'cut_off_frequency': float(self.butterworth_on_speed_cut_off_frequency_var.get())
                }
            elif filter_type == 'gaussian':
                settings['filtering']['gaussian'] = {
                    'sigma_kernel': float(self.gaussian_sigma_kernel_var.get())
                }
            elif filter_type == 'LOESS':
                settings['filtering']['LOESS'] = {
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
        header_frame = ctk.CTkFrame(self.frame)
        header_frame.pack(fill='x', padx=20, pady=(20, 0))
        
        ctk.CTkLabel(
            header_frame, 
            text=self.get_title(), 
            font=('Helvetica', 24, 'bold')
        ).pack(anchor='w')
        
        # Create scrollable content frame
        scroll_frame = ctk.CTkScrollableFrame(self.frame)
        scroll_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        if self.simplified:
            # Build simplified 2D interface
            self.build_2d_interface(scroll_frame)
        else:
            # Build full 3D interface
            self.build_3d_interface(scroll_frame)
        
        # Save Button
        save_button = ctk.CTkButton(
            self.frame,
            text=self.app.lang_manager.get_text('save_advanced_settings'),
            command=self.save_settings,
            height=40
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
        ctk.CTkOptionMenu(large_gap_frame, variable=self.fill_large_gaps_with_var, 
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
        
        # Use Augmentation
        augmentation_check = ctk.CTkCheckBox(
            kin_frame,
            text="Use Augmentation",
            variable=self.use_augmentation_var
        )
        augmentation_check.pack(pady=5, anchor='w')
        
        # Use Contacts & Muscles
        contacts_check = ctk.CTkCheckBox(
            kin_frame,
            text="Use Contacts & Muscles",
            variable=self.use_contacts_muscles_var
        )
        contacts_check.pack(pady=5, anchor='w')
        
        # Right-Left Symmetry
        symmetry_check = ctk.CTkCheckBox(
            kin_frame,
            text="Right-Left Symmetry",
            variable=self.right_left_symmetry_var
        )
        symmetry_check.pack(pady=5, anchor='w')
        
        # Remove Individual Scaling
        scaling_check = ctk.CTkCheckBox(
            kin_frame,
            text="Remove Individual Scaling Setup",
            variable=self.remove_individual_scaling_setup_var
        )
        scaling_check.pack(pady=5, anchor='w')
        
        # Remove Individual IK
        ik_check = ctk.CTkCheckBox(
            kin_frame,
            text="Remove Individual IK Setup",
            variable=self.remove_individual_IK_setup_var
        )
        ik_check.pack(pady=5, anchor='w')
    
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
        
        # Likelihood Threshold
        likelihood_frame = ctk.CTkFrame(pa_frame, fg_color="transparent")
        likelihood_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(likelihood_frame, text="Likelihood Threshold:", width=200).pack(side='left')
        ctk.CTkEntry(likelihood_frame, textvariable=self.likelihood_threshold_association_var, width=150).pack(side='left', padx=5)
        
        # Reprojection Error Threshold
        reproj_frame = ctk.CTkFrame(pa_frame, fg_color="transparent")
        reproj_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(reproj_frame, text="Reprojection Error Threshold:", width=200).pack(side='left')
        ctk.CTkEntry(reproj_frame, textvariable=self.reproj_error_threshold_association_var, width=150).pack(side='left', padx=5)
        
        # Tracked Keypoint
        tracked_frame = ctk.CTkFrame(pa_frame, fg_color="transparent")
        tracked_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(tracked_frame, text="Tracked Keypoint:", width=200).pack(side='left')
        ctk.CTkEntry(tracked_frame, textvariable=self.tracked_keypoint_var, width=150).pack(side='left', padx=5)
        
        # Triangulation Section
        tri_frame = self.create_section_frame(parent, self.app.lang_manager.get_text('triangulation'))
        
        # Reprojection Error Threshold
        tri_reproj_frame = ctk.CTkFrame(tri_frame, fg_color="transparent")
        tri_reproj_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(tri_reproj_frame, text="Reprojection Error Threshold:", width=200).pack(side='left')
        ctk.CTkEntry(tri_reproj_frame, textvariable=self.reproj_error_threshold_triangulation_var, width=150).pack(side='left', padx=5)
        
        # Likelihood Threshold
        tri_likelihood_frame = ctk.CTkFrame(tri_frame, fg_color="transparent")
        tri_likelihood_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(tri_likelihood_frame, text="Likelihood Threshold:", width=200).pack(side='left')
        ctk.CTkEntry(tri_likelihood_frame, textvariable=self.likelihood_threshold_triangulation_var, width=150).pack(side='left', padx=5)
        
        # Minimum Cameras
        min_cameras_frame = ctk.CTkFrame(tri_frame, fg_color="transparent")
        min_cameras_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(min_cameras_frame, text="Minimum Cameras:", width=200).pack(side='left')
        ctk.CTkEntry(min_cameras_frame, textvariable=self.min_cameras_for_triangulation_var, width=150).pack(side='left', padx=5)
        
        # Filtering Section
        filter_frame = self.create_section_frame(parent, self.app.lang_manager.get_text('filtering'))
        
        # Filter Type
        filter_type_frame = ctk.CTkFrame(filter_frame, fg_color="transparent")
        filter_type_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(filter_type_frame, text="Filter Type:", width=200).pack(side='left')
        filter_options = ['butterworth', 'kalman', 'gaussian', 'LOESS', 'median', 'butterworth_on_speed']
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
        
        # Make C3D
        make_c3d_check = ctk.CTkCheckBox(
            marker_frame,
            text="Make C3D",
            variable=self.make_c3d_var
        )
        make_c3d_check.pack(pady=5)
        
        # Kinematics Section
        kin_frame = self.create_section_frame(parent, self.app.lang_manager.get_text('kinematics'))
        
        # Use Augmentation
        augmentation_check = ctk.CTkCheckBox(
            kin_frame,
            text="Use Augmentation",
            variable=self.use_augmentation_var
        )
        augmentation_check.pack(pady=5, anchor='w')
        
        # Use Contacts & Muscles
        contacts_check = ctk.CTkCheckBox(
            kin_frame,
            text="Use Contacts & Muscles",
            variable=self.use_contacts_muscles_var
        )
        contacts_check.pack(pady=5, anchor='w')
        
        # Right-Left Symmetry
        symmetry_check = ctk.CTkCheckBox(
            kin_frame,
            text="Right-Left Symmetry",
            variable=self.right_left_symmetry_var
        )
        symmetry_check.pack(pady=5, anchor='w')
        
        # Remove Individual Scaling
        scaling_check = ctk.CTkCheckBox(
            kin_frame,
            text="Remove Individual Scaling Setup",
            variable=self.remove_individual_scaling_setup_var
        )
        scaling_check.pack(pady=5, anchor='w')
        
        # Remove Individual IK
        ik_check = ctk.CTkCheckBox(
            kin_frame,
            text="Remove Individual IK Setup",
            variable=self.remove_individual_IK_setup_var
        )
        ik_check.pack(pady=5, anchor='w')
    
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
            # Butterworth parameters
            cutoff_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            cutoff_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(cutoff_frame, text="Cutoff Frequency (Hz):", width=200).pack(side='left')
            ctk.CTkEntry(cutoff_frame, textvariable=self.filter_cutoff_var, width=150).pack(side='left', padx=5)
            
            order_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            order_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(order_frame, text="Filter Order:", width=200).pack(side='left')
            ctk.CTkEntry(order_frame, textvariable=self.filter_order_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'kalman':
            # Kalman parameters
            trust_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            trust_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(trust_frame, text="Trust Ratio:", width=200).pack(side='left')
            ctk.CTkEntry(trust_frame, textvariable=self.kalman_trust_ratio_var, width=150).pack(side='left', padx=5)
            
            smooth_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            smooth_frame.pack(fill='x', pady=5)
            ctk.CTkCheckBox(smooth_frame, text="Smooth", variable=self.kalman_smooth_var).pack(side='left')
            
        elif selected_filter == 'butterworth_on_speed':
            # Butterworth on Speed parameters
            cutoff_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            cutoff_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(cutoff_frame, text="Cutoff Frequency (Hz):", width=200).pack(side='left')
            ctk.CTkEntry(cutoff_frame, textvariable=self.butterworth_on_speed_cut_off_frequency_var, width=150).pack(side='left', padx=5)
            
            order_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            order_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(order_frame, text="Filter Order:", width=200).pack(side='left')
            ctk.CTkEntry(order_frame, textvariable=self.butterworth_on_speed_order_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'gaussian':
            # Gaussian parameters
            sigma_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            sigma_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(sigma_frame, text="Sigma Kernel (px):", width=200).pack(side='left')
            ctk.CTkEntry(sigma_frame, textvariable=self.gaussian_sigma_kernel_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'LOESS':
            # LOESS parameters
            values_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
            values_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(values_frame, text="Number of Values Used:", width=200).pack(side='left')
            ctk.CTkEntry(values_frame, textvariable=self.LOESS_nb_values_used_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'median':
            # Median parameters
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
            # Butterworth parameters
            cutoff_frame = ctk.CTkFrame(self.filter_params_2d_frame, fg_color="transparent")
            cutoff_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(cutoff_frame, text="Cutoff Frequency (Hz):", width=200).pack(side='left')
            ctk.CTkEntry(cutoff_frame, textvariable=self.filter_cutoff_var, width=150).pack(side='left', padx=5)
            
            order_frame = ctk.CTkFrame(self.filter_params_2d_frame, fg_color="transparent")
            order_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(order_frame, text="Filter Order:", width=200).pack(side='left')
            ctk.CTkEntry(order_frame, textvariable=self.filter_order_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'gaussian':
            # Gaussian parameters
            sigma_frame = ctk.CTkFrame(self.filter_params_2d_frame, fg_color="transparent")
            sigma_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(sigma_frame, text="Sigma Kernel (px):", width=200).pack(side='left')
            ctk.CTkEntry(sigma_frame, textvariable=self.gaussian_sigma_kernel_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'loess':
            # LOESS parameters
            values_frame = ctk.CTkFrame(self.filter_params_2d_frame, fg_color="transparent")
            values_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(values_frame, text="Number of Values Used:", width=200).pack(side='left')
            ctk.CTkEntry(values_frame, textvariable=self.LOESS_nb_values_used_var, width=150).pack(side='left', padx=5)
            
        elif selected_filter == 'median':
            # Median parameters
            kernel_frame = ctk.CTkFrame(self.filter_params_2d_frame, fg_color="transparent")
            kernel_frame.pack(fill='x', pady=5)
            ctk.CTkLabel(kernel_frame, text="Kernel Size:", width=200).pack(side='left')
            ctk.CTkEntry(kernel_frame, textvariable=self.median_kernel_size_var, width=150).pack(side='left', padx=5)
    
    def save_settings(self):
        """Save the advanced settings"""
        try:
            # Validate inputs
            self.validate_inputs()
            
            # Update the app with our settings - FIXED METHOD NAME
            if hasattr(self.app, 'update_tab_indicator'):
                self.app.update_tab_indicator('advanced', True)
            if hasattr(self.app, 'update_progress_bar'):
                progress_value = 85  # This is based on the progress_steps in Pose2SimApp
                self.app.update_progress_bar(progress_value)
            
            # Show success message
            messagebox.showinfo(
                self.app.lang_manager.get_text('success'),
                self.app.lang_manager.get_text('advanced settings saved')
            )
            
        except ValueError as e:
            messagebox.showerror(
                self.app.lang_manager.get_text('error'),
                str(e)
            )
    
    def validate_inputs(self):
        """Validate all input values"""
        errors = []
        
        # Frame rate (can be 'auto' or a number)
        frame_rate = self.frame_rate_var.get()
        if frame_rate != 'auto':
            try:
                float(frame_rate)
            except ValueError:
                errors.append("Frame Rate must be 'auto' or a number")
        
        # Frame range (must be a valid list or empty)
        frame_range = self.frame_range_var.get()
        if frame_range and frame_range != '[]':
            try:
                range_list = ast.literal_eval(frame_range)
                if not isinstance(range_list, list):
                    errors.append("Frame Range must be a list like [10, 300]")
            except (ValueError, SyntaxError):
                errors.append("Frame Range must be a valid list format like [10, 300]")
        
        # Person Association thresholds (must be numbers)
        if not self.simplified:
            try:
                float(self.likelihood_threshold_association_var.get())
            except ValueError:
                errors.append("Likelihood Threshold must be a number")
                
            try:
                float(self.reproj_error_threshold_association_var.get())
            except ValueError:
                errors.append("Reprojection Error Threshold must be a number")
            
            # Triangulation thresholds (must be numbers)
            try:
                float(self.reproj_error_threshold_triangulation_var.get())
            except ValueError:
                errors.append("Triangulation Reprojection Error Threshold must be a number")
                
            try:
                float(self.likelihood_threshold_triangulation_var.get())
            except ValueError:
                errors.append("Triangulation Likelihood Threshold must be a number")
                
            try:
                int(self.min_cameras_for_triangulation_var.get())
            except ValueError:
                errors.append("Minimum Cameras must be an integer")
        
        # If there are any errors, raise with all error messages
        if errors:
            raise ValueError("\n".join(errors))
        
        return True