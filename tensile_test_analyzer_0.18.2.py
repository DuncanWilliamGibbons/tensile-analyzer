import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

warnings.filterwarnings('ignore')

class TensileTestAnalyzer:
    """
    Comprehensive tensile test analyzer conforming to ASTM E8/E8M standards
    for determining mechanical properties from load-extension data.
    """
    
    def __init__(self, load_data, extension_data, specimen_params, strain_gauge_data=None):
        """
        Initialize the analyzer with test data and specimen parameters.
        
        Parameters:
        -----------
        load_data : array-like
            Load values (typically in N or lbf)
        extension_data : array-like
            Extension values (typically in mm or in)
        specimen_params : dict
            Dictionary containing specimen geometry and properties
        strain_gauge_data : array-like, optional
            Strain gauge data (if available)
        """
        self.load = np.array(load_data)
        self.extension = np.array(extension_data)
        self.specimen_params = specimen_params
        self.strain_gauge = np.array(strain_gauge_data) if strain_gauge_data is not None else None
        
        # Validate input data
        if len(self.load) == 0 or len(self.extension) == 0:
            raise ValueError("Load and extension data arrays must not be empty")
        
        if len(self.load) != len(self.extension):
            raise ValueError("Load and extension data arrays must have the same length")
        
        # Validate specimen parameters
        required_params = ['original_length', 'specimen_type']
        for param in required_params:
            if param not in specimen_params:
                raise ValueError(f"Required specimen parameter '{param}' is missing")
        
        if specimen_params['specimen_type'] not in ['round', 'flat']:
            raise ValueError("specimen_type must be 'round' or 'flat'")
        
        # Calculate basic parameters
        self.L0 = specimen_params['original_length']  # Original gauge length
        
        if specimen_params['specimen_type'] == 'round':
            if 'original_diameter' not in specimen_params:
                raise ValueError("original_diameter is required for round specimens")
            diameter = specimen_params['original_diameter']
            if diameter <= 0:
                raise ValueError("Diameter must be positive")
            self.A0 = np.pi * (diameter / 2) ** 2
        else:  # flat specimen
            if 'original_width' not in specimen_params or 'original_thickness' not in specimen_params:
                raise ValueError("original_width and original_thickness are required for flat specimens")
            width = specimen_params['original_width']
            thickness = specimen_params['original_thickness']
            if width <= 0 or thickness <= 0:
                raise ValueError("Width and thickness must be positive")
            self.A0 = width * thickness
        
        # Calculate engineering stress and strain with safety checks
        if self.A0 <= 0:
            raise ValueError("Cross-sectional area must be positive")
        if self.L0 <= 0:
            raise ValueError("Gauge length must be positive")
            
        self.stress = self.load / self.A0  # Engineering stress (MPa if load in N and area in mm²)
        self.strain = self.extension / self.L0  # Engineering strain (dimensionless)
        
        # Validate computed stress and strain values
        if np.any(np.isnan(self.stress)) or np.any(np.isnan(self.strain)):
            raise ValueError("Computed stress or strain contains NaN values")
        
        if np.any(np.isinf(self.stress)) or np.any(np.isinf(self.strain)):
            raise ValueError("Computed stress or strain contains infinite values")
        
        # Initialize results dictionary
        self.results = {}
        
    def smooth_data(self, window_length=11, polyorder=2):
        """
        Smooth the stress-strain data using Savitzky-Golay filter
        """
        if len(self.stress) > window_length:
            self.stress = savgol_filter(self.stress, window_length, polyorder)
            self.strain = savgol_filter(self.strain, window_length, polyorder)
    
    def find_proportional_limit(self):
        """
        Find the proportional limit using R-squared analysis to determine end of linear elastic region
        This is critical for proper modulus calculation as we only want the linear portion
        Enhanced version with better detection logic
        """
        try:
            r_squared_threshold = 0.99  # More reasonable threshold 
            min_points = 20  # Minimum points for reliable regression
            
            best_r_squared = 0
            best_end_idx = min_points
            best_slope = 0
            
            # Calculate expected elastic modulus range (typical values)
            expected_E_min = 50000   # 50 GPa minimum reasonable
            expected_E_max = 500000  # 500 GPa maximum reasonable
            
            # Search for the longest linear region with high R² and reasonable slope
            max_search_idx = min(len(self.stress) * 2 // 3, len(self.stress) - 5)  # Search up to 2/3 of data
            
            # Start with minimum points and expand until linearity degrades
            current_r_squared = 0
            degradation_count = 0
            last_good_idx = min_points
            
            for end_idx in range(min_points, max_search_idx, 2):  # Step by 2 for efficiency
                stress_segment = self.stress[:end_idx]
                strain_segment = self.strain[:end_idx]
                
                # Skip if we don't have enough variation in data
                if np.std(stress_segment) < 1.0 or np.std(strain_segment) < 1e-6:
                    continue
                
                # Skip if strain range is too small
                if (strain_segment[-1] - strain_segment[0]) < 1e-5:
                    continue
                
                slope, intercept, r_value, _, _ = stats.linregress(strain_segment, stress_segment)
                r_squared = r_value ** 2
                
                # Check if slope is reasonable for elastic modulus
                slope_reasonable = expected_E_min <= slope <= expected_E_max
                
                # Update if we find better linearity with reasonable slope
                if r_squared > best_r_squared and slope > 0 and slope_reasonable:
                    best_r_squared = r_squared
                    best_end_idx = end_idx
                    best_slope = slope
                
                # Track degradation in linearity
                if r_squared < current_r_squared - 0.01:  # Significant drop
                    degradation_count += 1
                    if degradation_count >= 3:  # Multiple consecutive drops
                        break
                else:
                    degradation_count = 0
                    if r_squared >= r_squared_threshold and slope_reasonable:
                        last_good_idx = end_idx
                
                current_r_squared = r_squared
                
                # Early termination if linearity is very poor
                if r_squared < 0.95 and end_idx > min_points * 3:
                    break
            
            # Use the last good index if we found reasonable linearity
            if best_r_squared >= 0.98 and best_slope > 0:
                final_idx = best_end_idx
            elif last_good_idx > min_points:
                final_idx = last_good_idx
                # Recalculate stats for this index
                stress_segment = self.stress[:final_idx]
                strain_segment = self.strain[:final_idx]
                slope, intercept, r_value, _, _ = stats.linregress(strain_segment, stress_segment)
                best_r_squared = r_value ** 2
                best_slope = slope
            else:
                # Conservative fallback: use first portion that shows consistent increase
                final_idx = min(len(self.stress) // 4, 100)
                stress_segment = self.stress[:final_idx]
                strain_segment = self.strain[:final_idx]
                if len(stress_segment) > 10:
                    slope, intercept, r_value, _, _ = stats.linregress(strain_segment, stress_segment)
                    best_r_squared = r_value ** 2
                    best_slope = slope
                print(f"Warning: Using conservative fallback elastic region ({final_idx} points)")
            
            self.results['proportional_limit'] = self.stress[final_idx-1]
            self.results['proportional_limit_strain'] = self.strain[final_idx-1]
            self.results['proportional_limit_index'] = final_idx-1
            self.results['proportional_limit_r_squared'] = best_r_squared
            self.results['elastic_modulus_estimate'] = best_slope
            
            print(f"Linear elastic region: 0 to {final_idx} points (R² = {best_r_squared:.4f})")
            print(f"Proportional limit: {self.results['proportional_limit']:.1f} MPa at {self.results['proportional_limit_strain']*100:.3f}% strain")
            print(f"Estimated elastic modulus: {best_slope/1000:.1f} GPa")
            
        except Exception as e:
            print(f"Error finding proportional limit: {e}")
            # Conservative fallback
            fallback_idx = min(len(self.stress) // 6, 80)
            self.results['proportional_limit'] = self.stress[fallback_idx]
            self.results['proportional_limit_strain'] = self.strain[fallback_idx]
            self.results['proportional_limit_index'] = fallback_idx
    
    def calculate_elastic_modulus_astm_e111(self, min_strain=0.0005, max_strain=0.0025, filter_points=3):
        """
        Calculate elastic modulus according to ASTM E111 using strain gauge data
        """
        try:
            if self.strain_gauge is None:
                raise ValueError("No strain gauge data available")
            
            # Filter data within the specified strain range
            mask = (self.strain_gauge >= min_strain) & (self.strain_gauge <= max_strain)
            
            if np.sum(mask) < filter_points:
                raise ValueError("Insufficient data points in strain range")
            
            stress_filtered = self.stress[mask]
            strain_filtered = self.strain_gauge[mask]
            
            # Linear regression to find elastic modulus
            slope, intercept, r_value, p_value, std_err = stats.linregress(strain_filtered, stress_filtered)
            
            self.results['elastic_modulus'] = slope  # MPa
            self.results['elastic_modulus_gpa'] = slope / 1000  # GPa
            self.results['elastic_modulus_r_squared'] = r_value ** 2
            self.results['elastic_modulus_used_strain_gauge'] = True
            self.results['elastic_modulus_strain_range'] = (min_strain, max_strain)
            
        except Exception as e:
            print(f"Error calculating elastic modulus (ASTM E111): {e}")
            # Fallback to extension-based calculation
            self.calculate_elastic_modulus_best_fit()
    
    def calculate_elastic_modulus_best_fit(self, min_points=10, filter_points=3):
        """
        Calculate elastic modulus using best fit to initial linear region (extension data)
        """
        try:
            # Use first portion of data for linear region
            max_idx = min(len(self.stress) // 3, len(self.stress) - 1)
            
            best_r_squared = 0
            best_slope = 0
            best_range = (0, min_points)
            
            # Try different ranges to find best linear fit
            for end_idx in range(min_points, max_idx):
                stress_segment = self.stress[:end_idx]
                strain_segment = self.strain[:end_idx]
                
                if len(stress_segment) < filter_points:
                    continue
                
                slope, intercept, r_value, _, _ = stats.linregress(strain_segment, stress_segment)
                r_squared = r_value ** 2
                
                if r_squared > best_r_squared and slope > 0:
                    best_r_squared = r_squared
                    best_slope = slope
                    best_range = (0, end_idx)
            
            self.results['elastic_modulus'] = best_slope  # MPa
            self.results['elastic_modulus_gpa'] = best_slope / 1000  # GPa
            self.results['elastic_modulus_r_squared'] = best_r_squared
            self.results['elastic_modulus_used_strain_gauge'] = False
            self.results['elastic_modulus_strain_range'] = best_range
            
        except Exception as e:
            print(f"Error calculating elastic modulus (best fit): {e}")
            # Final fallback: use simple slope between 10% and 40% of data
            n_total = len(self.strain)
            start_idx = n_total // 10
            end_idx = min(n_total // 2, n_total - 1)
            
            if end_idx > start_idx:
                slope = (self.stress[end_idx] - self.stress[start_idx]) / (self.strain[end_idx] - self.strain[start_idx])
                self.results['elastic_modulus'] = slope
                self.results['elastic_modulus_gpa'] = slope / 1000
                self.results['elastic_modulus_strain_range'] = (start_idx, end_idx)
            else:
                self.results['elastic_modulus'] = 200000  # Default 200 GPa
                self.results['elastic_modulus_gpa'] = 200
                self.results['elastic_modulus_strain_range'] = (0, 10)
            
            self.results['elastic_modulus_used_strain_gauge'] = False
    
    def calculate_elastic_modulus_astm_e111_enhanced(self, middle_fraction=0.5):
        """
        Calculate elastic modulus according to ASTM E111 using strain gauge data
        Enhanced version that uses the middle 50% of datapoints from the elastic region
        
        Parameters:
        -----------
        middle_fraction : float
            Fraction of middle datapoints to use from elastic region (default 0.5 = 50%)
            This excludes 25% from beginning and 25% from end of elastic region
        """
        try:
            if self.strain_gauge is None:
                raise ValueError("No strain gauge data available")
            
            # Find the proportional limit if not already calculated
            if 'proportional_limit_index' not in self.results:
                self.find_proportional_limit()
            
            # Get the elastic region boundaries
            start_idx = 0  # Start of data
            end_idx = self.results.get('proportional_limit_index', len(self.stress) // 4)
            
            # Ensure we have a reasonable elastic region
            if end_idx <= start_idx + 10:
                print(f"Warning: Short elastic region detected (only {end_idx - start_idx} points), using first quarter of data")
                end_idx = max(len(self.stress) // 4, start_idx + 20)
                end_idx = min(end_idx, len(self.stress) - 1)
            
            # Calculate the middle portion indices to use exactly 50% of middle datapoints
            elastic_length = end_idx - start_idx
            exclude_fraction = (1.0 - middle_fraction) / 2.0  # 0.25 for 50% middle
            offset_start_points = int(elastic_length * exclude_fraction)
            offset_end_points = int(elastic_length * exclude_fraction)
            
            # Define the middle portion indices (middle 50% of elastic region)
            middle_start_idx = start_idx + offset_start_points
            middle_end_idx = end_idx - offset_end_points
            
            # Ensure we have sufficient data points
            if middle_end_idx <= middle_start_idx + 5:
                print(f"Warning: Insufficient middle region, using available data")
                middle_start_idx = start_idx + max(1, elastic_length // 4)
                middle_end_idx = end_idx - max(1, elastic_length // 4)
            
            # Extract the middle 50% of the elastic region
            stress_middle = self.stress[middle_start_idx:middle_end_idx+1]
            strain_middle = self.strain_gauge[middle_start_idx:middle_end_idx+1]
            
            print(f"Elastic region analysis (strain gauge data):")
            print(f"  Total elastic region: indices {start_idx} to {end_idx} ({elastic_length} points)")
            print(f"  Middle {middle_fraction*100:.0f}% used: indices {middle_start_idx} to {middle_end_idx} ({len(stress_middle)} points)")
            print(f"  Excluded {exclude_fraction*100:.0f}% from start and {exclude_fraction*100:.0f}% from end")
            print(f"  Stress range: {stress_middle[0]:.0f} to {stress_middle[-1]:.0f} MPa")
            print(f"  Strain range: {strain_middle[0]*100:.3f}% to {strain_middle[-1]*100:.3f}%")
            
            
            # 1. Young's Modulus (linear regression on middle 50% of elastic region)
            slope, intercept, r_value, p_value, std_err = stats.linregress(strain_middle, stress_middle)
            
            # 2. Chord Modulus (between 25% and 75% of the selected elastic region)
            stress_25 = stress_middle[0] + 0.25 * (stress_middle[-1] - stress_middle[0])
            stress_75 = stress_middle[0] + 0.75 * (stress_middle[-1] - stress_middle[0])
            
            # Find closest points to these stress values
            idx_25 = np.argmin(np.abs(stress_middle - stress_25))
            idx_75 = np.argmin(np.abs(stress_middle - stress_75))
            
            if idx_25 != idx_75 and np.abs(strain_middle[idx_75] - strain_middle[idx_25]) > 1e-8:
                chord_modulus = (stress_middle[idx_75] - stress_middle[idx_25]) / (strain_middle[idx_75] - strain_middle[idx_25])
            else:
                chord_modulus = slope
            
            # 3. Tangent Modulus (derivative at midpoint)
            # Use local linear fit around midpoint
            mid_idx = len(strain_middle) // 2
            window = min(5, len(strain_middle) // 4)  # Use small window around midpoint
            
            if mid_idx >= window and mid_idx + window < len(strain_middle):
                strain_tangent = strain_middle[mid_idx-window:mid_idx+window+1]
                stress_tangent = stress_middle[mid_idx-window:mid_idx+window+1]
                tangent_slope, _, tangent_r, _, _ = stats.linregress(strain_tangent, stress_tangent)
                tangent_modulus = tangent_slope
            else:
                tangent_modulus = slope
            
            # Store all modulus values
            self.results['youngs_modulus'] = slope  # MPa
            self.results['youngs_modulus_gpa'] = slope / 1000  # GPa
            self.results['chord_modulus'] = chord_modulus  # MPa
            self.results['chord_modulus_gpa'] = chord_modulus / 1000  # GPa
            self.results['tangent_modulus'] = tangent_modulus  # MPa
            self.results['tangent_modulus_gpa'] = tangent_modulus / 1000  # GPa
            
            # Legacy compatibility
            self.results['elastic_modulus'] = slope  # MPa
            self.results['elastic_modulus_gpa'] = slope / 1000  # GPa
            
            # Statistical information
            self.results['elastic_modulus_r_squared'] = r_value ** 2
            self.results['elastic_modulus_used_strain_gauge'] = True
            self.results['elastic_modulus_stress_range'] = (stress_middle[0], stress_middle[-1])
            self.results['elastic_modulus_strain_range'] = (np.min(strain_middle), np.max(strain_middle))
            self.results['modulus_calculation_method'] = f"ASTM E111 - Strain Gauge Data (Middle {middle_fraction*100:.0f}%: {middle_start_idx}-{middle_end_idx})"
            self.results['data_points_used'] = len(strain_middle)
            self.results['excluded_percentage'] = exclude_fraction
            self.results['elastic_region_indices'] = (start_idx, end_idx)
            
            # Chord modulus calculation points (using stress values)
            self.results['chord_stress_points'] = (stress_25, stress_75)
            self.results['chord_strain_points'] = (strain_middle[idx_25], strain_middle[idx_75])
            
            print(f"ASTM E111 Enhanced Analysis (Middle {middle_fraction*100:.0f}% of Elastic Region):")
            print(f"  Young's Modulus: {slope/1000:.1f} GPa (R² = {r_value**2:.4f})")
            print(f"  Chord Modulus: {chord_modulus/1000:.1f} GPa")
            print(f"  Tangent Modulus: {tangent_modulus/1000:.1f} GPa")
            print(f"  Stress range: {stress_middle[0]:.0f} to {stress_middle[-1]:.0f} MPa")
            print(f"  Data points used: {len(strain_middle)} (middle {middle_fraction*100:.0f}% of elastic region)")
            
        except Exception as e:
            print(f"Error calculating enhanced elastic modulus (ASTM E111): {e}")
            # Fallback to basic extension-based calculation
            self.calculate_elastic_modulus_best_fit_enhanced()
    
    def calculate_elastic_modulus_best_fit_enhanced(self, middle_fraction=0.5):
        """
        Enhanced elastic modulus calculation using the middle 50% of the LINEAR ELASTIC region
        Uses only the portion of data before plastic deformation begins for maximum accuracy
        
        Parameters:
        -----------
        middle_fraction : float
            Fraction of middle datapoints to use from LINEAR ELASTIC region (default 0.5 = 50%)
            This excludes 25% from beginning and 25% from end of the linear elastic region
        """
        try:
            # Find the proportional limit if not already calculated (end of linear elastic region)
            if 'proportional_limit_index' not in self.results:
                self.find_proportional_limit()
            
            # Get the LINEAR ELASTIC region boundaries (before plastic deformation)
            start_idx = 0  # Start of data
            end_idx = self.results.get('proportional_limit_index', len(self.stress) // 6)
            
            print(f"Linear elastic region identified: indices 0 to {end_idx}")
            
            # Ensure we have a reasonable elastic region
            if end_idx <= start_idx + 20:
                print(f"Warning: Short linear elastic region detected (only {end_idx - start_idx} points)")
                end_idx = max(len(self.stress) // 6, start_idx + 25)
                end_idx = min(end_idx, len(self.stress) - 1)
            
            # Calculate the middle portion indices to use exactly 50% of middle datapoints
            elastic_length = end_idx - start_idx
            exclude_fraction = (1.0 - middle_fraction) / 2.0  # 0.25 for 50% middle
            offset_start_points = int(elastic_length * exclude_fraction)
            offset_end_points = int(elastic_length * exclude_fraction)
            
            # Define the middle portion indices (middle 50% of LINEAR ELASTIC region)
            middle_start_idx = start_idx + offset_start_points
            middle_end_idx = end_idx - offset_end_points
            
            # Ensure we have sufficient data points for reliable analysis
            if middle_end_idx <= middle_start_idx + 10:
                print(f"Warning: Insufficient middle region, adjusting boundaries")
                # Use more conservative approach
                middle_start_idx = start_idx + max(3, elastic_length // 6)
                middle_end_idx = end_idx - max(3, elastic_length // 6)
            
            # Extract the middle 50% of the LINEAR ELASTIC region
            stress_middle = self.stress[middle_start_idx:middle_end_idx+1]
            strain_middle = self.strain[middle_start_idx:middle_end_idx+1]
            
            print(f"Enhanced ASTM E111 modulus calculation:")
            print(f"  Linear elastic region: indices 0 to {end_idx} ({elastic_length} points)")
            print(f"  Middle {middle_fraction*100:.0f}% used: indices {middle_start_idx} to {middle_end_idx} ({len(stress_middle)} points)")
            print(f"  Excluded {exclude_fraction*100:.0f}% from start and {exclude_fraction*100:.0f}% from end of LINEAR region")
            print(f"  Stress range: {stress_middle[0]:.0f} to {stress_middle[-1]:.0f} MPa")
            print(f"  Strain range: {strain_middle[0]*100:.3f}% to {strain_middle[-1]*100:.3f}%")
            
            # 1. Young's Modulus (linear regression on middle 50% of LINEAR ELASTIC region)
            slope, intercept, r_value, p_value, std_err = stats.linregress(strain_middle, stress_middle)
            
            # 2. Chord Modulus (between 25% and 75% of the selected middle elastic region)
            stress_25 = stress_middle[0] + 0.25 * (stress_middle[-1] - stress_middle[0])
            stress_75 = stress_middle[0] + 0.75 * (stress_middle[-1] - stress_middle[0])
            
            # Find closest points to these stress values
            idx_25 = np.argmin(np.abs(stress_middle - stress_25))
            idx_75 = np.argmin(np.abs(stress_middle - stress_75))
            
            if idx_25 != idx_75 and np.abs(strain_middle[idx_75] - strain_middle[idx_25]) > 1e-8:
                chord_modulus = (stress_middle[idx_75] - stress_middle[idx_25]) / (strain_middle[idx_75] - strain_middle[idx_25])
            else:
                chord_modulus = slope
            
            # 3. Tangent Modulus (derivative at midpoint)
            mid_idx = len(strain_middle) // 2
            window = min(5, len(strain_middle) // 4)  # Use small window around midpoint
            
            if mid_idx >= window and mid_idx + window < len(strain_middle):
                strain_tangent = strain_middle[mid_idx-window:mid_idx+window+1]
                stress_tangent = stress_middle[mid_idx-window:mid_idx+window+1]
                tangent_slope, _, tangent_r, _, _ = stats.linregress(strain_tangent, stress_tangent)
                tangent_modulus = tangent_slope
            else:
                tangent_modulus = slope
            
            # Store all modulus values
            self.results['youngs_modulus'] = slope  # MPa
            self.results['youngs_modulus_gpa'] = slope / 1000  # GPa
            self.results['chord_modulus'] = chord_modulus  # MPa
            self.results['chord_modulus_gpa'] = chord_modulus / 1000  # GPa
            self.results['tangent_modulus'] = tangent_modulus  # MPa
            self.results['tangent_modulus_gpa'] = tangent_modulus / 1000  # GPa
            
            # Legacy compatibility
            self.results['elastic_modulus'] = slope  # MPa
            self.results['elastic_modulus_gpa'] = slope / 1000  # GPa
            
            # Enhanced statistical and method information
            self.results['elastic_modulus_r_squared'] = r_value ** 2
            self.results['elastic_modulus_used_strain_gauge'] = False
            self.results['elastic_modulus_strain_range'] = (middle_start_idx, middle_end_idx)
            self.results['modulus_calculation_method'] = f"ASTM E111 Enhanced - Middle {middle_fraction*100:.0f}% of Linear Elastic Region"
            self.results['data_points_used'] = len(strain_middle)
            self.results['excluded_percentage'] = exclude_fraction
            self.results['linear_elastic_region_indices'] = (start_idx, end_idx)
            self.results['modulus_stress_range'] = (stress_middle[0], stress_middle[-1])
            self.results['modulus_strain_range_values'] = (strain_middle[0], strain_middle[-1])
            
            print(f"Results:")
            print(f"  Young's Modulus: {slope/1000:.2f} GPa (R² = {r_value**2:.5f})")
            print(f"  Chord Modulus: {chord_modulus/1000:.2f} GPa")
            print(f"  Tangent Modulus: {tangent_modulus/1000:.2f} GPa")
            print(f"  Stress range: {stress_middle[0]:.0f} to {stress_middle[-1]:.0f} MPa")
            print(f"  Data points used: {len(strain_middle)} (middle {middle_fraction*100:.0f}% of elastic region)")
            
        except Exception as e:
            print(f"Error calculating enhanced elastic modulus: {e}")
            # Final fallback: use simple slope between conservative bounds
            n_total = len(self.strain)
            start_idx = max(3, n_total // 20)  # Skip initial settling
            end_idx = min(n_total // 4, n_total - 1)  # Conservative end
            
            if end_idx > start_idx + 5:
                slope = (self.stress[end_idx] - self.stress[start_idx]) / (self.strain[end_idx] - self.strain[start_idx])
                self.results['elastic_modulus'] = slope
                self.results['elastic_modulus_gpa'] = slope / 1000
                self.results['elastic_modulus_strain_range'] = (start_idx, end_idx)
                print(f"Fallback modulus: {slope/1000:.1f} GPa")
            else:
                self.results['elastic_modulus'] = 200000  # Default 200 GPa
                self.results['elastic_modulus_gpa'] = 200
                self.results['elastic_modulus_strain_range'] = (0, 10)
            
            self.results['elastic_modulus_used_strain_gauge'] = False
    
    def calculate_yield_strength(self, offset=0.002, force_use_strain_gauge=False, use_extension_modulus=False):
        """
        Calculate 0.2% offset yield strength according to ASTM E8
        Uses strain gauge data if available and contains matching data points
        """
        try:
            # Determine which strain data to use based on ASTM E8 requirements
            use_strain_gauge_data = False
            if hasattr(self, 'strain_gauge') and self.strain_gauge is not None:
                # Only use strain gauge if it has matching number of data points
                if len(self.strain_gauge) == len(self.stress):
                    use_strain_gauge_data = True
                
            # Override with forced setting if specified
            if force_use_strain_gauge:
                if self.strain_gauge is not None and len(self.strain_gauge) == len(self.stress):
                    use_strain_gauge_data = True
                else:
                    print("Warning: Strain gauge data not available or incompatible, using extension data")
                    use_strain_gauge_data = False
                
            # Select appropriate strain data
            if use_strain_gauge_data:
                strain_to_use = self.strain_gauge
                print("Using strain gauge data for yield strength calculation")
            else:
                strain_to_use = self.strain
                print("Using extension data for yield strength calculation")
                
            # Use the appropriate elastic modulus based on ASTM E8 requirements
            if use_extension_modulus:
                # Force use of extension-based modulus
                slope_mpa = self.results.get('elastic_modulus', 200000)
                if self.results.get('elastic_modulus_used_strain_gauge', False):
                    # Recalculate using extension data if needed
                    self.calculate_elastic_modulus_best_fit()
                    slope_mpa = self.results['elastic_modulus']
            else:
                # Use the best available modulus (prefer strain gauge if available)
                slope_mpa = self.results.get('elastic_modulus', 200000)
                
            # Find intercept for offset line (y = mx + b, where x=0 when strain=offset)
            intercept = -slope_mpa * offset
            
            # Calculate the offset line: stress = slope * (strain - offset)
            offset_line_stress = slope_mpa * (strain_to_use - offset) + intercept
            
            # Find intersection with stress-strain curve
            diff = self.stress - offset_line_stress
            sign_changes = np.where(np.diff(np.signbit(diff)))[0]
            
            if len(sign_changes) > 0:
                # Interpolate to find exact intersection
                idx = sign_changes[0]
                x1, y1 = strain_to_use[idx], self.stress[idx]
                x2, y2 = strain_to_use[idx + 1], self.stress[idx + 1]
                
                # Linear interpolation
                yield_strain = x1 + (0 - diff[idx]) * (x2 - x1) / (diff[idx + 1] - diff[idx])
                yield_stress = slope_mpa * (yield_strain - offset) + intercept
                
                # Find closest point in original data
                yield_idx = np.argmin(np.abs(strain_to_use - yield_strain))
            else:
                # Fallback: find minimum difference
                yield_idx = np.argmin(np.abs(diff))
                yield_stress = self.stress[yield_idx]
                yield_strain = strain_to_use[yield_idx]
                
            # Document the method used
            if use_extension_modulus:
                method_note = "Calculated using extension-based elastic modulus and extension strain data per ASTM E8"
            else:
                if use_strain_gauge_data:
                    method_note = "Calculated using strain gauge data and strain gauge-based elastic modulus"
                else:
                    method_note = "Calculated using extension data and extension-based elastic modulus per ASTM E8"
                
            self.results['yield_strength_formula'] = method_note
            
            return yield_stress, yield_strain
            
        except Exception as e:
            print(f"Error calculating yield strength: {e}")
            # Fallback: use 80% of maximum stress
            max_stress_idx = np.argmax(self.stress)
            fallback_yield = self.stress[max_stress_idx] * 0.8
            yield_idx = np.argmin(np.abs(self.stress - fallback_yield))
            return self.stress[yield_idx], self.strain[yield_idx]

    def calculate_ultimate_tensile_strength(self):
        """Calculate Ultimate Tensile Strength (UTS)"""
        max_idx = np.argmax(self.stress)
        self.results['ultimate_tensile_strength'] = self.stress[max_idx]
        self.results['strain_at_uts'] = self.strain[max_idx]
        self.results['uts_index'] = max_idx
    
    def calculate_elongation(self):
        """Calculate elongation at fracture and after fracture."""
        # Determine if this is direct stress-strain data or load-extension data
        is_stress_strain_input = hasattr(self, '_is_stress_strain_input') and self._is_stress_strain_input
        
        # Engineering strain at fracture (always calculated)
        max_strain = np.max(self.strain)
        self.results['strain_at_fracture'] = max_strain
        
        if is_stress_strain_input:
            # For direct stress-strain data: elongation at fracture = max strain × 100
            self.results['elongation_at_fracture'] = max_strain * 100  # %
            self.results['elongation_at_fracture_mm'] = max_strain * self.L0  # mm (equivalent)
            print(f"Stress-strain data: elongation at fracture = {max_strain * 100:.2f}% (max strain × 100)")
        else:
            # For load-extension data: elongation at fracture from extension
            max_extension = np.max(self.extension)
            self.results['elongation_at_fracture'] = (max_extension / self.L0) * 100  # %
            self.results['elongation_at_fracture_mm'] = max_extension  # mm
            print(f"Load-extension data: elongation at fracture = {(max_extension / self.L0) * 100:.2f}% (from extension)")
        
        # Engineering strain elongation (for reference)
        self.results['elongation_at_fracture_eng'] = max_strain * 100
        
        # Elongation after fracture: only if final gauge length is available and valid
        final_length = self.specimen_params.get('final_length', None)
        if final_length is not None and final_length > 0 and final_length != self.L0:
            self.results['elongation_after_fracture'] = ((final_length - self.L0) / self.L0) * 100  # %
            self.results['elongation_after_fracture_mm'] = final_length - self.L0  # mm
        else:
            self.results['elongation_after_fracture'] = None
            self.results['elongation_after_fracture_mm'] = None

    def calculate_reduction_of_area(self):
        """Calculate reduction of area at fracture - only if final dimensions are available"""
        try:
            if self.specimen_params['specimen_type'] == 'round':
                final_diameter = self.specimen_params.get('final_diameter', None)
                if final_diameter is not None and final_diameter > 0:
                    Af = np.pi * (final_diameter / 2) ** 2
                    self.results['reduction_of_area'] = (self.A0 - Af) / self.A0 * 100
                else:
                    self.results['reduction_of_area'] = None
            else:
                # For flat specimens
                final_width = self.specimen_params.get('final_width', None)
                final_thickness = self.specimen_params.get('final_thickness', None)
                
                if final_width is not None and final_thickness is not None and final_width > 0 and final_thickness > 0:
                    Af = final_width * final_thickness
                    self.results['reduction_of_area'] = (self.A0 - Af) / self.A0 * 100
                else:
                    self.results['reduction_of_area'] = None
                    
        except Exception as e:
            print(f"Error calculating reduction of area: {e}")
            self.results['reduction_of_area'] = None
    
    def calculate_resilience(self):
        """
        Calculate resilience (area under stress-strain curve up to yield point)
        Resilience is the energy absorbed per unit volume up to the yield point.
        Units: When stress is in MPa and strain is dimensionless, result is in MJ/m³
        """
        try:
            if 'yield_strain' in self.results:
                yield_strain = self.results['yield_strain']
                yield_idx = np.argmin(np.abs(self.strain - yield_strain))
                
                # Ensure we include the yield point
                yield_idx = min(yield_idx + 1, len(self.stress) - 1)
                
                # Calculate area under curve using trapezoidal integration
                stress_segment = self.stress[:yield_idx+1]
                strain_segment = self.strain[:yield_idx+1]
                
                # Filter out any negative or invalid values
                valid_mask = (stress_segment >= 0) & (strain_segment >= 0) & np.isfinite(stress_segment) & np.isfinite(strain_segment)
                if np.sum(valid_mask) < 2:
                    print("Warning: Insufficient valid data for resilience calculation")
                    self.results['resilience'] = 0
                    return
                
                stress_filtered = stress_segment[valid_mask]
                strain_filtered = strain_segment[valid_mask]
                
                # Ensure monotonic increasing strain
                if len(strain_filtered) > 1:
                    sort_idx = np.argsort(strain_filtered)
                    stress_filtered = stress_filtered[sort_idx]
                    strain_filtered = strain_filtered[sort_idx]
                
                # Calculate resilience using trapezoidal integration
                resilience = np.trapz(stress_filtered, strain_filtered)
                
                # Store both raw and formatted values
                self.results['resilience'] = resilience  # MJ/m³ (since stress in MPa, strain dimensionless)
                self.results['resilience_calculation_method'] = f"Integration up to yield point (strain = {yield_strain*100:.3f}%)"
                self.results['resilience_data_points'] = len(stress_filtered)
                self.results['resilience_yield_index'] = yield_idx
                
                print(f"Resilience calculated: {resilience:.3f} MJ/m³ (up to yield strain {yield_strain*100:.3f}%)")
                
            else:
                # Use proportional limit if yield not available
                if 'proportional_limit_strain' in self.results:
                    prop_strain = self.results['proportional_limit_strain']
                    prop_idx = self.results.get('proportional_limit_index', len(self.strain)//4)
                    
                    # Ensure index is valid
                    prop_idx = min(prop_idx, len(self.stress) - 1)
                    
                    stress_segment = self.stress[:prop_idx+1]
                    strain_segment = self.strain[:prop_idx+1]
                    
                    # Filter valid values
                    valid_mask = (stress_segment >= 0) & (strain_segment >= 0) & np.isfinite(stress_segment) & np.isfinite(strain_segment)
                    if np.sum(valid_mask) >= 2:
                        stress_filtered = stress_segment[valid_mask]
                        strain_filtered = strain_segment[valid_mask]
                        
                        # Ensure monotonic increasing strain
                        if len(strain_filtered) > 1:
                            sort_idx = np.argsort(strain_filtered)
                            stress_filtered = stress_filtered[sort_idx]
                            strain_filtered = strain_filtered[sort_idx]
                        
                        resilience = np.trapz(stress_filtered, strain_filtered)
                        self.results['resilience'] = resilience
                        self.results['resilience_calculation_method'] = f"Integration up to proportional limit (strain = {prop_strain*100:.3f}%)"
                        self.results['resilience_data_points'] = len(stress_filtered)
                        
                        print(f"Resilience calculated using proportional limit: {resilience:.3f} MJ/m³")
                    else:
                        print("Warning: Insufficient valid data for resilience calculation using proportional limit")
                        self.results['resilience'] = 0
                else:
                    # Fallback: use first quarter of data
                    fallback_idx = len(self.strain) // 4
                    stress_segment = self.stress[:fallback_idx]
                    strain_segment = self.strain[:fallback_idx]
                    
                    valid_mask = (stress_segment >= 0) & (strain_segment >= 0) & np.isfinite(stress_segment) & np.isfinite(strain_segment)
                    if np.sum(valid_mask) >= 2:
                        stress_filtered = stress_segment[valid_mask]
                        strain_filtered = strain_segment[valid_mask]
                        
                        if len(strain_filtered) > 1:
                            sort_idx = np.argsort(strain_filtered)
                            stress_filtered = stress_filtered[sort_idx]
                            strain_filtered = strain_filtered[sort_idx]
                        
                        resilience = np.trapz(stress_filtered, strain_filtered)
                        self.results['resilience'] = resilience
                        self.results['resilience_calculation_method'] = f"Fallback: First quarter of data ({fallback_idx} points)"
                        self.results['resilience_data_points'] = len(stress_filtered)
                        
                        print(f"Resilience calculated using fallback method: {resilience:.3f} MJ/m³")
                    else:
                        print("Warning: Cannot calculate resilience - insufficient valid data")
                        self.results['resilience'] = 0
                        
        except Exception as e:
            print(f"Error calculating resilience: {e}")
            self.results['resilience'] = 0
    
    def calculate_toughness(self):
        """
        Calculate toughness (total area under stress-strain curve)
        Toughness is the total energy absorbed per unit volume until fracture.
        Units: When stress is in MPa and strain is dimensionless, result is in MJ/m³
        """
        try:
            # Filter out any negative or invalid values
            valid_mask = (self.stress >= 0) & (self.strain >= 0) & np.isfinite(self.stress) & np.isfinite(self.strain)
            
            if np.sum(valid_mask) < 2:
                print("Warning: Insufficient valid data for toughness calculation")
                self.results['toughness'] = 0
                return
            
            stress_filtered = self.stress[valid_mask]
            strain_filtered = self.strain[valid_mask]
            
            # Ensure monotonic increasing strain for proper integration
            if len(strain_filtered) > 1:
                sort_idx = np.argsort(strain_filtered)
                stress_filtered = stress_filtered[sort_idx]
                strain_filtered = strain_filtered[sort_idx]
            
            # Calculate toughness using trapezoidal integration
            toughness = np.trapz(stress_filtered, strain_filtered)
            
            # Store results with additional information
            self.results['toughness'] = toughness  # MJ/m³
            self.results['toughness_calculation_method'] = "Total area under stress-strain curve"
            self.results['toughness_data_points'] = len(stress_filtered)
            self.results['toughness_max_strain'] = np.max(strain_filtered)
            self.results['toughness_max_stress'] = np.max(stress_filtered)
            
            print(f"Toughness calculated: {toughness:.3f} MJ/m³ (total area under curve)")
            
            # Calculate ratio to resilience if available
            if 'resilience' in self.results and self.results['resilience'] > 0:
                toughness_to_resilience_ratio = toughness / self.results['resilience']
                self.results['toughness_to_resilience_ratio'] = toughness_to_resilience_ratio
                print(f"Toughness/Resilience ratio: {toughness_to_resilience_ratio:.1f}")
                
        except Exception as e:
            print(f"Error calculating toughness: {e}")
            self.results['toughness'] = 0
    
    def fit_ramberg_osgood(self):
        """
        Fit Ramberg-Osgood equation to stress-strain data
        epsilon = sigma/E + (sigma/K)^n
        Uses elastic modulus from strain gauge data if available, else extension.
        Enhanced version with better initial guesses, bounds, and validation.
        """
        def ramberg_osgood_strain(sigma, K, n, E_fixed):
            """Ramberg-Osgood equation: returns strain given stress"""
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                try:
                    # Ensure inputs are valid
                    sigma = np.asarray(sigma)
                    sigma = np.where(sigma < 0, 0, sigma)  # Ensure non-negative stress
                    
                    # Elastic component
                    elastic_strain = sigma / E_fixed
                    
                    # Plastic component - protect against overflow
                    stress_ratio = np.where(K > 0, sigma / K, 0)
                    stress_ratio = np.where(stress_ratio > 1e6, 1e6, stress_ratio)  # Cap to prevent overflow
                    plastic_strain = np.power(stress_ratio, n)
                    
                    # Total strain
                    total_strain = elastic_strain + plastic_strain
                    
                    # Handle any invalid values
                    total_strain = np.where(np.isfinite(total_strain), total_strain, elastic_strain)
                    return total_strain
                    
                except (ValueError, TypeError, OverflowError):
                    # Fallback to elastic only
                    return np.asarray(sigma) / E_fixed
        
        try:
            # Use elastic modulus already calculated
            E_fixed = self.results.get('elastic_modulus', 200000)  # MPa
            if E_fixed <= 0 or E_fixed < 1000:  # Sanity check for reasonable modulus
                print(f"Warning: Unrealistic elastic modulus ({E_fixed:.1f} MPa), using fallback (200 GPa)")
                E_fixed = 200000  # Default fallback (200 GPa)
            
            # Use data up to ultimate tensile strength to avoid necking effects
            uts_idx = self.results.get('uts_index', len(self.stress) - 1)
            
            # For fitting, use a reasonable portion of the data
            # Skip initial points that might have noise/settling
            start_idx = max(5, len(self.stress) // 100)  # Skip first 1% or at least 5 points
            
            # Pre-screening: check if we have sufficient data for Ramberg-Osgood fitting
            max_strain_in_data = np.max(self.strain)
            max_stress_in_data = np.max(self.stress)
            
            # Check for sufficient plastic deformation using 0.2% offset method
            elastic_limit_strain = 0.002  # 0.2% offset commonly used for yield
            plastic_strain_estimate = max_strain_in_data - elastic_limit_strain
            
            print(f"Pre-screening for Ramberg-Osgood fit:")
            print(f"Max strain: {max_strain_in_data*100:.3f}%, Max stress: {max_stress_in_data:.1f} MPa")
            print(f"Estimated plastic strain: {plastic_strain_estimate*100:.3f}%")
            
            # Check for sufficient plastic deformation
            if max_strain_in_data < 0.005:  # Less than 0.5% total strain
                print("Total strain < 0.5% - data is purely elastic. Skipping Ramberg-Osgood fit.")
                raise ValueError("Insufficient total strain for Ramberg-Osgood fitting")
            
            if plastic_strain_estimate <= 0.001:  # Less than 0.1% plastic strain
                print("Plastic strain < 0.1% - insufficient plastic deformation. Skipping Ramberg-Osgood fit.")
                raise ValueError("Insufficient plastic deformation for Ramberg-Osgood fitting")
            
            # Use data up to UTS, but check if we have enough plastic deformation
            # For Ramberg-Osgood to be meaningful, we need some plastic deformation
            elastic_limit_strain = np.max(self.stress) * 0.5 / E_fixed  # Conservative elastic limit
            plastic_strain_ratio = max_strain_in_data / elastic_limit_strain if elastic_limit_strain > 0 else 0
            
            if plastic_strain_ratio < 3:  # Less than 3x elastic strain means limited plasticity
                print(f"Warning: Limited plastic deformation detected (ratio: {plastic_strain_ratio:.1f})")
                print(f"Max strain: {max_strain_in_data*100:.3f}%, Elastic limit: {elastic_limit_strain*100:.3f}%")
                
                # For low-strain data, use all available data and accept lower quality fits
                end_idx = min(uts_idx + 1, len(self.stress))
                min_r_squared = 0.3  # Very low threshold for limited plasticity
                
                # Tighter bounds for limited plasticity
                K_bounds_factor = (0.8, 2.0)  # Tighter bounds
                n_bounds = (0.05, 0.5)  # Lower strain hardening
            else:
                # For high-strain data, limit to reasonable strain (20% max)
                max_strain_for_fit = 0.20  # 20% strain limit
                strain_limited_idx = np.where(self.strain <= max_strain_for_fit)[0]
                if len(strain_limited_idx) > 20:
                    end_idx = strain_limited_idx[-1]
                else:
                    end_idx = min(uts_idx + 1, len(self.stress))
                min_r_squared = 0.7  # Higher threshold for good plastic deformation
                
                # Standard bounds for good plasticity
                K_bounds_factor = (0.5, 3.0)  # Standard bounds
                n_bounds = (0.02, 1.5)  # Standard bounds
            
            stress_fit = self.stress[start_idx:end_idx]
            strain_fit = self.strain[start_idx:end_idx]
            
            # Remove any problematic data points
            valid_mask = (stress_fit > 0) & (strain_fit > 0) & np.isfinite(stress_fit) & np.isfinite(strain_fit)
            stress_fit = stress_fit[valid_mask]
            strain_fit = strain_fit[valid_mask]
            
            if len(stress_fit) < 15:
                raise ValueError(f"Insufficient valid data points for fitting ({len(stress_fit)} points)")
            
            print(f"Ramberg-Osgood fitting with {len(stress_fit)} points")
            print(f"Stress range: {np.min(stress_fit):.1f} to {np.max(stress_fit):.1f} MPa")
            print(f"Strain range: {np.min(strain_fit)*100:.3f} to {np.max(strain_fit)*100:.3f} %")
            print(f"Using elastic modulus: {E_fixed/1000:.1f} GPa")
            
            # Improved initial parameter estimates
            max_stress = np.max(stress_fit)
            max_strain = np.max(strain_fit)
            
            # Estimate K from yield strength if available, otherwise use UTS
            if 'yield_strength' in self.results and self.results['yield_strength'] > 0:
                K_guess = self.results['yield_strength'] * 1.5  # Typically K is 1.2-2x yield strength
            else:
                K_guess = max_stress * 1.2
            
            # Better estimate of n from strain hardening behavior
            # For limited plastic deformation, use more conservative estimates
            if plastic_strain_ratio < 3:
                # Limited plasticity - use conservative values typical for elastic-plastic materials
                n_guess = 0.1  # Low strain hardening
                K_guess = max_stress * 1.1  # K close to max stress
                print(f"Using conservative parameters for limited plasticity")
            else:
                # Significant plasticity - estimate from data
                # Look at the stress-strain relationship in plastic region
                # Find where plastic deformation dominates (stress > 0.5 * yield or 0.3 * max_stress)
                if 'yield_strength' in self.results and self.results['yield_strength'] > 0:
                    plastic_threshold = self.results['yield_strength']
                else:
                    plastic_threshold = max_stress * 0.3
                
                plastic_mask = stress_fit > plastic_threshold
                
                if np.sum(plastic_mask) > 5:
                    # Estimate strain hardening exponent from plastic region
                    plastic_stress = stress_fit[plastic_mask]
                    plastic_strain_total = strain_fit[plastic_mask]
                    # Subtract elastic component to get plastic strain
                    plastic_strain = plastic_strain_total - plastic_stress / E_fixed
                    plastic_strain = np.maximum(plastic_strain, 1e-8)  # Ensure positive values
                    
                    # Fit log(plastic_strain) vs log(stress/K_guess) to estimate n
                    try:
                        log_strain = np.log(plastic_strain)
                        log_stress_ratio = np.log(plastic_stress / K_guess)
                        valid_log = np.isfinite(log_strain) & np.isfinite(log_stress_ratio)
                        if np.sum(valid_log) > 3:
                            slope, _ = np.polyfit(log_stress_ratio[valid_log], log_strain[valid_log], 1)
                            n_guess = max(0.05, min(1.0, abs(slope)))  # Clamp to reasonable range
                        else:
                            n_guess = 0.2
                    except:
                        n_guess = 0.2
                else:
                    n_guess = 0.2  # Default for most metals
            
            # Set reasonable bounds based on material behavior
            K_bounds = (max_stress * K_bounds_factor[0], max_stress * K_bounds_factor[1])
            # n_bounds already set above based on plasticity level
            
            print(f"Initial guess: K = {K_guess:.0f} MPa, n = {n_guess:.3f}")
            print(f"Bounds: K = [{K_bounds[0]:.0f}, {K_bounds[1]:.0f}], n = [{n_bounds[0]:.3f}, {n_bounds[1]:.3f}]")
            print(f"Minimum R² threshold: {min_r_squared:.1f}")
            
            # Multiple attempts with different initial guesses for robustness
            best_params = None
            best_r_squared = -1
            best_error = float('inf')
            
            initial_guesses = [
                [K_guess, n_guess],
                [K_guess * 0.8, n_guess * 0.5],
                [K_guess * 1.2, n_guess * 1.5],
                [max_stress, 0.1],
                [max_stress * 0.7, 0.3],
                [max_stress * 1.5, 0.15]
            ]
            
            for i, (K_init, n_init) in enumerate(initial_guesses):
                try:
                    # Ensure initial guess is within bounds
                    K_init = max(K_bounds[0], min(K_bounds[1], K_init))
                    n_init = max(n_bounds[0], min(n_bounds[1], n_init))
                    
                    popt, pcov = curve_fit(
                        lambda sigma, K, n: ramberg_osgood_strain(sigma, K, n, E_fixed),
                        stress_fit,
                        strain_fit,
                        p0=[K_init, n_init],
                        bounds=([K_bounds[0], n_bounds[0]], [K_bounds[1], n_bounds[1]]),
                        maxfev=5000,
                        method='trf',  # Trust Region Reflective algorithm
                        ftol=1e-12,
                        xtol=1e-12
                    )
                    
                    # Calculate fit quality metrics
                    strain_pred = ramberg_osgood_strain(stress_fit, popt[0], popt[1], E_fixed)
                    residuals = strain_fit - strain_pred
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((strain_fit - np.mean(strain_fit)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Also calculate mean absolute percentage error
                    mape = np.mean(np.abs(residuals / strain_fit)) * 100
                    
                    print(f"Attempt {i+1}: R² = {r_squared:.4f}, MAPE = {mape:.2f}%, K = {popt[0]:.0f}, n = {popt[1]:.3f}")
                    
                    # Prefer higher R² and lower error, with adaptive quality threshold
                    if r_squared > best_r_squared and r_squared > min_r_squared:
                        best_params = popt
                        best_r_squared = r_squared
                        best_error = mape
                    
                except Exception as e:
                    print(f"Attempt {i+1} failed: {e}")
                    continue
            
            if best_params is None:
                raise ValueError("All fitting attempts failed")
            
            # Validate final parameters
            K_final, n_final = best_params
            if not (K_bounds[0] <= K_final <= K_bounds[1]) or not (n_bounds[0] <= n_final <= n_bounds[1]):
                print(f"Warning: Fitted parameters near bounds: K={K_final:.0f}, n={n_final:.3f}")
                # Don't fail, just warn
            
            # Store results
            self.results['ramberg_osgood_params'] = {
                'E': E_fixed, 
                'K': K_final, 
                'n': n_final
            }
            self.results['ramberg_osgood_r_squared'] = best_r_squared
            self.results['ramberg_osgood_mape'] = best_error
            self.results['ramberg_osgood_used_strain_gauge'] = self.results.get('elastic_modulus_used_strain_gauge', False)
            self.results['ramberg_osgood_data_points'] = len(stress_fit)
            self.results['ramberg_osgood_stress_range'] = (np.min(stress_fit), np.max(stress_fit))
            self.results['ramberg_osgood_strain_range'] = (np.min(strain_fit), np.max(strain_fit))
            
            print(f"Ramberg-Osgood fit successful:")
            print(f"  E = {E_fixed/1000:.1f} GPa (fixed)")
            print(f"  K = {K_final:.0f} MPa")
            print(f"  n = {n_final:.3f}")
            print(f"  R² = {best_r_squared:.4f}")
            print(f"  MAPE = {best_error:.2f}%")
            print(f"  Data points used: {len(stress_fit)}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error fitting Ramberg-Osgood equation: {error_msg}")
            
            # Provide informative feedback about why the fit failed
            if "Insufficient" in error_msg:
                print("Recommendation: Ramberg-Osgood fitting is not suitable for this data.")
                print("Reasons:")
                print("- Insufficient plastic deformation")
                print("- Data appears to be predominantly elastic")
                print("- Consider using simple elastic models instead")
                
                # Store informative results for user
                self.results['ramberg_osgood_params'] = None
                self.results['ramberg_osgood_note'] = "Not applicable - insufficient plastic deformation"
                self.results['ramberg_osgood_recommendation'] = "Use elastic modulus analysis instead"
            else:
                print("Recommendation: Try with different data or check for:")
                print("- Data quality issues")
                print("- Insufficient data points")
                print("- Unrealistic stress-strain values")
                
                self.results['ramberg_osgood_params'] = None
                self.results['ramberg_osgood_note'] = "Fitting failed - check data quality"
            
            self.results['ramberg_osgood_used_strain_gauge'] = False
            self.results['ramberg_osgood_r_squared'] = 0

    def analyze(self, both_yields=False):
        """
        Perform complete analysis of tensile test data
        If both_yields is True, calculate yield strength from both extension and strain gauge (if available)
        """
        print("Analyzing tensile test data...")
        self.smooth_data()
        self.find_proportional_limit()
        
        # Calculate elastic modulus - Calculate BOTH strain gauge and extension modulus when both data sources are available
        # Enhanced calculation per ASTM E111 with Young's, chord, and tangent modulus
        has_valid_strain_gauge = False
        if hasattr(self, 'strain_gauge') and self.strain_gauge is not None and np.any(self.strain_gauge):
            # Check if strain gauge data has same length as stress data (matching data points)
            if len(self.strain_gauge) == len(self.stress):
                has_valid_strain_gauge = True
                print("Calculating modulus from both strain gauge and extension data...")
            elif len(self.strain_gauge) > 0:
                # Try to make strain gauge data compatible with stress data
                try:
                    min_length = min(len(self.strain_gauge), len(self.stress))
                    if min_length > 10:  # Need at least 10 points for meaningful analysis
                        # Truncate to shorter length
                        self.strain_gauge = self.strain_gauge[:min_length]
                        # Also truncate other arrays to match
                        original_stress = self.stress.copy()
                        original_strain = self.strain.copy()
                        original_load = self.load.copy()
                        original_extension = self.extension.copy()
                        
                        self.stress = self.stress[:min_length]
                        self.strain = self.strain[:min_length]
                        self.load = self.load[:min_length]
                        self.extension = self.extension[:min_length]
                        
                        has_valid_strain_gauge = True
                        print(f"Adjusted data length to {min_length} points to accommodate strain gauge data...")
                        
                        # Store original data for later restoration if needed
                        self._original_data = {
                            'stress': original_stress,
                            'strain': original_strain, 
                            'load': original_load,
                            'extension': original_extension
                        }
                    else:
                        print(f"Insufficient strain gauge data ({len(self.strain_gauge)} points), using extension data only")
                except Exception as e:
                    print(f"Error processing strain gauge data: {e}, using extension data only")
            
            if has_valid_strain_gauge:
                
                # Calculate strain gauge modulus (ASTM E111)
                self.calculate_elastic_modulus_astm_e111_enhanced()
                # Store strain gauge results with specific naming
                self.results['youngs_modulus_strain_gauge'] = self.results['youngs_modulus']
                self.results['youngs_modulus_strain_gauge_gpa'] = self.results['youngs_modulus_gpa']
                self.results['youngs_modulus_strain_gauge_r_squared'] = self.results['elastic_modulus_r_squared']
                self.results['chord_modulus_strain_gauge'] = self.results['chord_modulus']
                self.results['chord_modulus_strain_gauge_gpa'] = self.results['chord_modulus_gpa']
                self.results['tangent_modulus_strain_gauge'] = self.results['tangent_modulus']
                self.results['tangent_modulus_strain_gauge_gpa'] = self.results['tangent_modulus_gpa']
                
                # Calculate extension modulus
                self.calculate_elastic_modulus_best_fit_enhanced()
                # Store extension results with specific naming
                self.results['youngs_modulus_extension'] = self.results['youngs_modulus']
                self.results['youngs_modulus_extension_gpa'] = self.results['youngs_modulus_gpa']
                self.results['youngs_modulus_extension_r_squared'] = self.results['elastic_modulus_r_squared']
                self.results['chord_modulus_extension'] = self.results['chord_modulus']
                self.results['chord_modulus_extension_gpa'] = self.results['chord_modulus_gpa']
                self.results['tangent_modulus_extension'] = self.results['tangent_modulus']
                self.results['tangent_modulus_extension_gpa'] = self.results['tangent_modulus_gpa']
                
                # Calculate percent difference between the two modulus values
                sg_modulus = self.results['youngs_modulus_strain_gauge_gpa']
                ext_modulus = self.results['youngs_modulus_extension_gpa']
                if ext_modulus > 0:
                    percent_diff = abs(sg_modulus - ext_modulus) / ext_modulus * 100
                    self.results['youngs_modulus_percent_difference'] = percent_diff
                
                # Set primary modulus to strain gauge (preferred)
                self.results['youngs_modulus'] = self.results['youngs_modulus_strain_gauge']
                self.results['youngs_modulus_gpa'] = self.results['youngs_modulus_strain_gauge_gpa']
                self.results['elastic_modulus_used_strain_gauge'] = True
                self.results['has_both_modulus_sources'] = True
                
                print("Using strain gauge data for primary modulus (ASTM E111)")
            else:
                self.calculate_elastic_modulus_best_fit_enhanced()  # Uses extension data
                print("Strain gauge data length mismatch, using extension data for enhanced elastic modulus")
                self.results['elastic_modulus_used_strain_gauge'] = False
                self.results['has_both_modulus_sources'] = False
        else:
            self.calculate_elastic_modulus_best_fit_enhanced()  # Uses extension data
            print("No strain gauge data available, using extension data for enhanced elastic modulus")
            self.results['elastic_modulus_used_strain_gauge'] = False
            self.results['has_both_modulus_sources'] = False
        
        # For yield strength calculation: Use extension data for ASTM E8 compliance UNLESS strain gauge has matching data
        # Extension data is preferred for yield strength for consistency with ASTM E8, but strain gauge can be used if complete
        if has_valid_strain_gauge:
            # Use strain gauge data for yield strength since it has complete matching data points
            ys_ext, ys_ext_strain = self.calculate_yield_strength(offset=0.002, force_use_strain_gauge=True)
            self.results['yield_strength'] = ys_ext
            self.results['yield_strain'] = ys_ext_strain
            self.results['yield_calculation_method'] = "0.2% offset using strain gauge data"
        else:
            # Use extension-based data for yield strength calculation per ASTM E8
            ys_ext, ys_ext_strain = self.calculate_yield_strength(offset=0.002, force_use_strain_gauge=False, use_extension_modulus=True)
            self.results['yield_strength'] = ys_ext
            self.results['yield_strain'] = ys_ext_strain
            self.results['yield_calculation_method'] = "0.2% offset using extension data"
        
        # Optionally calculate strain gauge-based yield strength for reference
        if both_yields and has_valid_strain_gauge:
            ys_sg, ys_sg_strain = self.calculate_yield_strength(offset=0.002, force_use_strain_gauge=True)
            self.results['yield_strength_strain_gauge'] = ys_sg
            self.results['yield_strain_strain_gauge'] = ys_sg_strain
            
        self.calculate_ultimate_tensile_strength()
        self.calculate_elongation()
        self.calculate_reduction_of_area()
        self.calculate_resilience()
        self.calculate_toughness()
        self.fit_ramberg_osgood()
        
        self.results['fracture_strength'] = self.stress[-1]
        self.results['uniform_elongation'] = self.results.get('strain_at_uts', 0) * 100
        
        print("Analysis complete!")
        return self.results

    def format_results(self, units_system="metric"):
        """Format analysis results for display with specified units"""
        if not self.results:
            return "No analysis results available."
        
        # Get unit labels and conversion factors
        if units_system == "metric":
            units = {
                'stress': 'MPa',
                'length': 'mm', 
                'force': 'N',
                'modulus': 'GPa',
                'energy': 'MJ/m^3'
            }
            stress_factor = 1.0
            length_factor = 1.0
            modulus_factor = 1.0
            energy_factor = 1.0
        else:  # imperial
            units = {
                'stress': 'ksi',
                'length': 'in',
                'force': 'lbf', 
                'modulus': 'Msi',
                'energy': 'in-lbf/in^3'
            }
            stress_factor = 0.000145038  # MPa to ksi
            length_factor = 0.0393701    # mm to in
            modulus_factor = 0.000145038 # GPa to Msi
            energy_factor = 5.71015e-6   # MJ/m^3 to in-lbf/in^3
        
        text = f"TENSILE TEST ANALYSIS RESULTS ({units_system.upper()} UNITS)\n"
        text += "=" * 60 + "\n\n"
        
        # Material Properties
        text += "MATERIAL PROPERTIES:\n"
        text += "-" * 20 + "\n"
        
        # Enhanced modulus reporting per ASTM E111
        if self.results.get('has_both_modulus_sources', False):
            text += "YOUNG'S MODULUS (DUAL SOURCE ANALYSIS):\n"
            text += "-" * 40 + "\n"
            
            # Strain gauge modulus
            if 'youngs_modulus_strain_gauge_gpa' in self.results:
                sg_value = self.results['youngs_modulus_strain_gauge_gpa'] * modulus_factor
                sg_r2 = self.results.get('youngs_modulus_strain_gauge_r_squared', 0)
                text += f"  Strain Gauge: {sg_value:.1f} {units['modulus']} (R² = {sg_r2:.4f})\n"
            
            # Extension modulus
            if 'youngs_modulus_extension_gpa' in self.results:
                ext_value = self.results['youngs_modulus_extension_gpa'] * modulus_factor
                ext_r2 = self.results.get('youngs_modulus_extension_r_squared', 0)
                text += f"  Extension: {ext_value:.1f} {units['modulus']} (R² = {ext_r2:.4f})\n"
            
            # Percent difference
            if 'youngs_modulus_percent_difference' in self.results:
                percent_diff = self.results['youngs_modulus_percent_difference']
                text += f"  Difference: {percent_diff:.1f}%\n"
            
            text += f"  Primary (used): Strain Gauge\n\n"
        
        # Young's modulus (primary)
        if 'youngs_modulus_gpa' in self.results and self.results['youngs_modulus_gpa']:
            youngs_value = self.results['youngs_modulus_gpa'] * modulus_factor
            text += f"Young's Modulus: {youngs_value:.1f} {units['modulus']}"
            if self.results.get('elastic_modulus_used_strain_gauge'):
                text += " (strain gauge)\n"
            else:
                text += " (extension)\n"
                
        # Chord and tangent modulus
        if 'chord_modulus_gpa' in self.results and self.results['chord_modulus_gpa']:
            chord_value = self.results['chord_modulus_gpa'] * modulus_factor
            text += f"Chord Modulus: {chord_value:.1f} {units['modulus']}\n"
            
        if 'tangent_modulus_gpa' in self.results and self.results['tangent_modulus_gpa']:
            tangent_value = self.results['tangent_modulus_gpa'] * modulus_factor
            text += f"Tangent Modulus: {tangent_value:.1f} {units['modulus']}\n"
        
        if 'yield_strength' in self.results:
            yield_value = self.results['yield_strength'] * stress_factor
            text += f"Yield Strength (0.2% offset): {yield_value:.1f} {units['stress']}\n"
        
        if 'ultimate_tensile_strength' in self.results:
            uts_value = self.results['ultimate_tensile_strength'] * stress_factor
            text += f"Ultimate Tensile Strength: {uts_value:.1f} {units['stress']}\n"
        
        if 'proportional_limit' in self.results:
            prop_value = self.results['proportional_limit'] * stress_factor
            text += f"Proportional Limit: {prop_value:.1f} {units['stress']}\n"
        
        text += "\nDUCTILITY PROPERTIES:\n"
        text += "-" * 20 + "\n"
        
        if 'elongation_at_fracture' in self.results:
            text += f"Elongation at Fracture: {self.results['elongation_at_fracture']:.1f}%\n"
        
        if 'elongation_after_fracture' in self.results and self.results['elongation_after_fracture'] is not None:
            text += f"Elongation after Fracture: {self.results['elongation_after_fracture']:.1f}%\n"
        else:
            text += "Elongation after Fracture: Not available (final gauge length not reported)\n"
        
        if 'reduction_of_area' in self.results and self.results['reduction_of_area'] is not None:
            text += f"Reduction of Area: {self.results['reduction_of_area']:.1f}%\n"
        else:
            text += "Reduction of Area: Not available (final dimensions not reported)\n"
        
        text += "\nENERGY PROPERTIES:\n"
        text += "-" * 18 + "\n"
        
        if 'resilience' in self.results:
            resilience_value = self.results['resilience'] * energy_factor
            text += f"Resilience: {resilience_value:.2f} {units['energy']}\n"
        
        if 'toughness' in self.results:
            toughness_value = self.results['toughness'] * energy_factor
            text += f"Toughness: {toughness_value:.2f} {units['energy']}\n"
        
        if 'ramberg_osgood_params' in self.results and self.results['ramberg_osgood_params']:
            text += "\nRAMBERG-OSGOOD PARAMETERS:\n"
            text += "-" * 25 + "\n"
            ro = self.results['ramberg_osgood_params']
            E_value = ro['E'] * stress_factor
            K_value = ro['K'] * stress_factor
            text += f"E: {E_value:.0f} {units['stress']}"
            if self.results.get('ramberg_osgood_used_strain_gauge'):
                text += " (strain gauge)\n"
            else:
                text += " (extension)\n"
            text += f"K: {K_value:.1f} {units['stress']}\n"
            text += f"n: {ro['n']:.3f}\n"
        
        # ASTM E111 Analysis Details
        if 'modulus_calculation_method' in self.results:
            text += "\nASTM E111 ANALYSIS DETAILS:\n"
            text += "-" * 28 + "\n"
            text += f"Calculation Method: {self.results['modulus_calculation_method']}\n"
            
            if 'data_points_used' in self.results:
                text += f"Data Points Used: {self.results['data_points_used']}\n"
                
            if 'excluded_percentage' in self.results:
                text += f"Excluded from Ends: {self.results['excluded_percentage']*100:.1f}%\n"
                
            if 'elastic_modulus_r_squared' in self.results:
                text += f"Linear Fit R²: {self.results['elastic_modulus_r_squared']:.4f}\n"
        
        return text

    def plot_stress_strain(self, show_true_stress=False):
        """Plot stress-strain curve with annotations for key points"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if show_true_stress:
            # Calculate true stress and strain
            true_stress = self.stress * (1 + self.strain)
            true_strain = np.log(1 + self.strain)
            
            ax.plot(self.strain * 100, self.stress, 'b-', linewidth=2, label='Engineering Stress-Strain')
            ax.plot(true_strain * 100, true_stress, 'r--', linewidth=2, label='True Stress-Strain')
            ax.set_title('Engineering vs True Stress-Strain Curves with Key Points')
        else:
            ax.plot(self.strain * 100, self.stress, 'b-', linewidth=2, label='Stress-Strain Curve')
            ax.set_title('Engineering Stress-Strain Curve with Key Points')
        
        # Add 0.2% offset line for yield strength
        if 'yield_strength' in self.results and 'youngs_modulus' in self.results:
            offset_strain = 0.002  # 0.2% offset
            modulus = self.results['youngs_modulus']
            
            # Create offset line starting from 0.2% strain
            max_strain_for_line = min(self.results.get('yield_strain', 0.01), 0.01)
            offset_strain_range = np.linspace(offset_strain, max_strain_for_line + 0.005, 100)
            offset_stress = modulus * (offset_strain_range - offset_strain)
            
            ax.plot(offset_strain_range * 100, offset_stress, 'g--', linewidth=2, 
                   label='0.2% Offset Line', alpha=0.8)
        
        # Annotate yield strength
        if 'yield_strength' in self.results and 'yield_strain' in self.results:
            yield_stress = self.results['yield_strength']
            yield_strain = self.results['yield_strain'] * 100
            
            ax.plot(yield_strain, yield_stress, 'ro', markersize=8, label='Yield Strength')
            ax.annotate(f'Yield Strength\n{yield_stress:.1f} MPa\n({yield_strain:.2f}% strain)',
                       xy=(yield_strain, yield_stress), xytext=(yield_strain + 1, yield_stress + 50),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                       fontsize=10, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))
        
        # Annotate ultimate tensile strength
        if 'ultimate_tensile_strength' in self.results and 'strain_at_uts' in self.results:
            uts = self.results['ultimate_tensile_strength']
            uts_strain = self.results['strain_at_uts'] * 100
            
            ax.plot(uts_strain, uts, 'go', markersize=8, label='Ultimate Tensile Strength')
            ax.annotate(f'Ultimate Tensile Strength\n{uts:.1f} MPa\n({uts_strain:.2f}% strain)',
                       xy=(uts_strain, uts), xytext=(uts_strain + 1, uts - 50),
                       arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                       fontsize=10, ha='left', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.8))
        
        # Annotate fracture point
        if len(self.stress) > 0 and len(self.strain) > 0:
            fracture_stress = self.stress[-1]
            fracture_strain = self.strain[-1] * 100
            
            ax.plot(fracture_strain, fracture_stress, 'mo', markersize=8, label='Fracture Point')
            ax.annotate(f'Fracture Point\n{fracture_stress:.1f} MPa\n({fracture_strain:.2f}% strain)',
                       xy=(fracture_strain, fracture_stress), xytext=(fracture_strain - 3, fracture_stress + 30),
                       arrowprops=dict(arrowstyle='->', color='magenta', lw=1.5),
                       fontsize=10, ha='right', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='magenta', alpha=0.8))
        
        # Annotate proportional limit if available
        if 'proportional_limit' in self.results and 'proportional_limit_strain' in self.results:
            prop_stress = self.results['proportional_limit']
            prop_strain = self.results['proportional_limit_strain'] * 100
            
            ax.plot(prop_strain, prop_stress, 'co', markersize=6, label='Proportional Limit')
            ax.annotate(f'Proportional Limit\n{prop_stress:.1f} MPa',
                       xy=(prop_strain, prop_stress), xytext=(prop_strain + 0.5, prop_stress + 20),
                       arrowprops=dict(arrowstyle='->', color='cyan', lw=1),
                       fontsize=9, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='cyan', alpha=0.7))
        
        # Add elastic region highlighting
        if 'linear_elastic_region_indices' in self.results:
            start_idx, end_idx = self.results['linear_elastic_region_indices']
            elastic_strain = self.strain[start_idx:end_idx] * 100
            elastic_stress = self.stress[start_idx:end_idx]
            ax.plot(elastic_strain, elastic_stress, 'y-', linewidth=4, alpha=0.6, label='Linear Elastic Region')
        
        ax.set_xlabel('Strain (%)', fontsize=12)
        ax.set_ylabel('Stress (MPa)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Add text box with key properties
        if self.results:
            textstr = f"""Key Properties:
Young's Modulus: {self.results.get('youngs_modulus_gpa', 0):.1f} GPa
Yield Strength: {self.results.get('yield_strength', 0):.1f} MPa
UTS: {self.results.get('ultimate_tensile_strength', 0):.1f} MPa
Elongation: {self.results.get('elongation_at_fracture', 0):.1f}%"""
            
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    def plot_ramberg_osgood_fit(self):
        """Plot Ramberg-Osgood fit comparison with annotations and fit quality assessment"""
        if 'ramberg_osgood_params' not in self.results or not self.results['ramberg_osgood_params']:
            print("Ramberg-Osgood parameters not available")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left plot: Full stress-strain curve with fit
        ro_params = self.results['ramberg_osgood_params']
        E, K, n = ro_params['E'], ro_params['K'], ro_params['n']
        
        # Plot experimental data
        ax1.plot(self.strain * 100, self.stress, 'b-', linewidth=2, label='Experimental Data', alpha=0.8)
        
        # Generate Ramberg-Osgood fitted curve
        # Use stress range from experimental data
        stress_min = max(1, np.min(self.stress[self.stress > 0]))  # Avoid zero stress
        stress_max = np.max(self.stress)
        stress_fit_range = np.linspace(stress_min, stress_max, 500)
        
        # Calculate strain using Ramberg-Osgood equation: ε = σ/E + (σ/K)^n
        def ramberg_osgood_strain(sigma):
            with np.errstate(over='ignore', invalid='ignore'):
                elastic_strain = sigma / E
                plastic_strain = np.power(sigma / K, n)
                total_strain = elastic_strain + plastic_strain
                return np.where(np.isfinite(total_strain), total_strain, elastic_strain)
        
        strain_ro = ramberg_osgood_strain(stress_fit_range)
        
        # Plot the fit
        ax1.plot(strain_ro * 100, stress_fit_range, 'r--', linewidth=2, 
                label=f'Ramberg-Osgood Fit\nE={E/1000:.1f} GPa, K={K:.0f} MPa, n={n:.3f}')
        
        # Highlight fitting region if available
        if 'ramberg_osgood_stress_range' in self.results:
            stress_range = self.results['ramberg_osgood_stress_range']
            strain_range = self.results['ramberg_osgood_strain_range']
            
            # Find points in fitting range
            fit_mask = (self.stress >= stress_range[0]) & (self.stress <= stress_range[1])
            if np.any(fit_mask):
                ax1.plot(self.strain[fit_mask] * 100, self.stress[fit_mask], 'g-', 
                        linewidth=3, alpha=0.7, label=f'Fitting Region ({np.sum(fit_mask)} points)')
        
        # Add key point annotations
        if 'yield_strength' in self.results and 'yield_strain' in self.results:
            yield_stress = self.results['yield_strength']
            yield_strain = self.results['yield_strain'] * 100
            
            ax1.plot(yield_strain, yield_stress, 'go', markersize=8, label='Yield Strength')
            ax1.annotate(f'Yield: {yield_stress:.1f} MPa\n{yield_strain:.2f}% strain',
                        xy=(yield_strain, yield_stress), xytext=(yield_strain + 0.5, yield_stress + 50),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                        fontsize=9, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.8))
        
        if 'ultimate_tensile_strength' in self.results and 'strain_at_uts' in self.results:
            uts = self.results['ultimate_tensile_strength']
            uts_strain = self.results['strain_at_uts'] * 100
            
            ax1.plot(uts_strain, uts, 'mo', markersize=8, label='Ultimate Tensile Strength')
            ax1.annotate(f'UTS: {uts:.1f} MPa\n{uts_strain:.2f}% strain',
                        xy=(uts_strain, uts), xytext=(uts_strain + 0.5, uts - 50),
                        arrowprops=dict(arrowstyle='->', color='magenta', lw=1.5),
                        fontsize=9, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='magenta', alpha=0.8))
        
        # Annotate fracture point
        if len(self.stress) > 0 and len(self.strain) > 0:
            fracture_stress = self.stress[-1]
            fracture_strain = self.strain[-1] * 100
            
            ax1.plot(fracture_strain, fracture_stress, 'mo', markersize=8, label='Fracture Point')
            ax1.annotate(f'Fracture Point\n{fracture_stress:.1f} MPa\n({fracture_strain:.2f}% strain)',
                       xy=(fracture_strain, fracture_stress), xytext=(fracture_strain - 3, fracture_stress + 30),
                       arrowprops=dict(arrowstyle='->', color='magenta', lw=1.5),
                       fontsize=10, ha='right', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='magenta', alpha=0.8))
        
        # Annotate proportional limit if available
        if 'proportional_limit' in self.results and 'proportional_limit_strain' in self.results:
            prop_stress = self.results['proportional_limit']
            prop_strain = self.results['proportional_limit_strain'] * 100
            
            ax1.plot(prop_strain, prop_stress, 'co', markersize=6, label='Proportional Limit')
            ax1.annotate(f'Proportional Limit\n{prop_stress:.1f} MPa',
                       xy=(prop_strain, prop_stress), xytext=(prop_strain + 0.5, prop_stress + 20),
                       arrowprops=dict(arrowstyle='->', color='cyan', lw=1),
                       fontsize=9, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='cyan', alpha=0.7))
        
        # Add elastic region highlighting
        if 'linear_elastic_region_indices' in self.results:
            start_idx, end_idx = self.results['linear_elastic_region_indices']
            elastic_strain = self.strain[start_idx:end_idx] * 100
            elastic_stress = self.stress[start_idx:end_idx]
            ax1.plot(elastic_strain, elastic_stress, 'y-', linewidth=4, alpha=0.6, label='Linear Elastic Region')
        
        ax1.set_xlabel('Strain (%)', fontsize=12)
        ax1.set_ylabel('Stress (MPa)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        
        # Right plot: Residuals analysis
        if 'ramberg_osgood_stress_range' in self.results:
            # Calculate residuals for the fitting region
            stress_range = self.results['ramberg_osgood_stress_range']
            fit_mask = (self.stress >= stress_range[0]) & (self.stress <= stress_range[1])
            
            if np.any(fit_mask):
                stress_fit_data = self.stress[fit_mask]
                strain_fit_data = self.strain[fit_mask]
                strain_pred = ramberg_osgood_strain(stress_fit_data)
                residuals = strain_fit_data - strain_pred
                
                # Plot residuals vs stress
                ax2.scatter(stress_fit_data, residuals * 100, alpha=0.6, s=20)
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
                ax2.set_xlabel('Stress (MPa)', fontsize=12)
                ax2.set_ylabel('Residuals (% strain)', fontsize=12)
                ax2.set_title('Fitting Residuals', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Add statistics
                mean_residual = np.mean(residuals) * 100
                std_residual = np.std(residuals) * 100
                max_residual = np.max(np.abs(residuals)) * 100
                
                residual_text = f'''Residual Statistics:
Mean: {mean_residual:.4f}% strain
Std Dev: {std_residual:.4f}% strain
Max |Residual|: {max_residual:.4f}% strain'''
                
                ax2.text(0.02, 0.98, residual_text, transform=ax2.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Add overall fit quality information
        fit_text = f'''Ramberg-Osgood Parameters:
E = {E/1000:.1f} GPa (elastic modulus)
K = {K:.0f} MPa (strength coefficient)
n = {n:.3f} (strain hardening exponent)

Fit Quality:
R² = {self.results.get('ramberg_osgood_r_squared', 0):.4f}'''
        
        if 'ramberg_osgood_mape' in self.results:
            fit_text += f'\nMAPE = {self.results["ramberg_osgood_mape"]:.2f}%'
        
        fit_text += f'\nData Points = {self.results.get("ramberg_osgood_data_points", 0)}'
        
        # Add material interpretation
        if n < 0.1:
            material_type = "Low strain hardening (elastic-perfectly plastic)"
        elif n < 0.3:
            material_type = "Moderate strain hardening"
        else:
            material_type = "High strain hardening"
        
        fit_text += f'\n\nMaterial Behavior:\n{material_type}'
        
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
        fig.text(0.02, 0.98, fit_text, fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig


class StressStrainAnalyzer(TensileTestAnalyzer):
    """
    Specialized analyzer for pre-calculated stress-strain data
    Inherits from TensileTestAnalyzer but works with stress-strain instead of load-extension
    """
    
    def __init__(self, stress_data, strain_data, specimen_params):
        """
        Initialize the analyzer with stress-strain data
        
        Parameters:
        -----------
        stress_data : array-like
            Engineering stress values (MPa)
        strain_data : array-like
            Engineering strain values (dimensionless)
        specimen_params : dict
            Dictionary containing specimen geometry and properties
        """
        # Store the direct stress-strain data
        self.stress = np.array(stress_data)
        self.strain = np.array(strain_data)
        self.specimen_params = specimen_params
        self.strain_gauge = None  # No strain gauge data available for stress-strain input
        
        # Validate input data
        if len(self.stress) == 0 or len(self.strain) == 0:
            raise ValueError("Stress and strain data arrays must not be empty")
        
        if len(self.stress) != len(self.strain):
            raise ValueError("Stress and strain data arrays must have the same length")
        
        # Validate data quality
        if np.any(np.isnan(self.stress)) or np.any(np.isnan(self.strain)):
            raise ValueError("Stress or strain data contains NaN values")
        
        if np.any(np.isinf(self.stress)) or np.any(np.isinf(self.strain)):
            raise ValueError("Stress or strain data contains infinite values")
        
        # Validate specimen parameters
        required_params = ['original_length', 'specimen_type']
        for param in required_params:
            if param not in specimen_params:
                raise ValueError(f"Required specimen parameter '{param}' is missing")
        
        if specimen_params['specimen_type'] not in ['round', 'flat']:
            raise ValueError("specimen_type must be 'round' or 'flat'")
        
        # Calculate basic parameters
        self.L0 = specimen_params['original_length']  # Original gauge length
        
        if specimen_params['specimen_type'] == 'round':
            diameter = specimen_params['original_diameter']
            if diameter <= 0:
                raise ValueError("Diameter must be positive")
            self.A0 = np.pi * (diameter / 2) ** 2
        else:  # flat specimen
            width = specimen_params['original_width']
            thickness = specimen_params['original_thickness']
            if width <= 0 or thickness <= 0:
                raise ValueError("Width and thickness must be positive")
            self.A0 = width * thickness
        
        if self.L0 <= 0:
            raise ValueError("Gauge length must be positive")
        if self.A0 <= 0:
            raise ValueError("Cross-sectional area must be positive")
        
        # Calculate corresponding load and extension from stress and strain
        self.load = self.stress * self.A0  # Load = stress * area
        self.extension = self.strain * self.L0  # Extension = strain * gauge length
        
        # Initialize results dictionary
        self.results = {}
    
    def analyze(self, both_yields=False):
        """
        Perform complete analysis of stress-strain data using ASTM E111 and ASTM E8 methods
        """
        print("Analyzing stress-strain data...")
        
        # Mark this as stress-strain input for proper elongation calculation
        self._is_stress_strain_input = True
        
        # For stress-strain data, we don't need to smooth since it's already processed
        # Find proportional limit to establish linear elastic region
        self.find_proportional_limit()
        
        # Calculate elastic modulus using ASTM E111 method (middle 50% of elastic region)
        self.calculate_elastic_modulus_best_fit_enhanced()
        
        # Calculate yield strength using ASTM E8 0.2% offset method
        yield_stress, yield_strain = self.calculate_yield_strength(offset=0.002, force_use_strain_gauge=False, use_extension_modulus=False)
        self.results['yield_strength'] = yield_stress
        self.results['yield_strain'] = yield_strain
        
        # Calculate other mechanical properties
        self.calculate_ultimate_tensile_strength()
        self.calculate_elongation()
        self.calculate_reduction_of_area()
        self.calculate_resilience()
        self.calculate_toughness()
        self.fit_ramberg_osgood()
        
        # Additional calculations specific to stress-strain data
        self.results['fracture_strength'] = self.stress[-1]
        self.results['uniform_elongation'] = self.results.get('strain_at_uts', 0) * 100
        
        # Set calculation method for documentation
        self.results['modulus_calculation_method'] = "ASTM E111 - Middle 50% of linear elastic region"
        self.results['yield_calculation_method'] = "ASTM E8 - 0.2% offset method"
        
        print("Analysis complete!")
        return self.results

class EnhancedTensileGUI:
    """Enhanced GUI with advanced plotting capabilities"""
    
    def __init__(self, root):
        self.root = root
        self.analyzer = None
        self.analyzers = []  # Store all analyzers for multi-specimen analysis
        self.specimen_names = []  # Store specimen names
        self.plot_canvas = None
        self.results_text = None
        self.setup_gui()
    
    def setup_gui(self):
        self.root.title("Tensile Test Analyzer v0.17 - Enhanced Interface")
        self.root.geometry("1200x800")
        
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Import Load-Extension Data", command=self.import_load_extension)
        file_menu.add_command(label="Import Stress-Strain Data", command=self.import_stress_strain)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Plot menu
        self.plot_menu = tk.Menu(menubar, tearoff=0)
        self.plot_menu.add_command(label="Annotated Stress-Strain", command=self.plot_annotated_stress_strain, state="disabled")
        self.plot_menu.add_command(label="Engineering Stress-Strain", command=self.plot_eng_stress_strain, state="disabled")
        self.plot_menu.add_command(label="True vs Engineering Stress-Strain", command=self.plot_true_stress_strain, state="disabled")
        self.plot_menu.add_command(label="Ramberg-Osgood Fit", command=self.plot_ramberg_osgood, state="disabled")
        self.plot_menu.add_command(label="Elastic Modulus Analysis", command=self.plot_modulus_analysis, state="disabled")
        self.plot_menu.add_command(label="Energy Analysis", command=self.plot_energy_analysis, state="disabled")
        menubar.add_cascade(label="Plots", menu=self.plot_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Show Results", command=self.show_results, state="disabled")
        view_menu.add_command(label="Show Formulae", command=self.show_formulae)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for results
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(left_frame, text="Analysis Results:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        # Results text area
        text_frame = ttk.Frame(left_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        
        self.results_text = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel for plots
        right_frame = ttk.Frame(main_frame, width=500)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5,0))
        right_frame.pack_propagate(False)
        
        ttk.Label(right_frame, text="Plots:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        self.plot_frame = ttk.Frame(right_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        
        # Status bar
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready - Import data to begin analysis")
        self.status_label.pack(side=tk.LEFT)
        
        # Store menu references
        self.view_menu = view_menu
        
        # Initial welcome message
        welcome_msg = """Tensile Test Analyzer v0.17 - Enhanced Interface with Annotated Plots

New Features:
- ASTM E111 compliant modulus calculation using middle 50% of linear elastic region
- Fully annotated stress-strain curves with key points marked
- 0.2% offset line visualization for yield strength determination
- True stress-strain curve plotting with annotations
- Ramberg-Osgood parameter fitting and visualization with key points
- Enhanced energy analysis plotting (resilience and toughness) with annotations
- Enhanced error handling for missing final dimensions

Plotting Features:
- Annotated Stress-Strain: Complete plot with yield, UTS, fracture points, and offset line
- Key point markers: Yield strength, Ultimate tensile strength, Fracture point, Proportional limit
- Interactive annotations with values and coordinates
- Color-coded regions: Linear elastic, yield region, plastic deformation
- Property summary boxes on all plots

Key Improvements:
- More accurate elastic modulus calculation
- Better handling of incomplete specimen data
- Advanced plotting capabilities with comprehensive annotations
- Robust analysis methods with visual verification

Instructions:
1. Use File menu to import your data (Load-Extension or Stress-Strain)
2. View analysis results in this panel
3. Use Plots menu for annotated visualizations
4. Start with "Annotated Stress-Strain" for complete overview
5. Export results when complete

Ready to analyze your tensile test data with enhanced visualizations!"""
        
        self.results_text.insert("1.0", welcome_msg)
        self.results_text.config(state="disabled")
    
    def import_load_extension(self):
        """Import load-extension data"""
        file_path = filedialog.askopenfilename(
            title="Select Load-Extension Data File",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.process_load_extension_file(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process file: {str(e)}")
    
    def import_stress_strain(self):
        """Import stress-strain data"""
        file_path = filedialog.askopenfilename(
            title="Select Stress-Strain Data File",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.process_stress_strain_file(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process file: {str(e)}")
    
    def process_load_extension_file(self, file_path):
        """Process load-extension data file with automatic specimen dimension detection"""
        import pandas as pd
        
        if file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else:
            data = pd.read_csv(file_path)
        
        # Try to detect multiple specimens and extract specimen parameters from file
        specimens_data = self.detect_multiple_specimens_load_extension(data)
        
        if not specimens_data:
            raise ValueError("Could not detect valid specimen data in file")
        
        if len(specimens_data) == 1:
            # Single specimen - analyze directly
            self.analyze_single_specimen_load_extension(specimens_data[0], file_path)
            
        else:
            # Multiple specimens detected - ask user what to do
            choice = self.show_multiple_specimens_dialog(specimens_data, file_path, "load-extension")
            
            if choice == "all":
                # Analyze all specimens
                self.analyze_all_specimens_load_extension(specimens_data, file_path)
            elif choice == "select":
                # Let user select specific specimen
                selected_index = self.select_specimen_dialog(specimens_data, file_path, "load-extension")
                if selected_index is not None:
                    self.analyze_single_specimen_load_extension(specimens_data[selected_index], file_path)
            # If choice is None (cancelled), do nothing
    
    def analyze_single_specimen_load_extension(self, specimen_info, file_path):
        """Analyze a single load-extension specimen"""
        load_data = specimen_info['load']
        extension_data = specimen_info['extension']
        params = specimen_info['params']
        strain_gauge_data = specimen_info.get('strain_gauge', None)
        specimen_name = specimen_info.get('name', 'Specimen 1')
        
        # Create analyzer and run analysis
        self.analyzer = TensileTestAnalyzer(load_data, extension_data, params, strain_gauge_data)
        self.analyzer.analyze()
        
        # Clear multi-specimen data since we have single specimen
        self.analyzers = []
        self.specimen_names = []
        
        self.update_results_display()
        self.enable_menus()
        
        filename = os.path.basename(file_path)
        success_msg = f"Successfully analyzed load-extension data from {filename}\nSpecimen: {specimen_name}"
        if strain_gauge_data is not None:
            success_msg += "\nStrain gauge data included for enhanced modulus calculation"
        
        messagebox.showinfo("Success", success_msg)
        self.status_label.config(text=f"Analyzed: {filename} - {specimen_name}")
    
    def analyze_all_specimens_load_extension(self, specimens_data, file_path):
        """Analyze all load-extension specimens"""
        self.analyzers = []
        self.specimen_names = []
        failed_specimens = []
        
        for i, specimen_info in enumerate(specimens_data):
            try:
                load_data = specimen_info['load']
                extension_data = specimen_info['extension']
                params = specimen_info['params']
                strain_gauge_data = specimen_info.get('strain_gauge', None)
                specimen_name = specimen_info.get('name', f"Specimen {i+1}")
                
                # Create analyzer and run analysis
                analyzer = TensileTestAnalyzer(load_data, extension_data, params, strain_gauge_data)
                analyzer.analyze()
                
                self.analyzers.append(analyzer)
                self.specimen_names.append(specimen_name)
                
            except Exception as e:
                failed_specimens.append(f"Specimen {i+1}: {str(e)}")
                print(f"Warning: Failed to analyze specimen {i+1}: {str(e)}")
                continue
        
        # Clear single specimen analyzer since we have multiple
        self.analyzer = None
        
        if self.analyzers:
            self.update_results_display()
            self.enable_menus()
            
            filename = os.path.basename(file_path)
            success_msg = f"Successfully analyzed {len(self.analyzers)} out of {len(specimens_data)} specimens from {filename}"
            
            if failed_specimens:
                success_msg += f"\n\nFailed to analyze {len(failed_specimens)} specimens:\n" + "\n".join(failed_specimens[:3])
                if len(failed_specimens) > 3:
                    success_msg += f"\n... and {len(failed_specimens) - 3} more"
            
            messagebox.showinfo("Analysis Complete", success_msg)
            self.status_label.config(text=f"Analyzed: {filename} ({len(self.analyzers)} specimens)")
        else:
            messagebox.showerror("Error", "Failed to analyze any specimens from the file")
    
    def detect_multiple_specimens_load_extension(self, data):
        """Detect multiple specimens in load-extension data and extract parameters"""
        specimens = []
        
        # Strategy 1: Check if this is a QB2-style file with specimen info in header
        if self.is_qb2_style_file(data):
            return self.parse_qb2_style_file(data)
        
        # Strategy 2: Look for clear column headers and multiple data columns
        # Assume data starts after any header info
        data_start_row = self.find_data_start_row(data)
        
        if data_start_row is None:
            # Fallback: assume simple format starting from row 0
            data_start_row = 0
        
        # Extract numeric data
        numeric_data = data.iloc[data_start_row:].apply(pd.to_numeric, errors='coerce')
        
        # Find columns with sufficient non-null data
        valid_columns = []
        for col in numeric_data.columns:
            non_null_count = numeric_data[col].count()
            if non_null_count > 50:  # Need at least 50 data points
                valid_columns.append(col)
        
        if len(valid_columns) >= 2:
            # Assume first two columns are Load and Extension
            load_data = numeric_data[valid_columns[0]].dropna().values
            extension_data = numeric_data[valid_columns[1]].dropna().values
            
            # Check for strain gauge data
            strain_gauge_data = None
            if len(valid_columns) >= 3:
                strain_gauge_candidate = numeric_data[valid_columns[2]].dropna().values
                if len(strain_gauge_candidate) == len(load_data):
                    strain_gauge_data = strain_gauge_candidate
            
            # Try to extract specimen parameters from file
            params = self.extract_specimen_params_from_file(data)
            
            specimen_info = {
                'load': load_data,
                'extension': extension_data,
                'strain_gauge': strain_gauge_data,
                'params': params,
                'name': f"Specimen 1"
            }
            specimens.append(specimen_info)
        
        return specimens
    
    def process_stress_strain_file(self, file_path):
        """Process stress-strain data file with automatic parameter detection and multi-specimen support"""
        try:
            # Detect multiple specimens and extract parameters
            specimens = self.detect_multiple_specimens_stress_strain(file_path)
            
            if not specimens:
                # No specimens detected, ask user for parameters
                params = self.get_specimen_parameters()
                if not params:
                    return
                
                # Load simple data
                import pandas as pd
                if file_path.endswith('.xlsx'):
                    data = pd.read_excel(file_path)
                else:
                    data = pd.read_csv(file_path)
                
                if len(data.columns) < 2:
                    raise ValueError("File must have at least 2 columns (Stress, Strain)")
                
                stress_data = data.iloc[:, 0].dropna()
                strain_data = data.iloc[:, 1].dropna()
                
                specimens = [{'stress': stress_data.values, 'strain': strain_data.values, 'params': params, 'name': 'Specimen 1'}]
            
            # If multiple specimens, analyze all; otherwise use the single specimen
            if len(specimens) > 1:
                # Multiple specimens - analyze all
                self.analyzers = []
                self.specimen_names = []
                
                for i, specimen_data in enumerate(specimens):
                    try:
                        # Create analyzer and run analysis
                        analyzer = StressStrainAnalyzer(
                            specimen_data['stress'], 
                            specimen_data['strain'], 
                            specimen_data['params']
                        )
                        analyzer.analyze()
                        
                        self.analyzers.append(analyzer)
                        self.specimen_names.append(specimen_data['name'])
                        
                    except Exception as e:
                        print(f"Warning: Failed to analyze specimen {specimen_data['name']}: {str(e)}")
                        continue
                
                # Clear single specimen analyzer since we have multiple
                self.analyzer = None
                
                if self.analyzers:
                    self.update_results_display()
                    self.enable_menus()
                    
                    filename = os.path.basename(file_path)
                    success_msg = f"Successfully analyzed {len(self.analyzers)} specimens from {filename}"
                    messagebox.showinfo("Success", success_msg)
                    self.status_label.config(text=f"Analyzed: {filename} ({len(self.analyzers)} specimens)")
                else:
                    messagebox.showerror("Error", "Failed to analyze any specimens from the file")
                    return
                
            else:
                # Single specimen
                selected_data = specimens[0]
                
                # Create analyzer and run analysis
                self.analyzer = StressStrainAnalyzer(
                    selected_data['stress'], 
                    selected_data['strain'], 
                    selected_data['params']
                )
                self.analyzer.analyze()
                
                # Clear multi-specimen data since we have single specimen
                self.analyzers = []
                self.specimen_names = []
                
                self.update_results_display()
                self.enable_menus()
                
                filename = os.path.basename(file_path)
                specimen_name = selected_data['name']
                messagebox.showinfo("Success", f"Successfully analyzed stress-strain data from {filename}\nSpecimen: {specimen_name}")
                self.status_label.config(text=f"Analyzed: {filename} - {specimen_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process stress-strain file: {str(e)}")
            self.status_label.config(text="Error processing file")
    
    def get_specimen_parameters(self):
        """Get specimen parameters from user"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Specimen Parameters")
        dialog.geometry("400x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 100))
        
        params = {}
        
        # Specimen type
        ttk.Label(dialog, text="Specimen Type:", font=("Arial", 10, "bold")).pack(pady=5)
        specimen_type = tk.StringVar(value="round")
        ttk.Radiobutton(dialog, text="Round", variable=specimen_type, value="round").pack()
        ttk.Radiobutton(dialog, text="Flat", variable=specimen_type, value="flat").pack()
        
        # Original dimensions
        ttk.Label(dialog, text="Original Dimensions:", font=("Arial", 10, "bold")).pack(pady=(15,5))
        
        dim_frame = ttk.Frame(dialog)
        dim_frame.pack(pady=5)
        
               
        ttk.Label(dim_frame, text="Gauge Length (mm):").grid(row=0, column=0, sticky="w", padx=5)
        length_var = tk.StringVar(value="25.0")
        ttk.Entry(dim_frame, textvariable=length_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(dim_frame, text="Diameter (mm):").grid(row=1, column=0, sticky="w", padx=5)
        diameter_var = tk.StringVar(value="6.0")
        diameter_entry = ttk.Entry(dim_frame, textvariable=diameter_var, width=10)
        diameter_entry.grid(row=1, column=1, padx=5)
        
        ttk.Label(dim_frame, text="Width (mm):").grid(row=2, column=0, sticky="w", padx=5)
        width_var = tk.StringVar(value="12.5")
        width_entry = ttk.Entry(dim_frame, textvariable=width_var, width=10)
        width_entry.grid(row=2, column=1, padx=5)
        
        ttk.Label(dim_frame, text="Thickness (mm):").grid(row=3, column=0, sticky="w", padx=5)
        thickness_var = tk.StringVar(value="3.0")
        thickness_entry = ttk.Entry(dim_frame, textvariable=thickness_var, width=10)
        thickness_entry.grid(row=3, column=1, padx=5)
        
        # Final dimensions (optional)
        ttk.Label(dialog, text="Final Dimensions (Optional):", font=("Arial", 10, "bold")).pack(pady=(15,5))
        ttk.Label(dialog, text="Leave blank if not measured", font=("Arial", 8), foreground="gray").pack()
        
        final_frame = ttk.Frame(dialog)
        final_frame.pack(pady=5)
        
        ttk.Label(final_frame, text="Final Gauge Length (mm):").grid(row=0, column=0, sticky="w", padx=5)
        final_length_var = tk.StringVar()
        ttk.Entry(final_frame, textvariable=final_length_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(final_frame, text="Final Diameter (mm):").grid(row=1, column=0, sticky="w", padx=5)
        final_diameter_var = tk.StringVar()
        ttk.Entry(final_frame, textvariable=final_diameter_var, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(final_frame, text="Final Width (mm):").grid(row=2, column=0, sticky="w", padx=5)
        final_width_var = tk.StringVar()
        ttk.Entry(final_frame, textvariable=final_width_var, width=10).grid(row=2, column=1, padx=5)
        
        ttk.Label(final_frame, text="Final Thickness (mm):").grid(row=3, column=0, sticky="w", padx=5)
        final_thickness_var = tk.StringVar()
        ttk.Entry(final_frame, textvariable=final_thickness_var, width=10).grid(row=3, column=1, padx=5)
        
        def update_entries():
            if specimen_type.get() == "round":
                diameter_entry.config(state="normal")
                width_entry.config(state="disabled")
                thickness_entry.config(state="disabled")
            else:
                diameter_entry.config(state="disabled")
                width_entry.config(state="normal")
                thickness_entry.config(state="normal")
        
        specimen_type.trace('w', lambda *args: update_entries())
        update_entries()
        
        result = {"confirmed": False}
        
        def confirm():
            try:
                params['specimen_type'] = specimen_type.get()
                params['original_length'] = float(length_var.get())
                
                if specimen_type.get() == "round":
                    params['original_diameter'] = float(diameter_var.get())
                    params['original_width'] = 12.5  # Default
                    params['original_thickness'] = 3.0  # Default
                else:
                    params['original_diameter'] = 6.0  # Default
                    params['original_width'] = float(width_var.get())
                    params['original_thickness'] = float(thickness_var.get())
                
                # Final dimensions (optional)
                if final_length_var.get().strip():
                    params['final_length'] = float(final_length_var.get())
                else:
                    params['final_length'] = None
                    
                if final_diameter_var.get().strip():
                    params['final_diameter'] = float(final_diameter_var.get())
                else:
                    params['final_diameter'] = None
                    
                if final_width_var.get().strip():
                    params['final_width'] = float(final_width_var.get())
                else:
                    params['final_width'] = None
                    
                if final_thickness_var.get().strip():
                    params['final_thickness'] = float(final_thickness_var.get())
                else:
                    params['final_thickness'] = None
                
                result["confirmed"] = True
                result["params"] = params
                dialog.destroy()
                
            except ValueError as e:
                messagebox.showerror("Invalid Input", "Please enter valid numeric values for all dimensions.")
        
        def cancel():
            result["confirmed"] = False
            dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="OK", command=confirm).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=10)
        
        dialog.wait_window()
        
        if result["confirmed"]:
            return result["params"]
        else:
            return None
    
    def update_results_display(self):
        """Update the results text display"""
        if not self.results_text:
            # Create results text widget if it doesn't exist
            self.results_text = tk.Text(self.plot_frame, wrap=tk.WORD, font=("Consolas", 10))
            self.results_text.pack(fill=tk.BOTH, expand=True)
        
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        
        if self.analyzers:
            # Multi-specimen display
            content = f"Multiple Specimens Analyzed ({len(self.analyzers)} total)\n"
            content += "=" * 50 + "\n\n"
            content += "Use 'View > Show Results' menu to see detailed results for all specimens.\n\n"
            content += "Summary:\n"
            for i, analyzer in enumerate(self.analyzers):
                name = self.specimen_names[i] if i < len(self.specimen_names) else f"Specimen {i+1}"
                uts = analyzer.results.get('ultimate_strength', 0)
                yield_str = analyzer.results.get('yield_strength', 0)
                content += f"  {name}: UTS = {uts:.1f} MPa, Yield = {yield_str:.1f} MPa\n"
            self.results_text.insert("1.0", content)
        elif self.analyzer:
            # Single specimen display
            self.results_text.insert("1.0", self.analyzer.format_results())
        else:
            self.results_text.insert("1.0", "No analysis results available.")
        
        self.results_text.config(state="disabled")
    
    def enable_menus(self):
        """Enable menu items after successful analysis"""
        self.plot_menu.entryconfig("Annotated Stress-Strain", state="normal")
        self.plot_menu.entryconfig("Engineering Stress-Strain", state="normal")
        self.plot_menu.entryconfig("True vs Engineering Stress-Strain", state="normal")
        self.plot_menu.entryconfig("Ramberg-Osgood Fit", state="normal")
        self.plot_menu.entryconfig("Elastic Modulus Analysis", state="normal")
        self.plot_menu.entryconfig("Energy Analysis", state="normal")
        self.view_menu.entryconfig("Show Results", state="normal")
    
    def plot_annotated_stress_strain(self):
        """Plot fully annotated stress-strain curve with all key points"""
        analyzer = self.analyzer if self.analyzer else (self.analyzers[0] if self.analyzers else None)
        if analyzer:
            self.clear_plot()
            fig = analyzer.plot_stress_strain(show_true_stress=False)
            self.display_plot(fig)
    
    def plot_eng_stress_strain(self):
        """Plot engineering stress-strain curve"""
        analyzer = self.analyzer if self.analyzer else (self.analyzers[0] if self.analyzers else None)
        if analyzer:
            self.clear_plot()
            fig = analyzer.plot_stress_strain(show_true_stress=False)
            self.display_plot(fig)
    
    def plot_true_stress_strain(self):
        """Plot true vs engineering stress-strain comparison"""
        analyzer = self.analyzer if self.analyzer else (self.analyzers[0] if self.analyzers else None)
        if analyzer:
            self.clear_plot()
            fig = analyzer.plot_stress_strain(show_true_stress=True)
            self.display_plot(fig)
    
    def plot_ramberg_osgood(self):
        """Plot Ramberg-Osgood fit"""
        analyzer = self.analyzer if self.analyzer else (self.analyzers[0] if self.analyzers else None)
        if analyzer:
            self.clear_plot()
            fig = analyzer.plot_ramberg_osgood_fit()
            if fig:
                self.display_plot(fig)
            else:
                messagebox.showwarning("No Data", "Ramberg-Osgood parameters not available")
    
    def plot_modulus_analysis(self):
        """Plot elastic modulus analysis showing middle 50% usage"""
        analyzer = self.analyzer if self.analyzer else (self.analyzers[0] if self.analyzers else None)
        if analyzer:
            self.clear_plot()
            
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Plot full curve
            ax.plot(analyzer.strain * 100, analyzer.stress, 'b-', linewidth=2, 
                   label='Full Stress-Strain Curve', alpha=0.7)
            
            # Highlight linear elastic region if available
            if 'linear_elastic_region_indices' in analyzer.results:
                start_idx, end_idx = analyzer.results['linear_elastic_region_indices']
                elastic_strain = analyzer.strain[start_idx:end_idx] * 100
                elastic_stress = analyzer.stress[start_idx:end_idx]
                ax.plot(elastic_strain, elastic_stress, 'g-', linewidth=3, 
                       label=f'Linear Elastic Region ({end_idx-start_idx} points)')
            
            # Highlight middle 50% used for modulus calculation
            if 'elastic_modulus_strain_range' in analyzer.results:
                mid_start, mid_end = analyzer.results['elastic_modulus_strain_range']
                mid_strain = analyzer.strain[mid_start:mid_end] * 100
                mid_stress = analyzer.stress[mid_start:mid_end]
                ax.plot(mid_strain, mid_stress, 'r-', linewidth=4, 
                       label=f'Middle 50% Used for Modulus ({mid_end-mid_start} points)')
            
            # Add modulus line
            if 'elastic_modulus' in analyzer.results:
                modulus = analyzer.results['elastic_modulus']
                strain_range = analyzer.results.get('modulus_strain_range_values', (0, 0.005))
                strain_line = np.linspace(strain_range[0], strain_range[1], 100)
                stress_line = modulus * strain_line
                ax.plot(strain_line * 100, stress_line, 'k--', linewidth=2, 
                       label=f'Modulus Line: {modulus/1000:.1f} GPa (R² = {analyzer.results.get("elastic_modulus_r_squared", 0):.4f})')
            
            # Add proportional limit marker
            if 'proportional_limit' in analyzer.results:
                prop_stress = analyzer.results['proportional_limit']
                prop_strain = analyzer.results['proportional_limit_strain'] * 100
                ax.plot(prop_strain, prop_stress, 'mo', markersize=8, 
                       label=f'Proportional Limit: {prop_stress:.0f} MPa')
            
            ax.set_xlabel('Strain (%)')
            ax.set_ylabel('Stress (MPa)')
            ax.set_title('Elastic Modulus Analysis - ASTM E111 Enhanced Method')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add text box with analysis details
            if 'modulus_calculation_method' in analyzer.results:
                textstr = f"""Analysis Method: {analyzer.results['modulus_calculation_method']}
Data Points Used: {analyzer.results.get('data_points_used', 'N/A')}
Excluded from Ends: {analyzer.results.get('excluded_percentage', 0)*100:.1f}%
R² Value: {analyzer.results.get('elastic_modulus_r_squared', 0):.4f}"""
                
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            self.display_plot(fig)
    
    def plot_energy_analysis(self):
        """Plot energy analysis (resilience and toughness) with annotations"""
        analyzer = self.analyzer if self.analyzer else (self.analyzers[0] if self.analyzers else None)
        if analyzer:
            self.clear_plot()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Resilience plot (area under curve to yield point)
            ax1.plot(analyzer.strain * 100, analyzer.stress, 'b-', linewidth=2, label='Stress-Strain Curve')
            
            if 'yield_strain' in analyzer.results:
                yield_idx = np.argmin(np.abs(analyzer.strain - analyzer.results['yield_strain']))
                yield_strain = analyzer.strain[:yield_idx+1] * 100
                yield_stress = analyzer.stress[:yield_idx+1]
                
                ax1.fill_between(yield_strain, yield_stress, alpha=0.4, color='green', 
                               label=f'Resilience: {analyzer.results.get("resilience", 0):.2f} MJ/m³')
                
                # Mark yield point
                yield_stress_val = analyzer.results['yield_strength']
                yield_strain_val = analyzer.results['yield_strain'] * 100
                ax1.plot(yield_strain_val, yield_stress_val, 'ro', markersize=8, label='Yield Point')
                ax1.axvline(yield_strain_val, color='red', linestyle='--', alpha=0.7)
                
                # Annotate yield point
                ax1.annotate(f'Yield: {yield_stress_val:.1f} MPa\n{yield_strain_val:.2f}% strain',
                           xy=(yield_strain_val, yield_stress_val), 
                           xytext=(yield_strain_val + 0.5, yield_stress_val + 50),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                           fontsize=9, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))
            
            ax1.set_xlabel('Strain (%)', fontsize=11)
            ax1.set_ylabel('Stress (MPa)', fontsize=11)
            ax1.set_title('Resilience (Energy to Yield)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=9)
            
            # Toughness plot (total area under curve)
            ax2.plot(analyzer.strain * 100, analyzer.stress, 'b-', linewidth=2, label='Stress-Strain Curve')
            ax2.fill_between(analyzer.strain * 100, analyzer.stress, alpha=0.4, color='orange',
                           label=f'Toughness: {analyzer.results.get("toughness", 0):.2f} MJ/m³')
            
            # Mark key points on toughness plot
            if 'yield_strength' in analyzer.results:
                yield_stress_val = analyzer.results['yield_strength']
                yield_strain_val = analyzer.results['yield_strain'] * 100
                ax2.plot(yield_strain_val, yield_stress_val, 'ro', markersize=6, label='Yield Point')
            
            if 'ultimate_tensile_strength' in analyzer.results:
                uts = analyzer.results['ultimate_tensile_strength']
                uts_strain = analyzer.results['strain_at_uts'] * 100
                ax2.plot(uts_strain, uts, 'go', markersize=6, label='UTS')
            
            # Mark fracture point
            fracture_stress = analyzer.stress[-1]
            fracture_strain = analyzer.strain[-1] * 100
            ax2.plot(fracture_strain, fracture_stress, 'mo', markersize=6, label='Fracture Point')
            
            ax2.set_xlabel('Strain (%)', fontsize=11)
            ax2.set_ylabel('Stress (MPa)', fontsize=11)
            ax2.set_title('Toughness (Total Energy)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=9)
            
            # Add summary text box
            if analyzer.results:
                resilience = analyzer.results.get('resilience', 0)
                toughness = analyzer.results.get('toughness', 0)
                textstr = f"""Energy Properties Summary:
Resilience: {resilience:.2f} MJ/m³
Toughness: {toughness:.2f} MJ/m³
Ratio (T/R): {toughness/resilience:.1f}""" if resilience > 0 else f"""Energy Properties Summary:
Resilience: {resilience:.2f} MJ/m³
Toughness: {toughness:.2f} MJ/m³"""
                
                props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
                fig.text(0.02, 0.98, textstr, fontsize=10, verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            self.display_plot(fig)
    
    def clear_plot(self):
        """Clear the current plot"""
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None
    
    def display_plot(self, fig):
        """Display a matplotlib figure in the plot frame"""
        self.plot_canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_results(self):
        """Show results for all specimens in tabular format with properties on left and specimens on top"""
        self.clear_plot()
        
        # Create a new window for the tabular results view
        results_window = tk.Toplevel(self.root)
        results_window.title("Analysis Results - Tabular View")
        results_window.geometry("1000x700")
        results_window.transient(self.root)
        
        if not self.analyzers and not self.analyzer:
            tk.Label(results_window, text="No analysis results available. Please import and analyze data first.", 
                    font=("Arial", 12)).pack(pady=50)
            return
        
        # Create main frame with scrollbars
        main_frame = ttk.Frame(results_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview for tabular display
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Prepare data for table
        if self.analyzers:
            # Multiple specimens
            analyzers = self.analyzers
            specimen_names = self.specimen_names
        else:
            # Single specimen
            analyzers = [self.analyzer]
            specimen_names = ["Specimen 1"]
        
        # Define columns: Property + all specimen names
        columns = ["Property"] + specimen_names
        
        # Create treeview
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
        
        # Configure column headings
        for col in columns:
            tree.heading(col, text=col)
            if col == "Property":
                tree.column(col, width=200, anchor='w')
            else:
                tree.column(col, width=120, anchor='center')
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Define properties to display with their units and formatting
        properties = [
            ("Young's Modulus (GPa)", 'young_modulus', lambda x: f"{x/1000:.1f}" if x else "N/A"),
            ("Elastic Modulus (GPa)", 'elastic_modulus', lambda x: f"{x/1000:.1f}" if x else "N/A"),
            ("Yield Strength (MPa)", 'yield_strength', lambda x: f"{x:.1f}" if x else "N/A"),
            ("Ultimate Tensile Strength (MPa)", 'ultimate_strength', lambda x: f"{x:.1f}" if x else "N/A"),
            ("Fracture Strength (MPa)", 'fracture_strength', lambda x: f"{x:.1f}" if x else "N/A"),
            ("Proportional Limit (MPa)", 'proportional_limit', lambda x: f"{x:.1f}" if x else "N/A"),
            ("Elongation at Fracture (%)", 'elongation_at_fracture', lambda x: f"{x*100:.2f}" if x else "N/A"),
            ("Reduction of Area (%)", 'reduction_of_area', lambda x: f"{x*100:.2f}" if x else "N/A"),
            ("Toughness (MJ/m³)", 'toughness', lambda x: f"{x/1e6:.2f}" if x else "N/A"),
            ("Resilience (MJ/m³)", 'resilience', lambda x: f"{x/1e6:.2f}" if x else "N/A"),
            ("Strain Hardening Exponent", 'strain_hardening_exponent', lambda x: f"{x:.3f}" if x else "N/A"),
            ("Strength Coefficient (MPa)", 'strength_coefficient', lambda x: f"{x:.1f}" if x else "N/A"),
            ("Yield Strain (%)", 'yield_strain', lambda x: f"{x*100:.3f}" if x else "N/A"),
            ("Strain at UTS (%)", 'strain_at_uts', lambda x: f"{x*100:.3f}" if x else "N/A"),
            ("True Fracture Stress (MPa)", 'true_fracture_stress', lambda x: f"{x:.1f}" if x else "N/A"),
            ("True Fracture Strain", 'true_fracture_strain', lambda x: f"{x:.3f}" if x else "N/A"),
            ("Modulus R² Value", 'elastic_modulus_r_squared', lambda x: f"{x:.4f}" if x else "N/A"),
            ("Maximum Load (N)", 'maximum_load', lambda x: f"{x:.0f}" if x else "N/A"),
            ("Load at Yield (N)", 'load_at_yield', lambda x: f"{x:.0f}" if x else "N/A"),
        ]
        
        # Add specimen parameters section
        param_properties = [
            ("", "", lambda x: ""),  # Spacer
            ("SPECIMEN PARAMETERS", "", lambda x: ""),  # Header
            ("Specimen Type", 'specimen_type', lambda x: x if x else "N/A"),
            ("Original Length (mm)", 'original_length', lambda x: f"{x:.2f}" if x else "N/A"),
            ("Original Diameter (mm)", 'original_diameter', lambda x: f"{x:.2f}" if x else "N/A"),
            ("Original Width (mm)", 'original_width', lambda x: f"{x:.2f}" if x else "N/A"),
            ("Original Thickness (mm)", 'original_thickness', lambda x: f"{x:.2f}" if x else "N/A"),
            ("Cross-Sectional Area (mm²)", 'cross_sectional_area', lambda x: f"{x:.2f}" if x else "N/A"),
            ("Final Length (mm)", 'final_length', lambda x: f"{x:.2f}" if x else "N/A"),
            ("Final Diameter (mm)", 'final_diameter', lambda x: f"{x:.2f}" if x else "N/A"),
            ("Final Width (mm)", 'final_width', lambda x: f"{x:.2f}" if x else "N/A"),
            ("Final Thickness (mm)", 'final_thickness', lambda x: f"{x:.2f}" if x else "N/A"),
        ]
        
        # Combine all properties
        all_properties = properties + param_properties
        
        # Populate the table
        for prop_name, prop_key, formatter in all_properties:
            row_data = [prop_name]
            
            # Handle section headers and spacers
            if prop_key == "":
                if prop_name == "SPECIMEN PARAMETERS":
                    # Insert section header
                    row_data.extend([""] * len(specimen_names))
                    tree.insert("", tk.END, values=row_data, tags=("header",))
                elif prop_name == "":
                    # Insert spacer
                    row_data.extend([""] * len(specimen_names))
                    tree.insert("", tk.END, values=row_data)
                continue
            
            # Get values for each specimen
            for analyzer in analyzers:
                if prop_key in ['specimen_type', 'original_length', 'original_diameter', 
                               'original_width', 'original_thickness', 'final_length', 
                               'final_diameter', 'final_width', 'final_thickness']:
                    # Get from specimen parameters
                    value = analyzer.specimen_params.get(prop_key)
                    if prop_key == 'cross_sectional_area':
                        value = analyzer.A0
                elif prop_key in ['maximum_load', 'load_at_yield']:
                    # Calculate special properties
                    if prop_key == 'maximum_load':
                        value = np.max(analyzer.load) if hasattr(analyzer, 'load') else None
                    elif prop_key == 'load_at_yield':
                        yield_strain = analyzer.results.get('yield_strain')
                        if yield_strain and hasattr(analyzer, 'load') and hasattr(analyzer, 'strain'):
                            yield_idx = np.argmin(np.abs(analyzer.strain - yield_strain))
                            value = analyzer.load[yield_idx]
                        else:
                            value = None
                else:
                    # Get from results
                    value = analyzer.results.get(prop_key)
                
                formatted_value = formatter(value)
                row_data.append(formatted_value)
            
            tree.insert("", tk.END, values=row_data)
        
        # Configure tag styles
        tree.tag_configure("header", background="lightblue", font=("Arial", 10, "bold"))
        
        # Add export button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def export_table():
            """Export the table data to CSV"""
            file_path = filedialog.asksaveasfilename(
                title="Save Table Data",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                try:
                    # Prepare data for CSV export
                    csv_data = []
                    
                    # Header row
                    csv_data.append(columns)
                    
                    # Data rows
                    for prop_name, prop_key, formatter in all_properties:
                        if prop_key == "" and prop_name != "SPECIMEN PARAMETERS":
                            continue  # Skip spacers
                            
                        row_data = [prop_name]
                        
                        if prop_name == "SPECIMEN PARAMETERS":
                            row_data.extend([""] * len(specimen_names))
                            csv_data.append(row_data)
                            continue
                        
                        # Get values for each specimen
                        for analyzer in analyzers:
                            if prop_key in ['specimen_type', 'original_length', 'original_diameter', 
                                           'original_width', 'original_thickness', 'final_length', 
                                           'final_diameter', 'final_width', 'final_thickness']:
                                value = analyzer.specimen_params.get(prop_key)
                                if prop_key == 'cross_sectional_area':
                                    value = analyzer.A0
                            elif prop_key in ['maximum_load', 'load_at_yield']:
                                if prop_key == 'maximum_load':
                                    value = np.max(analyzer.load) if hasattr(analyzer, 'load') else None
                                elif prop_key == 'load_at_yield':
                                    yield_strain = analyzer.results.get('yield_strain')
                                    if yield_strain and hasattr(analyzer, 'load') and hasattr(analyzer, 'strain'):
                                        yield_idx = np.argmin(np.abs(analyzer.strain - yield_strain))
                                        value = analyzer.load[yield_idx]
                                    else:
                                        value = None
                            else:
                                value = analyzer.results.get(prop_key)
                            
                            formatted_value = formatter(value)
                            row_data.append(formatted_value)
                        
                        csv_data.append(row_data)
                    
                    # Write to CSV
                    import csv
                    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(csv_data)
                    
                    messagebox.showinfo("Success", f"Table data exported to {file_path}")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export table data: {str(e)}")
        
        ttk.Button(button_frame, text="Export to CSV", command=export_table).pack(side=tk.LEFT, padx=(0, 10))
        
        # Summary label
        summary_text = f"Displaying results for {len(analyzers)} specimen(s)"
        ttk.Label(button_frame, text=summary_text, font=("Arial", 10)).pack(side=tk.RIGHT)
    
    def export_results(self):
        """Export results to text file"""
        if not self.analyzer and not self.analyzers:
            messagebox.showwarning("No Data", "No analysis results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    if self.analyzers:
                        # Export all specimens
                        f.write(f"MULTI-SPECIMEN TENSILE TEST ANALYSIS RESULTS\n")
                        f.write(f"{'='*60}\n\n")
                        f.write(f"Total Specimens Analyzed: {len(self.analyzers)}\n\n")
                        
                        # Summary table
                        f.write(f"SUMMARY TABLE\n")
                        f.write(f"{'-'*120}\n")
                        f.write(f"{'Specimen':<15} {'E (GPa)':<10} {'Yield (MPa)':<12} {'UTS (MPa)':<12} {'Elongation (%)':<15} {'Toughness (MJ/m³)':<18}\n")
                        f.write(f"{'-'*120}\n")
                        
                        for i, analyzer in enumerate(self.analyzers):
                            name = self.specimen_names[i] if i < len(self.specimen_names) else f"Specimen {i+1}"
                            E = analyzer.results.get('young_modulus', 0) / 1000  # Convert to GPa
                            yield_strength = analyzer.results.get('yield_strength', 0)
                            uts = analyzer.results.get('ultimate_strength', 0)
                            elongation = analyzer.results.get('elongation_at_fracture', 0) * 100  # Convert to %
                            toughness = analyzer.results.get('toughness', 0) / 1e6  # Convert to MJ/m³
                            
                            f.write(f"{name:<15} {E:<10.1f} {yield_strength:<12.1f} {uts:<12.1f} {elongation:<15.2f} {toughness:<18.2f}\n")
                        
                        f.write(f"{'-'*120}\n\n")
                        
                        # Detailed results for each specimen
                        for i, analyzer in enumerate(self.analyzers):
                            name = self.specimen_names[i] if i < len(self.specimen_names) else f"Specimen {i+1}"
                            f.write(f"DETAILED RESULTS - {name.upper()}\n")
                            f.write(f"{'='*50}\n")
                            f.write(analyzer.format_results())
                            f.write(f"\n{'='*50}\n\n")
                    else:
                        # Export single specimen
                        f.write(self.analyzer.format_results())
                
                specimen_count = len(self.analyzers) if self.analyzers else 1
                messagebox.showinfo("Success", f"Results for {specimen_count} specimen(s) exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def show_formulae(self):
        """Show formulae used in analysis"""
        formulae_text = """TENSILE TEST ANALYSIS FORMULAE

Basic Equations:
- Engineering Stress: σ = F / A0
- Engineering Strain: ε = ΔL / L0
- True Stress: σt = σ(1 + ε)
- True Strain: εt = ln(1 + ε)

ASTM E111 Elastic Modulus:
- Uses middle 50% of linear elastic region
- Linear regression: E = Δσ / Δε
- Chord Modulus: E = (σ75 - σ25) / (ε75 - ε25)
- Tangent Modulus: Local derivative at midpoint

Mechanical Properties:
- Yield Strength: 0.2% offset method
- Ultimate Tensile Strength: Maximum stress
- Proportional Limit: End of linear region (R² analysis)
- Elongation at Fracture: (Lf - L0) / L0 × 100%
- Reduction of Area: (A0 - Af) / A0 × 100%

Energy Properties:
- Resilience: ∫[0 to εy] σ dε (area to yield)
- Toughness: ∫[0 to εf] σ dε (total area)

Ramberg-Osgood Model:
- ε = σ/E + (σ/K)^n
- Where K = strength coefficient, n = strain hardening exponent"""
        
        # Create a new window for formulae
        formulae_window = tk.Toplevel(self.root)
        formulae_window.title("Analysis Formulae")
        formulae_window.geometry("600x500")
        
        text_widget = tk.Text(formulae_window, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(formulae_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.insert("1.0", formulae_text)
        text_widget.config(state="disabled")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """Tensile Test Analyzer v0.17.1
Enhanced Interface with Fully Annotated Plotting

Key Features:
- ASTM E8/E8M and E111 compliant analysis
- Enhanced modulus calculation using middle 50% of linear elastic region
- Fully annotated stress-strain curves with key points and values
- 0.2% offset line visualization for yield strength determination
- True stress-strain curve plotting with annotations
- Ramberg-Osgood parameter fitting and visualization with key points
- Enhanced energy analysis plotting with annotated regions
- Robust handling of missing final dimensions
- Load-extension and stress-strain data support

Analysis Methods:
- Young's, Chord, and Tangent Modulus calculation
- Strain gauge data support for enhanced accuracy  
- Automatic proportional limit detection
- Advanced error handling and data validation

Enhanced Plotting Capabilities:
- Annotated Stress-Strain: Complete visualization with all key points marked
- Interactive annotations showing exact values and coordinates
- Color-coded markers: Yield (red), UTS (green), Fracture (magenta), Proportional limit (cyan)
- 0.2% offset line clearly displayed for yield determination
- Linear elastic region highlighting
- Property summary boxes on all plots
- Energy analysis with filled regions showing resilience and toughness
- Ramberg-Osgood fit with annotated key points

Visual Features:
- Professional-quality plots with clear annotations
- Coordinated color scheme across all plots
- Comprehensive legends and labels
- Property value callouts with connecting arrows
- Transparent overlays for better readability

Data Handling:
- Handles missing final dimensions gracefully
- Supports both round and flat specimens
- Optional strain gauge data integration
- Comprehensive results export
- Multi-specimen file detection and selection

Duncan, 2025"""
        messagebox.showinfo("About", about_text)
    
    def is_qb2_style_file(self, data):
        """Check if this is a QB2-style file with specimen parameters in header"""
        # Look for specific QB2 indicators
        first_col_values = data.iloc[:5, 0].astype(str).str.lower()
        qb2_indicators = ['specimen', 'area', 'gauge length', 'load']
        
        for indicator in qb2_indicators:
            if any(indicator in val for val in first_col_values):
                return True
        return False
    
    def parse_qb2_style_file(self, data):
        """Parse QB2-style file format"""
        specimens = []
        
        # Extract specimen parameters from header
        params = {
            'specimen_type': 'round',  # Default
            'original_length': 25.0,   # Default gauge length in mm
            'original_diameter': 6.0,  # Default diameter in mm
            'original_width': 12.5,    # Default width for flat specimens
            'original_thickness': 3.0, # Default thickness for flat specimens
            'final_length': None,
            'final_diameter': None,
            'final_width': None,
            'final_thickness': None
        }
        
        # Look for specimen parameters in first few rows
        for i in range(min(10, len(data))):
            row = data.iloc[i]
            first_cell = str(row.iloc[0]).lower()
            
            if 'gauge length' in first_cell and len(row) > 1:
                try:
                    params['original_length'] = float(row.iloc[1])
                except (ValueError, TypeError):
                    pass
            
            elif 'area' in first_cell and len(row) > 1:
                try:
                    area_cm2 = float(row.iloc[1])
                    # Convert cm² to mm² and calculate diameter
                    area_mm2 = area_cm2 * 100
                    diameter = 2 * np.sqrt(area_mm2 / np.pi)
                    params['original_diameter'] = diameter
                except (ValueError, TypeError):
                    pass
        
        # Find data start row
        data_start_row = None
        for i in range(len(data)):
            if str(data.iloc[i, 0]).lower() in ['load', '(n)']:
                data_start_row = i + 1
                break
        
        if data_start_row and data_start_row < len(data):
            # Extract load and extension data
            numeric_data = data.iloc[data_start_row:].apply(pd.to_numeric, errors='coerce')
            
            if len(numeric_data.columns) >= 4:  # Load, Stress, Strain, Extension
                load_data = numeric_data.iloc[:, 0].dropna().values
                extension_data = numeric_data.iloc[:, 3].dropna().values  # Extension is usually 4th column
                
                # Check for strain gauge data (usually in column 2 or 6)
                strain_gauge_data = None
                for col_idx in [2, 6]:
                    if len(numeric_data.columns) > col_idx:
                        strain_candidate = numeric_data.iloc[:, col_idx].dropna().values
                        if len(strain_candidate) == len(load_data):
                            strain_gauge_data = strain_candidate
                            break
                
                specimen_info = {
                    'load': load_data,
                    'extension': extension_data,
                    'strain_gauge': strain_gauge_data,
                    'params': params,
                    'name': "QB2 Specimen"
                }
                specimens.append(specimen_info)
        
        return specimens
    
    def find_data_start_row(self, data):
        """Find where numeric data starts"""
        for i in range(min(20, len(data))):
            row = data.iloc[i]
            # Check if this row has mostly numeric data
            numeric_count = 0
            for val in row:
                try:
                    float(val)
                    numeric_count += 1
                except:
                    pass
            
            if numeric_count >= 2:  # At least 2 numeric columns
                return i
        
        return None
    
    def extract_specimen_params_from_file(self, data):
        """Extract specimen parameters from file headers or metadata"""
        params = {
            'specimen_type': 'round',
            'original_length': 25.4,   # Default gauge length in mm
            'original_diameter': 6.0,  # Default diameter in mm
            'original_width': 12.5,    # Default width for flat specimens
            'original_thickness': 3.0, # Default thickness for flat specimens
            'final_length': None,
            'final_diameter': None,
            'final_width': None,
            'final_thickness': None
        }
        
        # Search for dimension keywords in first 20 rows
        for i in range(min(20, len(data))):
            for j in range(min(10, len(data.columns))):
                try:
                    cell_value = str(data.iloc[i, j]).lower()
                    next_cell = data.iloc[i, j+1] if j+1 < len(data.columns) else None
                    
                    if 'diameter' in cell_value and next_cell is not None:
                        try:
                            params['original_diameter'] = float(next_cell)
                        except (ValueError, TypeError):
                            pass
                    
                    elif 'length' in cell_value and 'gauge' in cell_value and next_cell is not None:
                        try:
                            params['original_length'] = float(next_cell)
                        except (ValueError, TypeError):
                            pass
                    
                    elif 'width' in cell_value and next_cell is not None:
                        try:
                            params['original_width'] = float(next_cell)
                        except (ValueError, TypeError):
                            pass
                    
                    elif 'thickness' in cell_value and next_cell is not None:
                        try:
                            params['original_thickness'] = float(next_cell)
                        except (ValueError, TypeError):
                            pass
                
                except (ValueError, TypeError, IndexError):
                    continue
        
        return params
    
    def detect_multiple_specimens_stress_strain(self, file_path):
        """
        Detect multiple specimens in stress-strain data files and extract parameters
        
        Returns:
        --------
        list: List of specimen dictionaries with stress, strain, params, and name
        """
        import pandas as pd
        
        try:
            # Read the file
            if file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            else:
                data = pd.read_csv(file_path)
            
            specimens = []
            
            # Check if this looks like a multi-specimen stress-strain file
            # Look for pattern: Stress, Strain, param1, param2, param3, Stress, Strain, param1, param2, param3, etc.
            if len(data.columns) >= 6:  # At least 2 specimens worth of data
                num_specimens = 0
                col_index = 0
                
                while col_index + 1 < len(data.columns):
                    # Look for stress-strain pair
                    stress_col = data.columns[col_index]
                    strain_col = data.columns[col_index + 1]
                    
                    # Check if columns contain "stress" and "strain" keywords (case insensitive)
                    if ('stress' in str(stress_col).lower() and 'strain' in str(strain_col).lower()) or num_specimens == 0:
                        # Extract data for this specimen
                        stress_data = data.iloc[:, col_index].dropna()
                        strain_data = data.iloc[:, col_index + 1].dropna()
                        
                        if len(stress_data) > 10 and len(strain_data) > 10:  # Reasonable amount of data
                            # Look for embedded parameters in the next few columns or the data itself
                            params = self.extract_specimen_params_from_stress_strain_data(data, col_index)
                            
                            specimens.append({
                                'stress': stress_data.values,
                                'strain': strain_data.values,
                                'params': params,
                                'name': f'Specimen {num_specimens + 1}'
                            })
                            num_specimens += 1
                        
                        # Move to next potential specimen (usually 6 columns apart)
                        # Pattern: Stress, Strain, param1, param2, param3, spacer
                        col_index += 6
                    else:
                        col_index += 1
                
                if num_specimens > 0:
                    return specimens
            
            # If no multi-specimen pattern detected, try simple two-column format
            if len(data.columns) >= 2:
                # Try to use first two columns as stress and strain
                stress_data = data.iloc[:, 0].dropna()
                strain_data = data.iloc[:, 1].dropna()
                
                if len(stress_data) > 10 and len(strain_data) > 10:  # Reasonable amount of data
                    # Use default parameters since we can't extract from simple format
                    params = {
                        'specimen_type': 'round',
                        'original_length': 25.4,   # Default gauge length in mm
                        'original_diameter': 6.0,  # Default diameter in mm
                        'original_width': 12.5,    # Default width for flat specimens
                        'original_thickness': 3.0, # Default thickness for flat specimens
                        'final_length': None,
                        'final_diameter': None,
                        'final_width': None,
                        'final_thickness': None
                    }
                    
                    specimens.append({
                        'stress': stress_data.values,
                        'strain': strain_data.values,
                        'params': params,
                        'name': 'Specimen 1'
                    })
                    return specimens
            
            # If still no valid data found, return empty list
            return []
            
        except Exception as e:
            print(f"Error detecting specimens in stress-strain file: {e}")
            return []
    
    def extract_specimen_params_from_stress_strain_data(self, data, stress_col_index):
        """
        Extract specimen parameters from stress-strain data file
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data frame containing the file data
        stress_col_index : int
            Column index where stress data starts for this specimen
            
        Returns:
        --------
        dict: Dictionary with specimen parameters
        """
        params = {
            'specimen_type': 'round',  # Default assumption
            'original_length': 25.4,   # Default gauge length in mm
            'original_diameter': 6.0,  # Default diameter in mm
            'original_width': 12.5,    # Default width for flat specimens
            'original_thickness': 3.0, # Default thickness for flat specimens
            'final_length': None,
            'final_diameter': None,
            'final_width': None,
            'final_thickness': None
        }
        
        try:
            # Look for parameters in columns adjacent to stress-strain data
            # Common pattern: Stress, Strain, diameter, empty, gauge_length, empty
            if stress_col_index + 4 < len(data.columns):
                # Check if there are numeric values in parameter columns
                param_col_1 = stress_col_index + 2  # Usually diameter
                param_col_2 = stress_col_index + 4  # Usually gauge length
                
                # Get first non-null value from these columns
                param1_val = None
                param2_val = None
                
                for idx in range(min(10, len(data))):  # Check first 10 rows
                    if pd.notna(data.iloc[idx, param_col_1]):
                        try:
                            param1_val = float(data.iloc[idx, param_col_1])
                            break
                        except (ValueError, TypeError):
                            continue
                
                for idx in range(min(10, len(data))):
                    if pd.notna(data.iloc[idx, param_col_2]):
                        try:
                            param2_val = float(data.iloc[idx, param_col_2])
                            break
                        except (ValueError, TypeError):
                            continue
                
                # Assign parameters based on typical values
                # Diameter typically 3-20mm, gauge length typically 10-100mm
                if param1_val is not None and 2 <= param1_val <= 20:
                    params['original_diameter'] = param1_val
                if param2_val is not None and 10 <= param2_val <= 100:
                    params['original_length'] = param2_val
                    
                # If values are swapped, try to correct
                if param2_val is not None and 2 <= param2_val <= 20 and param1_val is not None and 10 <= param1_val <= 100:
                    params['original_diameter'] = param2_val
                    params['original_length'] = param1_val
                    
        except Exception as e:
            print(f"Warning: Could not extract parameters from stress-strain data: {e}")
        
        return params

    def select_specimen_dialog(self, specimen_names_or_data, file_path=None, data_type=None):
        """
        Dialog for selecting a specimen when multiple are detected
        
        Parameters:
        -----------
        specimen_names_or_data : list
            Either a list of specimen names or a list of specimen data dictionaries
        file_path : str, optional
            Path to the file being processed
        data_type : str, optional
            Type of data being processed ('load-extension' or 'stress-strain')
            
        Returns:
        --------
        int or None: Index of selected specimen, or None if cancelled
        """
        
        # Handle both formats: list of names or list of data dictionaries
        if isinstance(specimen_names_or_data[0], dict):
            # List of specimen data dictionaries
            specimen_names = [spec['name'] for spec in specimen_names_or_data]
            specimen_data = specimen_names_or_data
        else:
            # List of specimen names
            specimen_names = specimen_names_or_data
            specimen_data = None
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Specimen")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 150, self.root.winfo_rooty() + 150))
        
        # Header
        header_text = f"Multiple specimens detected in the data file."
        if file_path:
            header_text += f"\nFile: {os.path.basename(file_path)}"
        if data_type:
            header_text += f"\nData type: {data_type}"
        
        ttk.Label(dialog, text=header_text, font=("Arial", 10)).pack(pady=10)
        ttk.Label(dialog, text="Please select which specimen to analyze:", font=("Arial", 10, "bold")).pack(pady=5)
        
        # Specimen selection frame
        selection_frame = ttk.Frame(dialog)
        selection_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Listbox for specimen selection
        listbox = tk.Listbox(selection_frame, height=6)
        listbox.pack(fill=tk.BOTH, expand=True)
        
        # Populate listbox with specimen information
        for i, name in enumerate(specimen_names):
            display_text = name
            if specimen_data:
                spec = specimen_data[i]
                if 'stress' in spec:
                    display_text += f" ({len(spec['stress'])} data points)"
                elif 'load' in spec:
                    display_text += f" ({len(spec['load'])} data points)"
                    
                # Add parameter info if available
                if 'params' in spec:
                    params = spec['params']
                    if 'original_diameter' in params:
                        display_text += f" - Ø{params['original_diameter']:.1f}mm"
                    if 'original_length' in params:
                        display_text += f", L0={params['original_length']:.1f}mm"
            
            listbox.insert(tk.END, display_text)
        
        # Select first item by default
        listbox.selection_set(0)
        
        result = {"selection": None}
        
        def confirm():
            selection = listbox.curselection()
            if selection:
                result["selection"] = selection[0]
                
                # If this is for load-extension data, process the selected specimen
                if specimen_data and data_type == "load-extension":
                    try:
                        selected_spec = specimen_data[result["selection"]]
                        
                        # Create analyzer and run analysis
                        self.analyzer = TensileTestAnalyzer(
                            selected_spec['load'], 
                            selected_spec['extension'], 
                            selected_spec['params'], 
                            selected_spec.get('strain_gauge', None)
                        )
                        self.analyzer.analyze()
                        
                        self.update_results_display()
                        self.enable_menus()
                        
                        filename = os.path.basename(file_path) if file_path else "data file"
                        success_msg = f"Successfully analyzed {data_type} data from {filename}\nSpecimen: {selected_spec['name']}"
                        if selected_spec.get('strain_gauge') is not None:
                            success_msg += "\nStrain gauge data included for enhanced modulus calculation"
                        
                        messagebox.showinfo("Success", success_msg)
                        self.status_label.config(text=f"Analyzed: {filename} - {selected_spec['name']}")
                        
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to analyze selected specimen: {str(e)}")
                        result["selection"] = None
                
                dialog.destroy()
            else:
                messagebox.showwarning("No Selection", "Please select a specimen to analyze.")
        
        def cancel():
            result["selection"] = None
            dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Analyze Selected", command=confirm).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=10)
        
        dialog.wait_window()
        
        return result["selection"]

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedTensileGUI(root)
    root.mainloop()
