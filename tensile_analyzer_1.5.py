"""
Tensile Analyzer Program
Compliant with ASTM E8 and E112 standards

Author: Duncan W. Gibbons, Ph.D.
Date: August 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
from scipy import optimize, integrate
from scipy.signal import savgol_filter
import warnings


# ============================================================================
# DATA LOADER MODULE
# ============================================================================

class DataLoader:
    """Class to handle loading tensile test data from various file formats"""
    
    def __init__(self):
        # Required columns (1-6 must contain data)
        self.required_columns = [
            'time_s',
            'displacement_mm', 
            'force_kN',
            'strain_mm_mm',
            'original_gauge_length_mm',
            'original_diameter_mm'
        ]
        
        # Optional columns (7-8) - may contain limited data or be missing entirely
        self.optional_columns = [
            'final_gauge_length_mm',
            'final_diameter_mm'
        ]
        
        # All possible columns in order
        self.all_columns = self.required_columns + self.optional_columns
        
        # Minimum required columns (columns 1-6 are required)
        self.min_required_columns = 6
    
    def load_data(self, file_path):
        """
        Load tensile test data from file
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
            
        Returns:
        --------
        pandas.DataFrame
            Loaded and validated data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                data = pd.read_csv(file_path, header=0)  # First row is header
            elif file_ext in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path, header=0)  # First row is header
            elif file_ext == '.txt':
                # Try different delimiters for text files
                for delimiter in ['\t', ',', ' ', ';']:
                    try:
                        data = pd.read_csv(file_path, delimiter=delimiter, header=0)  # First row is header
                        if len(data.columns) >= self.min_required_columns:  # Should have at least 6 columns
                            break
                    except:
                        continue
                else:
                    raise ValueError("Could not parse text file with any common delimiter")
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
        
        # Validate and clean data
        data = self._validate_and_clean_data(data)
        
        return data
    
    def _validate_and_clean_data(self, data):
        """
        Validate and clean the loaded data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw loaded data
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned and validated data
        """
        # Remove columns that are completely empty
        data = data.dropna(axis=1, how='all')
        
        # Check if we have enough required columns
        if len(data.columns) < self.min_required_columns:
            available_cols = len(data.columns)
            error_msg = f"Insufficient data columns. Found {available_cols} columns, need at least {self.min_required_columns}.\n"
            error_msg += "Required columns (1-6): Time, Displacement, Force, Strain, Original Length, Original Diameter\n"
            error_msg += "Optional columns (7-8): Final Length, Final Diameter"
            raise ValueError(error_msg)
        
        # Determine how many columns we can use
        num_cols = min(len(data.columns), len(self.all_columns))
        data = data.iloc[:, :num_cols].copy()
        
        # Assign column names based on available columns
        column_names = self.all_columns[:num_cols]
        data.columns = column_names
        
        # Check which columns have sufficient data
        column_status = {}
        
        # Check time-series columns (1-4) - must have data for most timesteps
        time_series_cols = ['time_s', 'displacement_mm', 'force_kN', 'strain_mm_mm']
        for col in time_series_cols:
            if col in data.columns:
                non_null_count = data[col].notna().sum()
                total_count = len(data)
                
                if non_null_count == 0:
                    column_status[col] = 'empty'
                    print(f"ERROR: Required time-series column '{col}' is empty!")
                    raise ValueError(f"Required time-series column '{col}' contains no data")
                elif non_null_count < total_count * 0.9:  # Less than 90% of timesteps
                    column_status[col] = 'sparse'
                    print(f"WARNING: Time-series column '{col}' has sparse data ({non_null_count}/{total_count} points)")
                else:
                    column_status[col] = 'adequate'
            else:
                column_status[col] = 'missing'
                print(f"ERROR: Required time-series column '{col}' not found!")
                raise ValueError(f"Required time-series column '{col}' is missing from data file")
        
        # Check dimension columns (5-6) - only need a few valid measurements
        dimension_cols = ['original_gauge_length_mm', 'original_diameter_mm']
        for col in dimension_cols:
            if col in data.columns:
                non_null_count = data[col].notna().sum()
                
                if non_null_count == 0:
                    column_status[col] = 'empty'
                    print(f"ERROR: Required dimension column '{col}' is empty!")
                    raise ValueError(f"Required dimension column '{col}' contains no data")
                elif non_null_count < 1:  # Need at least 1 measurement
                    column_status[col] = 'insufficient'
                    print(f"ERROR: Dimension column '{col}' has no valid measurements!")
                    raise ValueError(f"Dimension column '{col}' needs at least one valid measurement")
                else:
                    column_status[col] = 'adequate'
                    if non_null_count < 10:  # Few measurements is normal for dimensions
                        print(f"Note: Dimension column '{col}' has {non_null_count} measurement(s) - this is normal")
            else:
                column_status[col] = 'missing'
                print(f"ERROR: Required dimension column '{col}' not found!")
                raise ValueError(f"Required dimension column '{col}' is missing from data file")
        
        # Check optional final measurement columns (7-8) - may have limited or no data
        final_measurement_cols = ['final_gauge_length_mm', 'final_diameter_mm']
        for col in final_measurement_cols:
            if col in data.columns:
                non_null_count = data[col].notna().sum()
                
                if non_null_count == 0:
                    column_status[col] = 'empty'
                    print(f"Note: Optional final measurement column '{col}' is empty - related calculations will be skipped")
                elif non_null_count < 1:
                    column_status[col] = 'insufficient'
                    print(f"Note: Optional final measurement column '{col}' has no valid measurements - related calculations will be skipped")
                else:
                    column_status[col] = 'adequate'
                    print(f"Note: Final measurement column '{col}' has {non_null_count} measurement(s)")
            else:
                column_status[col] = 'missing'
                print(f"Note: Optional final measurement column '{col}' not found - related calculations will be skipped")
        
        # Store column status for later use
        data.attrs['column_status'] = column_status
        
        # Convert time-series columns (1-4) to numeric
        time_series_cols = ['time_s', 'displacement_mm', 'force_kN', 'strain_mm_mm']
        for col in time_series_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Convert dimension columns (5-6) to numeric - these may have few values
        dimension_cols = ['original_gauge_length_mm', 'original_diameter_mm']
        for col in dimension_cols:
            if col in data.columns and column_status.get(col) == 'adequate':
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Average multiple measurements if present
                non_null_values = data[col].dropna()
                if len(non_null_values) > 1:
                    avg_value = non_null_values.mean()
                    print(f"Note: Averaged {len(non_null_values)} measurements in '{col}' to {avg_value:.3f}")
                    data[col] = avg_value
                elif len(non_null_values) == 1:
                    # Single measurement - use it for all rows
                    single_value = non_null_values.iloc[0]
                    data[col] = single_value
                    print(f"Note: Using single measurement {single_value:.3f} for '{col}'")
        
        # Convert optional final measurement columns (7-8) to numeric
        final_cols = ['final_gauge_length_mm', 'final_diameter_mm']
        for col in final_cols:
            if col in data.columns and column_status.get(col) == 'adequate':
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Average multiple measurements if present
                non_null_values = data[col].dropna()
                if len(non_null_values) > 1:
                    avg_value = non_null_values.mean()
                    print(f"Note: Averaged {len(non_null_values)} measurements in '{col}' to {avg_value:.3f}")
                    data[col] = avg_value
                elif len(non_null_values) == 1:
                    # Single measurement - use it for all rows
                    single_value = non_null_values.iloc[0]
                    data[col] = single_value
                    print(f"Note: Using single measurement {single_value:.3f} for '{col}'")
        
        # Handle special case: estimate final_diameter if missing but other data available
        if ('final_gauge_length_mm' in data.columns and 
            'original_diameter_mm' in data.columns and
            column_status.get('final_gauge_length_mm') == 'adequate' and
            column_status.get('original_diameter_mm') == 'adequate' and
            column_status.get('final_diameter_mm') in ['empty', 'missing', 'insufficient']):
            
            # Get the final gauge length value
            final_length = data['final_gauge_length_mm'].dropna().iloc[-1] if not data['final_gauge_length_mm'].dropna().empty else None
            original_length = data['original_gauge_length_mm'].iloc[0]
            original_diameter = data['original_diameter_mm'].iloc[0]
            
            if final_length is not None and original_length is not None and original_diameter is not None:
                # Estimate final diameter using volume conservation
                estimated_final_diameter = original_diameter * np.sqrt(original_length / final_length)
                data['final_diameter_mm'] = np.nan
                data.loc[data.index[-1], 'final_diameter_mm'] = estimated_final_diameter
                column_status['final_diameter_mm'] = 'estimated'
                print(f"Final diameter estimated as {estimated_final_diameter:.3f} mm using volume conservation")
        
        # Remove rows where ALL time-series columns have NaN values
        initial_rows = len(data)
        time_series_cols_present = [col for col in time_series_cols if col in data.columns]
        data = data.dropna(subset=time_series_cols_present, how='all')
        final_rows = len(data)
        
        if final_rows == 0:
            raise ValueError("No valid data rows found after cleaning")
        
        if initial_rows != final_rows:
            print(f"Warning: {initial_rows - final_rows} rows removed due to invalid data")
        
        # Basic validation checks on required columns
        self._perform_data_validation(data, column_status)
        
        return data
    
    def _perform_data_validation(self, data, column_status):
        """
        Perform basic validation checks on the data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to validate
        column_status : dict
            Status of all columns
        """
        # Check for negative values in time-series data
        if (data['force_kN'] < 0).any():
            print("Warning: Negative force values found - this may indicate compression data")
        
        # Validate dimension columns - these should have positive values
        if ('original_gauge_length_mm' in data.columns and 
            column_status.get('original_gauge_length_mm') == 'adequate'):
            orig_length = data['original_gauge_length_mm'].dropna().iloc[0]
            if orig_length <= 0:
                print("Warning: Invalid original gauge length value found")
        
        if ('original_diameter_mm' in data.columns and 
            column_status.get('original_diameter_mm') == 'adequate'):
            orig_diameter = data['original_diameter_mm'].dropna().iloc[0]
            if orig_diameter <= 0:
                print("Warning: Invalid original diameter value found")
        
        # Check if time is monotonically increasing
        if not data['time_s'].is_monotonic_increasing:
            print("Warning: Time values are not monotonically increasing")
        
        # Check for reasonable strain values (typically between 0 and 1 for most materials)
        max_strain = data['strain_mm_mm'].max()
        if max_strain > 2:
            print(f"Warning: Very high strain values detected (max: {max_strain:.2f})")
        
        # Check displacement consistency
        if (data['displacement_mm'] < 0).any():
            print("Warning: Negative displacement values found")
        
        # Report on data availability
        time_series_cols = ['time_s', 'displacement_mm', 'force_kN', 'strain_mm_mm']
        time_series_available = [col for col in time_series_cols if col in data.columns and column_status.get(col) == 'adequate']
        
        dimension_cols = ['original_gauge_length_mm', 'original_diameter_mm']
        dimension_available = [col for col in dimension_cols if col in data.columns and column_status.get(col) == 'adequate']
        
        final_cols = ['final_gauge_length_mm', 'final_diameter_mm']  
        final_available = [col for col in final_cols if col in data.columns and column_status.get(col) in ['adequate', 'estimated']]
        
        print(f"Data validation complete. {len(data)} valid data points loaded.")
        print(f"Time-series columns (1-4) available: {', '.join(time_series_available)}")
        print(f"Dimension columns (5-6) available: {', '.join(dimension_available)}")
        if final_available:
            print(f"Final measurement columns (7-8) with data: {', '.join(final_available)}")
        
        missing_final_cols = [col for col in final_cols if column_status.get(col) in ['missing', 'empty', 'insufficient']]
        if missing_final_cols:
            print(f"Final measurement columns (7-8) unavailable: {', '.join(missing_final_cols)}")
            print("Note: Some analysis features will be limited due to missing final measurement data")


# ============================================================================
# TENSILE ANALYSIS MODULE
# ============================================================================

class TensileAnalyzer:
    """
    Main class for tensile test data analysis according to ASTM E8 and E112
    """
    
    def __init__(self, data):
        """
        Initialize the analyzer with tensile test data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Tensile test data with required columns
        """
        self.data = data
        self.engineering_stress = None
        self.engineering_strain = None
        self.true_stress = None
        self.true_strain = None
        
        # Check which columns are available
        self.column_status = getattr(data, 'attrs', {}).get('column_status', {})
        
        # Calculate basic stress and strain
        self._calculate_stress_strain()
    
    def _calculate_stress_strain(self):
        """Calculate engineering and true stress-strain curves"""
        # We now require diameter data (column 6) - it should always be available
        if ('original_diameter_mm' in self.data.columns and 
            self.column_status.get('original_diameter_mm') in ['adequate', 'sparse']):
            
            # Original cross-sectional area (assuming circular cross-section)
            original_area = np.pi * (self.data['original_diameter_mm'] / 2) ** 2
            
            # Engineering stress (σ = F/A₀)
            self.engineering_stress = (self.data['force_kN'] * 1000) / original_area  # MPa
            
        else:
            # This should not happen with the new requirements, but handle gracefully
            print("ERROR: Original diameter data is required but not available!")
            raise ValueError("Original diameter data (column 6) is required for analysis")
        
        # Engineering strain (ε = strain from data)
        self.engineering_strain = self.data['strain_mm_mm']
        
        # True stress and strain calculations
        # True stress (σₜ = σ(1 + ε))
        self.true_stress = self.engineering_stress * (1 + self.engineering_strain)
        
        # True strain (εₜ = ln(1 + ε))
        self.true_strain = np.log(1 + self.engineering_strain)
        
        # Apply smoothing to reduce noise
        self.engineering_stress = self._smooth_data(self.engineering_stress)
        self.true_stress = self._smooth_data(self.true_stress)
    
    def _smooth_data(self, data, window_length=11, polyorder=3):
        """Apply Savitzky-Golay smoothing to data"""
        if len(data) < window_length:
            window_length = len(data) // 2 * 2 + 1  # Ensure odd number
            if window_length < 3:
                return data
        
        if window_length <= polyorder:
            polyorder = window_length - 1
        
        try:
            return savgol_filter(data, window_length, polyorder)
        except:
            return data
    
    def analyze(self):
        """
        Perform complete tensile analysis according to ASTM E8 and E112
        
        Returns:
        --------
        dict
            Dictionary containing all analysis results
        """
        results = {}
        
        # Calculate test information first
        results.update(self._calculate_test_information())
        
        # Calculate mechanical properties
        results.update(self._calculate_yield_strength())
        results.update(self._calculate_ultimate_strength())
        results.update(self._calculate_fracture_strength())
        results.update(self._calculate_uniform_elongation())
        results.update(self._calculate_elastic_modulus())
        results.update(self._calculate_elongations())
        results.update(self._calculate_moduli())
        results.update(self._fit_ramberg_osgood())
        results.update(self._fit_hollomon_equation())
        
        return results
    
    def _calculate_yield_strength(self, offset=0.002):
        """
        Calculate yield strength using 0.2% offset method (ASTM E8)
        
        Parameters:
        -----------
        offset : float
            Offset strain for yield strength determination (default 0.2% = 0.002)
        
        Returns:
        --------
        dict
            Yield strength and related parameters
        """
        # Find elastic region (first ~30% of data or up to obvious yielding)
        elastic_end_idx = min(len(self.engineering_strain) // 3, 
                             np.argmax(self.engineering_strain > 0.02 * np.max(self.engineering_strain)))
        
        if elastic_end_idx < 10:  # Need minimum points for reliable fit
            elastic_end_idx = min(50, len(self.engineering_strain) // 2)
        
        # Fit linear line to elastic region
        elastic_strain = self.engineering_strain[:elastic_end_idx]
        elastic_stress = self.engineering_stress[:elastic_end_idx]
        
        # Remove points near zero to avoid fitting issues
        valid_points = (elastic_strain > 0.0001) & (elastic_stress > 0)
        elastic_strain = elastic_strain[valid_points]
        elastic_stress = elastic_stress[valid_points]
        
        if len(elastic_strain) < 5:
            raise ValueError("Insufficient data points in elastic region")
        
        # Linear fit: σ = E * ε
        elastic_modulus = np.polyfit(elastic_strain, elastic_stress, 1)[0]
        
        # Create offset line: σ = E * (ε - offset)
        offset_strain = self.engineering_strain + offset
        offset_line = elastic_modulus * offset_strain
        
        # Find intersection of offset line with stress-strain curve
        differences = np.abs(self.engineering_stress - offset_line)
        yield_idx = np.argmin(differences[elastic_end_idx:]) + elastic_end_idx
        
        yield_strength = float(self.engineering_stress[yield_idx])
        yield_strain = float(self.engineering_strain[yield_idx])
        
        return {
            'yield_strength': yield_strength,
            'yield_strain': yield_strain,
            'yield_stress_index': yield_idx
        }
    
    def _calculate_ultimate_strength(self):
        """
        Calculate ultimate tensile strength (maximum stress)
        
        Returns:
        --------
        dict
            Ultimate strength and related parameters
        """
        uts_idx = np.argmax(self.engineering_stress)
        ultimate_strength = float(self.engineering_stress[uts_idx])
        ultimate_strain = float(self.engineering_strain[uts_idx])
        
        return {
            'ultimate_strength': ultimate_strength,
            'ultimate_strain': ultimate_strain,
            'ultimate_stress_index': uts_idx
        }
    
    def _calculate_fracture_strength(self):
        """
        Calculate fracture strength (stress at final data point)
        
        Returns:
        --------
        dict
            Fracture strength and related parameters
        """
        # Fracture occurs at the last data point
        fracture_strength = float(self.engineering_stress.iloc[-1] if hasattr(self.engineering_stress, 'iloc') 
                                else self.engineering_stress[-1])
        fracture_strain = float(self.engineering_strain.iloc[-1] if hasattr(self.engineering_strain, 'iloc') 
                              else self.engineering_strain[-1])
        fracture_idx = len(self.engineering_stress) - 1
        
        return {
            'fracture_strength': fracture_strength,
            'fracture_strain': fracture_strain,
            'fracture_stress_index': fracture_idx
        }
    
    def _calculate_uniform_elongation(self):
        """
        Calculate uniform elongation (elongation at ultimate tensile strength)
        
        Uniform elongation is the elongation when maximum stress is reached.
        This represents the elongation at the point of maximum load before necking begins.
        
        Returns:
        --------
        dict
            Uniform elongation parameters
        """
        # Uniform elongation is the elongation (percentage) at maximum stress (UTS)
        uts_idx = np.argmax(self.engineering_stress)
        uniform_elongation = float(self.engineering_strain[uts_idx]) * 100  # Convert to percentage
        
        return {
            'uniform_elongation': uniform_elongation,
            'uniform_elongation_strain': float(self.engineering_strain[uts_idx])
        }
    
    def _calculate_test_information(self):
        """
        Calculate basic test information (max force, max displacement, test time)
        
        Returns:
        --------
        dict
            Test information parameters
        """
        # Maximum force
        max_force = float(self.data['force_kN'].max())
        
        # Maximum displacement
        max_displacement = float(self.data['displacement_mm'].max())
        
        # Test time (total duration)
        test_time = float(self.data['time_s'].max() - self.data['time_s'].min())
        
        return {
            'max_force': max_force,
            'max_displacement': max_displacement,
            'test_time': test_time
        }
    
    def _calculate_elastic_modulus(self):
        """
        Calculate elastic modulus using middle 50% of elastic region (ASTM E112)
        
        Returns:
        --------
        dict
            Elastic modulus
        """
        # Find the linear elastic region
        # Look for the region where stress increases most linearly with strain
        
        # Calculate derivative to find most linear region
        if len(self.engineering_strain) < 20:
            raise ValueError("Insufficient data for elastic modulus calculation")
        
        # Find the region with most consistent slope
        window_size = min(50, len(self.engineering_strain) // 4)
        best_r_squared = 0
        best_modulus = 0
        
        for start_idx in range(0, len(self.engineering_strain) - window_size, 5):
            end_idx = start_idx + window_size
            
            strain_window = self.engineering_strain[start_idx:end_idx]
            stress_window = self.engineering_stress[start_idx:end_idx]
            
            # Skip if too close to zero
            if strain_window.max() < 0.001:
                continue
            
            # Linear fit
            if len(strain_window) > 5:
                try:
                    coeffs = np.polyfit(strain_window, stress_window, 1)
                    modulus = coeffs[0]
                    
                    # Calculate R-squared
                    ss_res = np.sum((stress_window - (modulus * strain_window + coeffs[1])) ** 2)
                    ss_tot = np.sum((stress_window - np.mean(stress_window)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    if r_squared > best_r_squared and modulus > 0:
                        best_r_squared = r_squared
                        best_modulus = modulus
                        
                except:
                    continue
        
        if best_modulus == 0:
            # Fallback: use first 20% of data
            elastic_end = int(0.2 * len(self.engineering_strain))
            elastic_strain = self.engineering_strain[:elastic_end]
            elastic_stress = self.engineering_stress[:elastic_end]
            
            valid_points = (elastic_strain > 0.0001) & (elastic_stress > 1)
            if np.sum(valid_points) > 5:
                best_modulus = np.polyfit(elastic_strain[valid_points], 
                                        elastic_stress[valid_points], 1)[0]
            else:
                best_modulus = self.engineering_stress.max() / self.engineering_strain.max()
        
        return {
            'elastic_modulus': float(best_modulus),
            'elastic_modulus_r_squared': float(best_r_squared)
        }
    
    def _calculate_elongations(self):
        """
        Calculate elongation at fracture and elongation after fracture
        
        Elongation at fracture is the elongation when maximum strain is reached (final data point).
        This represents the total elongation at the point of specimen fracture.
        
        Returns:
        --------
        dict
            Elongation values
        """
        results = {}
        
        # Elongation at fracture (strain at maximum strain - final data point)
        elongation_at_fracture = float(self.engineering_strain.iloc[-1] if hasattr(self.engineering_strain, 'iloc') 
                                     else self.engineering_strain[-1]) * 100  # Convert to percentage
        results['elongation_at_fracture'] = elongation_at_fracture
        
        # Elongation after fracture - only if we have the optional final gauge length data
        if ('final_gauge_length_mm' in self.data.columns and 
            'original_gauge_length_mm' in self.data.columns and
            self.column_status.get('final_gauge_length_mm') in ['adequate', 'sparse'] and
            self.column_status.get('original_gauge_length_mm') in ['adequate', 'sparse']):
            
            # Use final gauge length measurements
            original_length = self.data['original_gauge_length_mm'].iloc[0]
            final_length = self.data['final_gauge_length_mm'].iloc[-1]
            elongation_after_fracture = ((final_length - original_length) / original_length) * 100
            results['elongation_after_fracture'] = elongation_after_fracture
            
        else:
            # Use final strain as estimate only if final gauge length data is missing
            if not any(self.column_status.get(col) in ['adequate', 'sparse'] 
                      for col in ['final_gauge_length_mm']):
                # Set to None to indicate unavailable
                results['elongation_after_fracture'] = None
                print("Note: Elongation after fracture not calculated - final gauge length data not available (column 7)")
            else:
                elongation_after_fracture = float(self.engineering_strain.iloc[-1]) * 100
                results['elongation_after_fracture'] = elongation_after_fracture
        
        # Reduction of area calculation - only if final diameter/width data is available
        if ('final_diameter_mm' in self.data.columns and 
            'original_diameter_mm' in self.data.columns and
            self.column_status.get('final_diameter_mm') in ['adequate', 'sparse'] and
            self.column_status.get('original_diameter_mm') in ['adequate', 'sparse']):
            
            # Get diameter values, ensuring they are not NaN
            original_diameter = self.data['original_diameter_mm'].dropna().iloc[0] if not self.data['original_diameter_mm'].dropna().empty else None
            final_diameter = self.data['final_diameter_mm'].dropna().iloc[-1] if not self.data['final_diameter_mm'].dropna().empty else None
            
            if original_diameter is not None and final_diameter is not None and original_diameter > 0 and final_diameter > 0:
                # Calculate areas for circular cross-section
                original_area = np.pi * (original_diameter / 2) ** 2  # A₀ = π(d₀/2)²
                final_area = np.pi * (final_diameter / 2) ** 2        # Aᶠ = π(dᶠ/2)²
                
                # Calculate reduction of area as percentage: ((A₀ - Aᶠ) / A₀) × 100
                reduction_of_area = ((original_area - final_area) / original_area) * 100
                results['reduction_of_area'] = reduction_of_area
                
                print(f"Reduction of area calculated: Original area = {original_area:.3f} mm², Final area = {final_area:.3f} mm², Reduction = {reduction_of_area:.2f}%")
            else:
                results['reduction_of_area'] = None
                print("Note: Reduction of area not calculated - invalid diameter data")
                
        # Alternative: Check for rectangular specimens (width instead of diameter)
        elif ('final_width_mm' in self.data.columns and 'thickness_mm' in self.data.columns and
              'original_width_mm' in self.data.columns and
              self.column_status.get('final_width_mm') in ['adequate', 'sparse'] and
              self.column_status.get('original_width_mm') in ['adequate', 'sparse']):
            
            # Get width and thickness values for rectangular specimens
            original_width = self.data['original_width_mm'].dropna().iloc[0] if not self.data['original_width_mm'].dropna().empty else None
            final_width = self.data['final_width_mm'].dropna().iloc[-1] if not self.data['final_width_mm'].dropna().empty else None
            thickness = self.data['thickness_mm'].dropna().iloc[0] if not self.data['thickness_mm'].dropna().empty else None
            
            if original_width is not None and final_width is not None and thickness is not None and all(v > 0 for v in [original_width, final_width, thickness]):
                # Calculate areas for rectangular cross-section
                original_area = original_width * thickness  # A₀ = width × thickness
                final_area = final_width * thickness        # Aᶠ = final_width × thickness (assuming thickness constant)
                
                # Calculate reduction of area as percentage: ((A₀ - Aᶠ) / A₀) × 100
                reduction_of_area = ((original_area - final_area) / original_area) * 100
                results['reduction_of_area'] = reduction_of_area
                
                print(f"Reduction of area calculated (rectangular): Original area = {original_area:.3f} mm², Final area = {final_area:.3f} mm², Reduction = {reduction_of_area:.2f}%")
            else:
                results['reduction_of_area'] = None
                print("Note: Reduction of area not calculated - invalid width/thickness data")
                
        else:
            # Set to None to indicate unavailable - no estimation
            results['reduction_of_area'] = None
            if not any(self.column_status.get(col) in ['adequate', 'sparse'] 
                      for col in ['final_diameter_mm', 'final_width_mm']):
                print("Note: Reduction of area not calculated - final diameter/width data not available (column 8)")
        
        return results
    
    def _calculate_moduli(self):
        """
        Calculate modulus of toughness and resilience
        
        Returns:
        --------
        dict
            Toughness and resilience moduli
        """
        # Modulus of resilience (area under stress-strain curve up to yield point)
        yield_results = self._calculate_yield_strength()
        yield_idx = yield_results['yield_stress_index']
        
        resilience_strain = self.engineering_strain[:yield_idx+1]
        resilience_stress = self.engineering_stress[:yield_idx+1]
        
        # Ensure we have valid data points
        if len(resilience_strain) > 1:
            # Use numpy.trapezoid instead of deprecated trapz
            try:
                modulus_of_resilience = np.trapezoid(resilience_stress, resilience_strain)  # MPa
            except AttributeError:
                # Fallback for older numpy versions
                modulus_of_resilience = np.trapz(resilience_stress, resilience_strain)  # MPa
        else:
            modulus_of_resilience = 0.0
        
        # Modulus of toughness (area under entire stress-strain curve)
        if len(self.engineering_strain) > 1:
            try:
                modulus_of_toughness = np.trapezoid(self.engineering_stress, self.engineering_strain)  # MPa
            except AttributeError:
                # Fallback for older numpy versions
                modulus_of_toughness = np.trapz(self.engineering_stress, self.engineering_strain)  # MPa
        else:
            modulus_of_toughness = 0.0
        
        return {
            'modulus_of_resilience': float(modulus_of_resilience),
            'modulus_of_toughness': float(modulus_of_toughness)
        }
    
    def _fit_ramberg_osgood(self):
        """
        Fit Ramberg-Osgood equation variant to the data
        ε = σ/E + (σ/K)^n
        Where: ε is total strain, σ is stress, E is Young's modulus, 
               K is strength coefficient, n is hardening exponent
        
        Returns:
        --------
        dict
            Ramberg-Osgood parameters
        """
        try:
            # Get elastic modulus and yield strength
            elastic_modulus = self._calculate_elastic_modulus()['elastic_modulus']
            yield_results = self._calculate_yield_strength()
            yield_stress = yield_results['yield_strength']
            yield_idx = yield_results['yield_stress_index']
            
            # Use data beyond yield point for better fitting
            # Ramberg-Osgood is primarily for plastic deformation
            strain_data = self.engineering_strain[yield_idx:]
            stress_data = self.engineering_stress[yield_idx:]
            
            # Remove any decreasing portions (necking region)
            max_stress_idx = np.argmax(stress_data)
            strain_data = strain_data[:max_stress_idx+1]
            stress_data = stress_data[:max_stress_idx+1]
            
            # Ensure we have enough data points
            if len(strain_data) < 10:
                return {'ramberg_osgood': {'K': 0, 'n': 0, 'r_squared': 0}}
            
            # Alternative Ramberg-Osgood equation: ε = σ/E + (σ/K)^n
            # Where K is strength coefficient and n is hardening exponent
            def ramberg_osgood_variant_residual(params, strain, stress, E):
                K, n = params
                if K <= 0 or n <= 0:
                    return np.inf
                
                try:
                    # Calculate elastic strain component
                    elastic_strain = stress / E
                    
                    # Calculate plastic strain component using variant: (σ/K)^n
                    # For very large n values or stress values, use log-space calculations
                    if n > 5 or np.max(stress) > K * 10:
                        # Use log-space calculation: log(plastic_strain) = n*log(σ/K)
                        stress_ratio = np.maximum(stress / K, 1e-10)  # Avoid log(0)
                        log_plastic_strain = n * np.log(stress_ratio)
                        
                        # Check for overflow before exponentiating
                        if np.any(log_plastic_strain > 10):  # e^10 ≈ 22000, reasonable limit
                            return np.inf
                        
                        plastic_strain = np.exp(log_plastic_strain)
                    else:
                        plastic_strain = (stress / K) ** n
                    
                    # Check for reasonable plastic strain values
                    if not np.all(np.isfinite(plastic_strain)) or np.max(plastic_strain) > 50:
                        return np.inf
                    
                    # Total predicted strain
                    predicted_strain = elastic_strain + plastic_strain
                    
                    # Calculate residuals with proper weighting
                    residuals = strain - predicted_strain
                    
                    # Weight residuals to emphasize higher stress regions
                    weights = (stress / np.max(stress)) ** 0.3
                    weighted_residuals = residuals * weights
                    
                    result = np.sum(weighted_residuals ** 2)
                    
                    # Return infinity if result is not finite
                    if not np.isfinite(result):
                        return np.inf
                    
                    return result
                    
                except (OverflowError, ValueError, RuntimeWarning, FloatingPointError):
                    return np.inf
            
            # Alternative formulation for plastic strain: ε_p = (σ/K)^n
            def ramberg_osgood_variant_plastic_residual(params, strain, stress, E):
                K, n = params
                if K <= 0 or n <= 0:
                    return np.inf
                
                try:
                    # Calculate elastic strain
                    elastic_strain = stress / E
                    
                    # Calculate plastic strain (total - elastic)
                    plastic_strain_observed = strain - elastic_strain
                    
                    # Variant plastic strain prediction: (σ/K)^n
                    plastic_strain_predicted = (stress / K) ** n
                    
                    # Only consider positive plastic strains
                    valid_mask = plastic_strain_observed > 0
                    if np.sum(valid_mask) < 5:
                        return np.inf
                    
                    residuals = plastic_strain_observed[valid_mask] - plastic_strain_predicted[valid_mask]
                    
                    # Apply weighting based on stress level
                    stress_weights = (stress[valid_mask] / np.max(stress)) ** 0.3
                    weighted_residuals = residuals * stress_weights
                    
                    return np.sum(weighted_residuals ** 2)
                    
                except (OverflowError, ValueError, RuntimeWarning):
                    return np.inf
            
            # Try multiple initial guesses to find global minimum
            best_result = None
            best_r_squared = -np.inf
            
            # Different initial guesses for K (strength coefficient) and n (hardening exponent)
            # K should be related to yield strength, typically 1-3 times yield strength
            yield_strength_estimate = np.max(stress_data) * 0.8  # Rough estimate
            initial_guesses = [
                [yield_strength_estimate * 1.0, 0.1],     # K ≈ yield, low n
                [yield_strength_estimate * 1.5, 0.2],     # K > yield, low n
                [yield_strength_estimate * 0.8, 0.5],     # K < yield, medium n
                [yield_strength_estimate * 2.0, 0.8],     # K >> yield, high n < 1
                [yield_strength_estimate * 0.5, 1.5],     # K < yield, n > 1
                [yield_strength_estimate * 3.0, 1.2],     # K >>> yield, n > 1
                [yield_strength_estimate * 1.2, 2.0],     # K > yield, high n
                [yield_strength_estimate * 0.3, 0.3],     # K << yield, low n
                [yield_strength_estimate * 4.0, 1.0],     # K >>>> yield, medium n
                [yield_strength_estimate * 0.6, 0.5],     # K < yield, medium n
                [yield_strength_estimate * 1.8, 2.5],     # K >> yield, high n
                [yield_strength_estimate * 0.1, 5.0],     # K <<< yield, very high n
            ]
            
            # Bounds for parameters: K > 0 (strength coefficient), n > 0 (hardening exponent)
            max_stress = np.max(stress_data)
            bounds = [(max_stress * 0.01, max_stress * 10.0), (0.01, 10.0)]
            
            for initial_guess in initial_guesses:
                try:
                    # Try both total strain and plastic strain formulations
                    for residual_func in [ramberg_osgood_variant_residual, ramberg_osgood_variant_plastic_residual]:
                        # Perform optimization with multiple methods
                        methods = ['L-BFGS-B', 'TNC']
                        
                        for method in methods:
                            try:
                                result = optimize.minimize(
                                    residual_func,
                                    initial_guess,
                                    args=(strain_data, stress_data, elastic_modulus),
                                    bounds=bounds,
                                    method=method,
                                    options={'maxiter': 1000} if method == 'L-BFGS-B' else {}
                                )
                                
                                if result.success and not np.isnan(result.fun):
                                    K, n = result.x
                                    
                                    # Additional validation with expanded bounds
                                    if K > 0 and n > 0 and K < max_stress * 20 and n < 15:
                                        # Calculate R-squared for total strain prediction
                                        elastic_strain = stress_data / elastic_modulus
                                        
                                        # Handle potential overflow for very large K values
                                        try:
                                            plastic_strain = (stress_data / K) ** n
                                            
                                            # Check for reasonable plastic strain values
                                            if np.all(np.isfinite(plastic_strain)) and np.max(plastic_strain) < 100:
                                                predicted_strain = elastic_strain + plastic_strain
                                                
                                                # Check for reasonable predictions
                                                if np.all(np.isfinite(predicted_strain)) and np.all(predicted_strain >= 0):
                                                    ss_res = np.sum((strain_data - predicted_strain) ** 2)
                                                    ss_tot = np.sum((strain_data - np.mean(strain_data)) ** 2)
                                                    
                                                    if ss_tot > 1e-12:  # Avoid division by zero
                                                        r_squared = 1 - (ss_res / ss_tot)
                                                        
                                                        # Keep the best result
                                                        if r_squared > best_r_squared and r_squared > -0.5:  # Allow slightly negative R²
                                                            best_r_squared = r_squared
                                                            best_result = {
                                                                'K': float(K),
                                                                'n': float(n),
                                                                'r_squared': float(r_squared)
                                                            }
                                        except (OverflowError, RuntimeWarning):
                                            continue
                            except:
                                continue
                                        
                except:
                    continue
            
            # Try differential evolution for global optimization as fallback
            if best_result is None or best_r_squared < 0.6:
                try:
                    def objective(params):
                        result = ramberg_osgood_variant_residual(params, strain_data, stress_data, elastic_modulus)
                        # Add penalty for extreme parameter values to guide optimization
                        K, n = params
                        if K > max_stress * 8:  # Penalty for very high K
                            result += (K - max_stress * 8) * 1e-6
                        if n > 8:  # Penalty for very high n
                            result += (n - 8) * 1e-3
                        return result
                    
                    # Use differential evolution with multiple strategies and expanded search
                    strategies = ['best1bin', 'best2bin', 'rand1bin', 'currenttobest1bin']
                    
                    for strategy in strategies:
                        result_de = optimize.differential_evolution(
                            objective,
                            bounds,
                            strategy=strategy,
                            maxiter=750,  # Increased iterations
                            popsize=20,   # Increased population
                            seed=42,
                            atol=1e-10,
                            tol=1e-10,
                            workers=1,    # Single worker for consistency
                            updating='deferred'  # Better for noisy functions
                        )
                        
                        if result_de.success and not np.isnan(result_de.fun):
                            K, n = result_de.x
                            
                            # Validate parameters with expanded bounds
                            if K > 0 and n > 0 and K < max_stress * 20 and n < 15:
                                # Calculate R-squared with overflow protection
                                try:
                                    elastic_strain = stress_data / elastic_modulus
                                    plastic_strain = (stress_data / K) ** n
                                    
                                    # Check for reasonable plastic strain values
                                    if np.all(np.isfinite(plastic_strain)) and np.max(plastic_strain) < 100:
                                        predicted_strain = elastic_strain + plastic_strain
                                        
                                        if np.all(np.isfinite(predicted_strain)) and np.all(predicted_strain >= 0):
                                            ss_res = np.sum((strain_data - predicted_strain) ** 2)
                                            ss_tot = np.sum((strain_data - np.mean(strain_data)) ** 2)
                                            
                                            if ss_tot > 1e-12:
                                                r_squared = 1 - (ss_res / ss_tot)
                                                
                                                if best_result is None or r_squared > best_r_squared:
                                                    best_result = {
                                                        'K': float(K),
                                                        'n': float(n),
                                                        'r_squared': float(r_squared)
                                                    }
                                                    best_r_squared = r_squared
                                except (OverflowError, RuntimeWarning):
                                    continue
                except:
                    pass
            
            # Try basin-hopping for very challenging cases
            if best_result is None or best_r_squared < 0.4:
                try:
                    def objective_basin(params):
                        if params[0] <= 0 or params[1] <= 0 or params[0] > max_stress * 15 or params[1] > 10:
                            return 1e10
                        return ramberg_osgood_variant_residual(params, strain_data, stress_data, elastic_modulus)
                    
                    # Try basin-hopping from multiple starting points
                    best_basin_result = None
                    best_basin_r_squared = -np.inf
                    
                    starting_points = [
                        [yield_strength_estimate * 1.0, 0.5], 
                        [yield_strength_estimate * 2.0, 1.0], 
                        [yield_strength_estimate * 0.5, 2.0], 
                        [yield_strength_estimate * 3.0, 3.0], 
                        [yield_strength_estimate * 0.1, 0.8]
                    ]
                    
                    for start_point in starting_points:
                        try:
                            result_basin = optimize.basinhopping(
                                objective_basin,
                                start_point,
                                niter=100,
                                T=1.0,
                                stepsize=0.5,
                                minimizer_kwargs={
                                    'method': 'L-BFGS-B',
                                    'bounds': bounds
                                },
                                seed=42
                            )
                            
                            if result_basin.lowest_optimization_result.success:
                                K, n = result_basin.x
                                
                                if K > 0 and n > 0 and K < max_stress * 20 and n < 15:
                                    try:
                                        elastic_strain = stress_data / elastic_modulus
                                        plastic_strain = (stress_data / K) ** n
                                        
                                        if np.all(np.isfinite(plastic_strain)) and np.max(plastic_strain) < 100:
                                            predicted_strain = elastic_strain + plastic_strain
                                            
                                            if np.all(np.isfinite(predicted_strain)) and np.all(predicted_strain >= 0):
                                                ss_res = np.sum((strain_data - predicted_strain) ** 2)
                                                ss_tot = np.sum((strain_data - np.mean(strain_data)) ** 2)
                                                
                                                if ss_tot > 1e-12:
                                                    r_squared = 1 - (ss_res / ss_tot)
                                                    
                                                    if r_squared > best_basin_r_squared:
                                                        best_basin_r_squared = r_squared
                                                        best_basin_result = {
                                                            'K': float(K),
                                                            'n': float(n),
                                                            'r_squared': float(r_squared)
                                                        }
                                    except (OverflowError, RuntimeWarning):
                                        continue
                        except:
                            continue
                    
                    # Use basin-hopping result if it's better
                    if (best_basin_result is not None and 
                        (best_result is None or best_basin_r_squared > best_r_squared)):
                        best_result = best_basin_result
                        best_r_squared = best_basin_r_squared
                        
                except:
                    pass
            
            # Return best result or default values
            if best_result is not None:
                # Calculate elastic and plastic components at yield stress for reference
                K = best_result['K']
                n = best_result['n']
                
                # Calculate components at yield stress
                elastic_component_at_yield = yield_stress / elastic_modulus
                plastic_component_at_yield = (yield_stress / K) ** n
                
                # Add components to the result
                best_result['elastic_component_at_yield'] = float(elastic_component_at_yield)
                best_result['plastic_component_at_yield'] = float(plastic_component_at_yield)
                
                return {'ramberg_osgood': best_result}
            else:
                return {'ramberg_osgood': {'K': 0, 'n': 0, 'r_squared': 0, 
                                         'elastic_component_at_yield': 0, 'plastic_component_at_yield': 0}}
                
        except Exception as e:
            warnings.warn(f"Ramberg-Osgood fitting failed: {str(e)}")
            return {'ramberg_osgood': {'K': 0, 'n': 0, 'r_squared': 0, 
                                     'elastic_component_at_yield': 0, 'plastic_component_at_yield': 0}}
    
    def _fit_hollomon_equation(self):
        """
        Fit Hollomon equation to the true stress-strain data
        σₜ = K × εₜⁿ (where σₜ is true stress, εₜ is true strain)
        
        Returns:
        --------
        dict
            Hollomon equation parameters
        """
        try:
            # Use plastic portion of the true stress-strain curve (beyond yield)
            yield_results = self._calculate_yield_strength()
            yield_idx = yield_results['yield_stress_index']
            
            if yield_idx >= len(self.true_strain) - 10:
                return {'hollomon': {'K': 0, 'n': 0, 'r_squared': 0}}
            
            # Take data beyond yield point for true stress-strain
            true_strain_plastic = self.true_strain[yield_idx:]
            true_stress_plastic = self.true_stress[yield_idx:]
            
            # Remove any decreasing portions (necking region) and ensure positive values
            max_stress_idx = np.argmax(true_stress_plastic)
            true_strain_plastic = true_strain_plastic[:max_stress_idx+1]
            true_stress_plastic = true_stress_plastic[:max_stress_idx+1]
            
            # Filter out zero or negative values for log transformation
            valid_mask = (true_strain_plastic > 1e-6) & (true_stress_plastic > 0)
            true_strain_plastic = true_strain_plastic[valid_mask]
            true_stress_plastic = true_stress_plastic[valid_mask]
            
            if len(true_strain_plastic) < 10:
                return {'hollomon': {'K': 0, 'n': 0, 'r_squared': 0}}
            
            # Fit using logarithmic transformation: ln(σₜ) = ln(K) + n × ln(εₜ)
            try:
                log_strain = np.log(true_strain_plastic)
                log_stress = np.log(true_stress_plastic)
                
                # Linear fit in log space
                coeffs = np.polyfit(log_strain, log_stress, 1)
                n_hollomon = coeffs[0]  # Strain hardening exponent
                K_hollomon = np.exp(coeffs[1])  # Strength coefficient
                
                # Calculate R-squared in original space
                predicted_stress = K_hollomon * (true_strain_plastic ** n_hollomon)
                ss_res = np.sum((true_stress_plastic - predicted_stress) ** 2)
                ss_tot = np.sum((true_stress_plastic - np.mean(true_stress_plastic)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                return {
                    'hollomon': {
                        'K': float(K_hollomon),
                        'n': float(n_hollomon),
                        'r_squared': float(r_squared)
                    }
                }
                
            except (ValueError, np.linalg.LinAlgError):
                # Fallback to non-linear fitting if linear method fails
                def hollomon_residual(params, strain, stress):
                    K, n = params
                    if K <= 0 or n <= 0:
                        return np.inf
                    predicted = K * (strain ** n)
                    return np.sum((stress - predicted) ** 2)
                
                # Initial guess
                initial_guess = [true_stress_plastic.max(), 0.2]
                
                # Bounds for parameters
                bounds = [(true_stress_plastic.max() * 0.5, true_stress_plastic.max() * 2), (0.05, 1.0)]
                
                result = optimize.minimize(
                    hollomon_residual,
                    initial_guess,
                    args=(true_strain_plastic, true_stress_plastic),
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success:
                    K, n = result.x
                    
                    # Calculate R-squared
                    predicted = K * (true_strain_plastic ** n)
                    ss_res = np.sum((true_stress_plastic - predicted) ** 2)
                    ss_tot = np.sum((true_stress_plastic - np.mean(true_stress_plastic)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    return {
                        'hollomon': {
                            'K': float(K),
                            'n': float(n),
                            'r_squared': float(r_squared)
                        }
                    }
                else:
                    return {'hollomon': {'K': 0, 'n': 0, 'r_squared': 0}}
                
        except Exception as e:
            warnings.warn(f"Hollomon equation fitting failed: {str(e)}")
            return {'hollomon': {'K': 0, 'n': 0, 'r_squared': 0}}


# ============================================================================
# GUI APPLICATION CLASS
# ============================================================================


class TensileAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tensile Analyzer (ASTM E8 & ASTM E112)")
        self.root.geometry("1200x800")
        
        # Maximize the window
        self.root.state('zoomed')  # Windows-specific maximization
        
        # Initialize variables
        self.data = None
        self.analyzer = None
        self.results = None
        
        self.setup_menu()
        self.setup_gui()
        
    def setup_menu(self):
        """Setup the menu bar"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Data...", command=self.import_data)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Analysis", command=self.run_analysis)
        analysis_menu.add_command(label="Clear Results", command=self.clear_results)
        
        # Plot menu
        plot_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Plotting", menu=plot_menu)
        plot_menu.add_command(label="Stress-Strain with Properties", command=self.plot_stress_strain_properties)
        plot_menu.add_command(label="Stress-Strain with Toughness/Resilience", command=self.plot_toughness_resilience)
        plot_menu.add_command(label="Engineering vs True Stress-Strain", command=self.plot_engineering_vs_true)
        plot_menu.add_command(label="Ramberg-Osgood Fit", command=self.plot_ramberg_osgood)
        plot_menu.add_command(label="Hollomon Equation Fit", command=self.plot_hollomon_equation)
        plot_menu.add_separator()
        plot_menu.add_command(label="Export Current Plot...", command=self.export_plot)
        plot_menu.add_command(label="Export All Plots...", command=self.export_all_plots)
        plot_menu.add_command(label="Clear Plot", command=self.clear_plot)
        
        # Help menu
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="View Formulae", command=self.show_formulae)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel for data info and results
        left_panel = ttk.LabelFrame(main_frame, text="Data & Results", padding="5")
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(1, weight=1)
        
        # Data info frame
        data_frame = ttk.LabelFrame(left_panel, text="Data Information", padding="5")
        data_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        data_frame.columnconfigure(1, weight=1)
        
        ttk.Label(data_frame, text="File:").grid(row=0, column=0, sticky=tk.W)
        self.file_label = ttk.Label(data_frame, text="No file loaded")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        ttk.Label(data_frame, text="Rows:").grid(row=1, column=0, sticky=tk.W)
        self.rows_label = ttk.Label(data_frame, text="-")
        self.rows_label.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Button(data_frame, text="Import Data", command=self.import_data).grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(data_frame, text="Run Analysis", command=self.run_analysis).grid(row=3, column=0, columnspan=2, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(left_panel, text="Analysis Results", padding="5")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Results tree
        self.results_tree = ttk.Treeview(results_frame, columns=("Value", "Unit"), show="tree headings")
        self.results_tree.heading("#0", text="Property")
        self.results_tree.heading("Value", text="Value")
        self.results_tree.heading("Unit", text="Unit")
        self.results_tree.column("#0", width=200)
        self.results_tree.column("Value", width=100)
        self.results_tree.column("Unit", width=80)
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Right panel for plots
        plot_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="5")
        plot_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Plot toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(toolbar_frame, text="Clear Plot", command=self.clear_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar_frame, text="Export Plot", command=self.export_plot).pack(side=tk.LEFT, padx=5)
        
    def import_data(self):
        """Import data from file"""
        file_path = filedialog.askopenfilename(
            title="Select Tensile Test Data File",
            filetypes=[
                ("All supported", "*.csv;*.xlsx;*.xls;*.txt"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx;*.xls"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                loader = DataLoader()
                self.data = loader.load_data(file_path)
                
                self.file_label.config(text=os.path.basename(file_path))
                self.rows_label.config(text=str(len(self.data)))
                
                messagebox.showinfo("Success", f"Data loaded successfully!\n{len(self.data)} rows imported.")
                
                # Plot raw data
                self.plot_raw_data()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
    
    def plot_raw_data(self):
        """Plot raw stress-strain data"""
        if self.data is None:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Check if we have diameter data for proper stress calculation
        column_status = getattr(self.data, 'attrs', {}).get('column_status', {})
        has_diameter = ('original_diameter_mm' in self.data.columns and 
                       column_status.get('original_diameter_mm') in ['adequate', 'sparse'])
        
        if has_diameter:
            # Calculate engineering stress - should always work now since diameter is required
            stress = self.data['force_kN'] * 1000 / (np.pi * (self.data['original_diameter_mm'] / 2) ** 2)
            strain = self.data['strain_mm_mm']
            
            ax.plot(strain, stress, 'b-', linewidth=1, label='Raw Data')
            ax.set_xlabel('Engineering Strain (mm/mm)')
            ax.set_ylabel('Engineering Stress (MPa)')
            ax.set_title('Raw Tensile Test Data')
        else:
            # This should not happen with new requirements, but handle gracefully
            force = self.data['force_kN'] * 1000  # Convert to N
            strain = self.data['strain_mm_mm']
            
            ax.plot(strain, force, 'b-', linewidth=1, label='Raw Data')
            ax.set_xlabel('Engineering Strain (mm/mm)')
            ax.set_ylabel('Force (N)')
            ax.set_title('Raw Tensile Test Data (Force vs Strain - Missing Diameter Data!)')
            print("ERROR: Plotting force instead of stress - diameter data required!")
            
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.canvas.draw()
    
    def run_analysis(self):
        """Run the tensile analysis"""
        if self.data is None:
            messagebox.showerror("Error", "Please import data first!")
            return
            
        try:
            self.analyzer = TensileAnalyzer(self.data)
            self.results = self.analyzer.analyze()
            
            self.update_results_display()
            messagebox.showinfo("Success", "Analysis completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
    
    def update_results_display(self):
        """Update the results tree with analysis results"""
        # Clear existing results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if self.results is None:
            return
        
        # Check what data is available
        column_status = getattr(self.data, 'attrs', {}).get('column_status', {})
        has_diameter = ('original_diameter_mm' in self.data.columns and 
                       column_status.get('original_diameter_mm') in ['adequate', 'sparse'])
        
        # Test Information
        test_info = self.results_tree.insert("", "end", text="Test Information")
        self.results_tree.insert(test_info, "end", text="Maximum Force", 
                                values=(f"{self.results['max_force']:.3f}", "kN"))
        self.results_tree.insert(test_info, "end", text="Maximum Displacement", 
                                values=(f"{self.results['max_displacement']:.3f}", "mm"))
        self.results_tree.insert(test_info, "end", text="Test Duration", 
                                values=(f"{self.results['test_time']:.2f}", "s"))
        
        # Mechanical Properties
        mech_props = self.results_tree.insert("", "end", text="Mechanical Properties")
        
        # With the new requirements, we should always have diameter data
        self.results_tree.insert(mech_props, "end", text="Yield Strength (0.2% offset)", 
                                values=(f"{self.results['yield_strength']:.2f}", "MPa"))
        self.results_tree.insert(mech_props, "end", text="Ultimate Tensile Strength", 
                                values=(f"{self.results['ultimate_strength']:.2f}", "MPa"))
        self.results_tree.insert(mech_props, "end", text="Fracture Strength", 
                                values=(f"{self.results['fracture_strength']:.2f}", "MPa"))
        self.results_tree.insert(mech_props, "end", text="Elastic Modulus", 
                                values=(f"{self.results['elastic_modulus']/1000:.2f}", "GPa"))
        self.results_tree.insert(mech_props, "end", text="Uniform Elongation", 
                                values=(f"{self.results['uniform_elongation']:.2f}", "%"))
        
        self.results_tree.insert(mech_props, "end", text="Elongation at Fracture", 
                                values=(f"{self.results['elongation_at_fracture']:.2f}", "%"))
        
        # Handle elongation after fracture - may be None if final gauge length not available
        elongation_after_fracture = self.results.get('elongation_after_fracture')
        if elongation_after_fracture is not None:
            self.results_tree.insert(mech_props, "end", text="Elongation after Fracture", 
                                    values=(f"{elongation_after_fracture:.2f}", "%"))
        else:
            self.results_tree.insert(mech_props, "end", text="Elongation after Fracture", 
                                    values=("-", "% (data not available)"))
        
        # Handle reduction of area - may be None if final diameter not available
        reduction_of_area = self.results.get('reduction_of_area')
        if reduction_of_area is not None:
            self.results_tree.insert(mech_props, "end", text="Reduction of Area", 
                                    values=(f"{reduction_of_area:.2f}", "%"))
        else:
            self.results_tree.insert(mech_props, "end", text="Reduction of Area", 
                                    values=("-", "% (data not available)"))
        
        # Energy Properties
        energy_props = self.results_tree.insert("", "end", text="Energy Properties")
        self.results_tree.insert(energy_props, "end", text="Modulus of Toughness", 
                                values=(f"{self.results['modulus_of_toughness']:.2f}", "MJ/m³"))
        self.results_tree.insert(energy_props, "end", text="Modulus of Resilience", 
                                values=(f"{self.results['modulus_of_resilience']:.3f}", "MJ/m³"))
        
        # Ramberg-Osgood Parameters (only if available and meaningful)
        if ('ramberg_osgood' in self.results and 
            self.results['ramberg_osgood']['r_squared'] > 0.1):
            ro_params = self.results_tree.insert("", "end", text="Ramberg-Osgood Parameters")
            self.results_tree.insert(ro_params, "end", text="K (Strength Coefficient)", 
                                    values=(f"{self.results['ramberg_osgood']['K']:.1f}", "MPa"))
            self.results_tree.insert(ro_params, "end", text="n (Hardening Exponent)", 
                                    values=(f"{self.results['ramberg_osgood']['n']:.3f}", "-"))
            self.results_tree.insert(ro_params, "end", text="R² (Fit Quality)", 
                                    values=(f"{self.results['ramberg_osgood']['r_squared']:.4f}", "-"))
            
            # Add elastic and plastic components at yield
            if 'elastic_component_at_yield' in self.results['ramberg_osgood']:
                self.results_tree.insert(ro_params, "end", text="Elastic Component at Yield", 
                                        values=(f"{self.results['ramberg_osgood']['elastic_component_at_yield']:.6f}", "mm/mm"))
                self.results_tree.insert(ro_params, "end", text="Plastic Component at Yield", 
                                        values=(f"{self.results['ramberg_osgood']['plastic_component_at_yield']:.6f}", "mm/mm"))
        
        # Hollomon Equation Parameters (only if available and meaningful)
        if ('hollomon' in self.results and 
            self.results['hollomon']['r_squared'] > 0.1):
            hollomon_params = self.results_tree.insert("", "end", text="Hollomon Equation Parameters")
            self.results_tree.insert(hollomon_params, "end", text="K (Strength Coefficient)", 
                                    values=(f"{self.results['hollomon']['K']:.2f}", "MPa"))
            self.results_tree.insert(hollomon_params, "end", text="n (Strain Hardening Exponent)", 
                                    values=(f"{self.results['hollomon']['n']:.3f}", "-"))
            self.results_tree.insert(hollomon_params, "end", text="R² (Fit Quality)", 
                                    values=(f"{self.results['hollomon']['r_squared']:.4f}", "-"))
        
        # Add notes about optional columns status
        missing_optional_cols = [col for col in ['final_gauge_length_mm', 'final_diameter_mm'] 
                               if column_status.get(col) in ['missing', 'empty', 'insufficient']]
        if missing_optional_cols:
            note = self.results_tree.insert("", "end", text="Optional Column Status")
            for col in missing_optional_cols:
                col_name = "Final Gauge Length" if col == 'final_gauge_length_mm' else "Final Diameter"
                col_num = "7" if col == 'final_gauge_length_mm' else "8"
                self.results_tree.insert(note, "end", text=f"Column {col_num} ({col_name})", 
                                       values=("Not available", column_status.get(col, 'missing')))
        
        # Expand all items
        for item in self.results_tree.get_children():
            self.results_tree.item(item, open=True)
    
    def plot_stress_strain_properties(self):
        """Plot stress-strain curve with annotated mechanical properties"""
        if self.data is None or self.results is None:
            messagebox.showerror("Error", "Please import data and run analysis first!")
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Check if we have diameter data
        column_status = getattr(self.data, 'attrs', {}).get('column_status', {})
        has_diameter = ('original_diameter_mm' in self.data.columns and 
                       column_status.get('original_diameter_mm') in ['adequate', 'sparse'])
        
        # Get data from analyzer
        strain = self.analyzer.engineering_strain
        stress = self.analyzer.engineering_stress
        
        ax.plot(strain, stress, 'b-', linewidth=2, label='Stress-Strain Curve')
        
        # Annotate key points
        yield_strain = self.results['yield_strain']
        yield_stress = self.results['yield_strength']
        ult_strain = self.results['ultimate_strain']
        ult_stress = self.results['ultimate_strength']
        fracture_strain = self.results['fracture_strain']
        fracture_stress = self.results['fracture_strength']
        
        # With new requirements, we should always have diameter data
        ax.plot(yield_strain, yield_stress, 'ro', markersize=8, label=f'Yield Strength ({yield_stress:.1f} MPa)')
        ax.plot(ult_strain, ult_stress, 'go', markersize=8, label=f'Ultimate Tensile Strength ({ult_stress:.1f} MPa)')
        ax.plot(fracture_strain, fracture_stress, 'mo', markersize=8, label=f'Fracture Strength ({fracture_stress:.1f} MPa)')
        ylabel = 'Engineering Stress (MPa)'
        title = 'Stress-Strain Curve with Mechanical Properties'
        
        # Add elastic modulus line for middle 50% of elastic region
        elastic_region = strain <= yield_strain * 1.2
        if np.any(elastic_region):
            elastic_strain_full = strain[elastic_region]
            
            # Calculate middle 50% of elastic region
            start_idx = len(elastic_strain_full) // 4  # Start at 25%
            end_idx = start_idx + len(elastic_strain_full) // 2  # End at 75%
            
            if end_idx < len(elastic_strain_full) and start_idx < end_idx:
                elastic_strain_middle = elastic_strain_full[start_idx:end_idx]
                elastic_line_middle = self.results['elastic_modulus'] * elastic_strain_middle
                
                # Plot elastic modulus line for middle 50%
                ax.plot(elastic_strain_middle, elastic_line_middle, 'purple', linewidth=2, 
                       alpha=0.8, label=f'Elastic Modulus ({self.results["elastic_modulus"]/1000:.1f} GPa)')
                
                # Add dots at beginning and end of elastic modulus measurement region
                ax.plot(elastic_strain_middle.iloc[0], elastic_line_middle.iloc[0], 'purple', marker='o', 
                       markersize=6, markerfacecolor='white', markeredgewidth=2)
                ax.plot(elastic_strain_middle.iloc[-1], elastic_line_middle.iloc[-1], 'purple', marker='o', 
                       markersize=6, markerfacecolor='white', markeredgewidth=2)
        
        # Add 0.2% offset line through yield point and extend by 10% of yield strength value
        offset = 0.002  # 0.2% offset
        elastic_modulus = self.results['elastic_modulus']
        
        # Calculate strain corresponding to stress 10% above yield strength
        stress_extension = yield_stress + (yield_stress * 0.1)  # 10% above yield strength
        strain_extension = stress_extension / elastic_modulus + offset  # Corresponding strain
        
        # Create offset line from offset strain to extended strain
        offset_strain_range = np.linspace(offset, strain_extension, 100)
        offset_line = elastic_modulus * (offset_strain_range - offset)
        
        ax.plot(offset_strain_range, offset_line, 'orange', linestyle=':', linewidth=2, 
               alpha=0.8, label='0.2% Offset Line')
        
        ax.set_xlabel('Engineering Strain (mm/mm)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.canvas.draw()
    
    def plot_toughness_resilience(self):
        """Plot stress-strain curve with highlighted areas for toughness and resilience"""
        if self.data is None or self.results is None:
            messagebox.showerror("Error", "Please import data and run analysis first!")
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        strain = self.analyzer.engineering_strain
        stress = self.analyzer.engineering_stress
        
        ax.plot(strain, stress, 'b-', linewidth=2, label='Stress-Strain Curve')
        
        # Highlight resilience area (up to yield point)
        yield_strain = self.results['yield_strain']
        resilience_mask = strain <= yield_strain
        if np.any(resilience_mask):
            ax.fill_between(strain[resilience_mask], 0, stress[resilience_mask], 
                           alpha=0.3, color='green', label=f'Resilience ({self.results["modulus_of_resilience"]:.3f} MJ/m³)')
        
        # Highlight toughness area (entire curve)
        ax.fill_between(strain, 0, stress, alpha=0.2, color='red', 
                       label=f'Toughness ({self.results["modulus_of_toughness"]:.2f} MJ/m³)')
        
        ax.set_xlabel('Engineering Strain (mm/mm)')
        ax.set_ylabel('Engineering Stress (MPa)')
        ax.set_title('Modulus of Toughness and Resilience')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.canvas.draw()
    
    def plot_engineering_vs_true(self):
        """Plot engineering and true stress-strain curves"""
        if self.data is None or self.results is None:
            messagebox.showerror("Error", "Please import data and run analysis first!")
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        eng_strain = self.analyzer.engineering_strain
        eng_stress = self.analyzer.engineering_stress
        true_strain = self.analyzer.true_strain
        true_stress = self.analyzer.true_stress
        
        ax.plot(eng_strain, eng_stress, 'b-', linewidth=2, label='Engineering Stress-Strain')
        ax.plot(true_strain, true_stress, 'r-', linewidth=2, label='True Stress-Strain')
        
        # Annotate ultimate tensile strength points
        ult_strain_eng = self.results['ultimate_strain']
        ult_stress_eng = self.results['ultimate_strength']
        ult_idx = self.results['ultimate_stress_index']
        
        # Use same UTS calculation as Hollomon plot for consistency
        uts_idx_true = np.argmax(true_stress)
        ult_strain_true = true_strain[uts_idx_true]
        ult_stress_true = true_stress[uts_idx_true]
        
        # Plot ultimate points
        ax.plot(ult_strain_eng, ult_stress_eng, 'bo', markersize=8, 
               label=f'Engineering UTS ({ult_stress_eng:.1f} MPa)')
        ax.plot(ult_strain_true, ult_stress_true, 'ro', markersize=8, 
               label=f'True UTS ({ult_stress_true:.1f} MPa)')
        
        # Annotate fracture points (final data points)
        fracture_strain_eng = eng_strain.iloc[-1] if hasattr(eng_strain, 'iloc') else eng_strain[-1]
        fracture_stress_eng = eng_stress.iloc[-1] if hasattr(eng_stress, 'iloc') else eng_stress[-1]
        fracture_strain_true = true_strain.iloc[-1] if hasattr(true_strain, 'iloc') else true_strain[-1]
        fracture_stress_true = true_stress.iloc[-1] if hasattr(true_stress, 'iloc') else true_stress[-1]
        
        # Plot fracture points
        ax.plot(fracture_strain_eng, fracture_stress_eng, 'bs', markersize=8, 
               label=f'Engineering Fracture ({fracture_stress_eng:.1f} MPa)')
        ax.plot(fracture_strain_true, fracture_stress_true, 'rs', markersize=8, 
               label=f'True Fracture ({fracture_stress_true:.1f} MPa)')
        
        ax.set_xlabel('Strain (mm/mm)')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title('Engineering vs True Stress-Strain Curves')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.canvas.draw()
    
    def plot_ramberg_osgood(self):
        """Plot stress-strain curve with Ramberg-Osgood fit"""
        if self.data is None or self.results is None or 'ramberg_osgood' not in self.results:
            messagebox.showerror("Error", "Please import data and run analysis first!")
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        strain = self.analyzer.engineering_strain
        stress = self.analyzer.engineering_stress
        
        # Plot experimental data
        ax.plot(strain, stress, 'b-', linewidth=2, label='Experimental Data')
        
        # Get yield point for visualization
        yield_results = self._calculate_yield_strength() if hasattr(self, '_calculate_yield_strength') else self.analyzer._calculate_yield_strength()
        yield_idx = yield_results['yield_stress_index']
        yield_strain = yield_results['yield_strain']
        yield_stress = yield_results['yield_strength']
        
        # Ramberg-Osgood variant fit using ε = σ/E + (σ/K)^n
        ro_params = self.results['ramberg_osgood']
        E = self.results['elastic_modulus']
        K = ro_params['K']
        n = ro_params['n']
        r_squared = ro_params['r_squared']
        
        if K > 0 and n > 0:
            # Create stress range for fitting - extend from origin to UTS
            uts_idx = np.argmax(stress)
            uts_stress = float(stress[uts_idx])  # Ensure we get exact UTS value
            stress_fit = np.linspace(0, uts_stress, 1000)
            
            # Calculate strain using Ramberg-Osgood variant equation: ε = σ/E + (σ/K)^n
            elastic_strain_fit = stress_fit / E
            plastic_strain_fit = (stress_fit / K) ** n
            strain_fit = elastic_strain_fit + plastic_strain_fit
            
            # Plot the fit line
            ax.plot(strain_fit, stress_fit, 'r--', linewidth=3, alpha=0.8,
                   label=f'Ramberg-Osgood Fit: ε = σ/{E/1000:.1f}k + (σ/{K:.1f})^{n:.3f} (R²={r_squared:.3f})')
        
        # Add data points at key locations for better visualization
        ax.plot(yield_strain, yield_stress, 'ro', markersize=8, 
               label=f'YS ({yield_stress:.0f} MPa)')
        
        # Mark UTS point
        uts_idx = np.argmax(stress)
        uts_strain = strain[uts_idx]
        uts_stress = stress[uts_idx]
        ax.plot(uts_strain, uts_stress, 'go', markersize=8,
               label=f'UTS ({uts_stress:.0f} MPa)')
        
        ax.set_xlabel('Engineering Strain (mm/mm)')
        ax.set_ylabel('Engineering Stress (MPa)')
        ax.set_title('Ramberg-Osgood Curve Fitting')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        self.canvas.draw()
    
    def plot_hollomon_equation(self):
        """Plot true stress-strain curve with Hollomon equation fit"""
        if self.data is None or self.results is None or 'hollomon' not in self.results:
            messagebox.showerror("Error", "Please import data and run analysis first!")
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        true_strain = self.analyzer.true_strain
        true_stress = self.analyzer.true_stress
        
        ax.plot(true_strain, true_stress, 'b-', linewidth=2, label='Experimental True Stress-Strain')
        
        # Hollomon equation fit
        hollomon_params = self.results['hollomon']
        K = hollomon_params['K']
        n = hollomon_params['n']
        
        if K > 0 and n > 0:
            # Create fit line using Hollomon equation: σₜ = K × εₜⁿ
            # Extend from very small positive value (closer to origin) to UTS point
            uts_idx = np.argmax(true_stress)
            uts_true_strain = float(true_strain[uts_idx])  # Ensure we get exact UTS strain value
            strain_fit = np.linspace(1e-6, uts_true_strain, 1000)  # Start from much smaller value closer to origin
            stress_fit = K * (strain_fit ** n)
            
            ax.plot(strain_fit, stress_fit, 'r--', linewidth=2, 
                   label=f'Hollomon Fit: σₜ = {K:.1f} × εₜ^{n:.3f} (R²={hollomon_params["r_squared"]:.4f})')
        
        # Mark Yield Point on true stress-strain curve
        yield_results = self._calculate_yield_strength() if hasattr(self, '_calculate_yield_strength') else self.analyzer._calculate_yield_strength()
        yield_idx = yield_results['yield_stress_index']
        yield_true_strain = true_strain[yield_idx]
        yield_true_stress = true_stress[yield_idx]
        ax.plot(yield_true_strain, yield_true_stress, 'ro', markersize=8,
               label=f'True YS ({yield_true_stress:.0f} MPa)')
        
        # Mark UTS point on true stress-strain curve
        uts_idx = np.argmax(true_stress)
        uts_true_strain = true_strain[uts_idx]
        uts_true_stress = true_stress[uts_idx]
        ax.plot(uts_true_strain, uts_true_stress, 'go', markersize=8,
               label=f'True UTS ({uts_true_stress:.0f} MPa)')
        
        ax.set_xlabel('True Strain (mm/mm)')
        ax.set_ylabel('True Stress (MPa)')
        ax.set_title('Hollomon Equation Curve Fitting')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.canvas.draw()
    
    def clear_plot(self):
        """Clear the current plot"""
        self.figure.clear()
        self.canvas.draw()
    
    def export_plot(self):
        """Export the current plot to an image file"""
        if self.figure.get_axes():
            file_path = filedialog.asksaveasfilename(
                title="Export Plot",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("JPEG files", "*.jpg"),
                    ("TIFF files", "*.tiff"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                try:
                    # Save the figure with high DPI and tight bounding box
                    self.figure.savefig(file_path, dpi=300, bbox_inches='tight', 
                                      facecolor='white', edgecolor='none')
                    messagebox.showinfo("Success", f"Plot exported successfully to:\n{file_path}")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export plot:\n{str(e)}")
        else:
            messagebox.showwarning("Warning", "No plot to export. Please create a plot first.")
    
    def export_all_plots(self):
        """Export all available plots to image files"""
        if self.data is None or self.results is None:
            messagebox.showerror("Error", "Please import data and run analysis first!")
            return
        
        # Ask user to select a directory
        directory = filedialog.askdirectory(
            title="Select Directory to Export All Plots"
        )
        
        if directory:
            try:
                exported_files = []
                
                # List of available plots
                plots = [
                    ("Stress-Strain with Properties", self.plot_stress_strain_properties, "stress_strain_properties"),
                    ("Stress-Strain with Toughness/Resilience", self.plot_toughness_resilience, "toughness_resilience"),
                    ("Engineering vs True Stress-Strain", self.plot_engineering_vs_true, "engineering_vs_true")
                ]
                
                # Add Ramberg-Osgood plot if available
                if ('ramberg_osgood' in self.results and 
                    self.results['ramberg_osgood']['r_squared'] > 0.1):
                    plots.append(("Ramberg-Osgood Fit", self.plot_ramberg_osgood, "ramberg_osgood"))
                
                # Add Hollomon equation plot if available
                if ('hollomon' in self.results and 
                    self.results['hollomon']['r_squared'] > 0.1):
                    plots.append(("Hollomon Equation Fit", self.plot_hollomon_equation, "hollomon_equation"))
                
                # Generate and save each plot
                for plot_name, plot_function, filename in plots:
                    # Generate the plot
                    plot_function()
                    
                    # Save the plot
                    file_path = os.path.join(directory, f"{filename}.png")
                    self.figure.savefig(file_path, dpi=300, bbox_inches='tight', 
                                      facecolor='white', edgecolor='none')
                    exported_files.append(file_path)
                
                # Show success message
                files_list = "\n".join([os.path.basename(f) for f in exported_files])
                messagebox.showinfo("Success", 
                    f"All plots exported successfully to:\n{directory}\n\nFiles created:\n{files_list}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export plots:\n{str(e)}")
    
    def clear_results(self):
        """Clear all results"""
        self.data = None
        self.analyzer = None
        self.results = None
        
        self.file_label.config(text="No file loaded")
        self.rows_label.config(text="-")
        
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        self.clear_plot()
    
    def export_results(self):
        """Export analysis results to file"""
        if self.results is None:
            messagebox.showerror("Error", "No results to export!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("Tensile Test Analysis Results\n")
                    f.write("=" * 40 + "\n\n")
                    
                    f.write("Test Information:\n")
                    f.write(f"Maximum Force: {self.results['max_force']:.3f} kN\n")
                    f.write(f"Maximum Displacement: {self.results['max_displacement']:.3f} mm\n")
                    f.write(f"Test Duration: {self.results['test_time']:.2f} s\n\n")
                    
                    f.write("Mechanical Properties:\n")
                    f.write(f"Yield Strength (0.2% offset): {self.results['yield_strength']:.2f} MPa\n")
                    f.write(f"Ultimate Tensile Strength: {self.results['ultimate_strength']:.2f} MPa\n")
                    f.write(f"Fracture Strength: {self.results['fracture_strength']:.2f} MPa\n")
                    f.write(f"Elastic Modulus: {self.results['elastic_modulus']/1000:.2f} GPa\n")
                    f.write(f"Uniform Elongation: {self.results['uniform_elongation']:.2f} %\n")
                    f.write(f"Elongation at Fracture: {self.results['elongation_at_fracture']:.2f} %\n")
                    
                    # Handle elongation after fracture - may be None
                    elongation_after_fracture = self.results.get('elongation_after_fracture')
                    if elongation_after_fracture is not None:
                        f.write(f"Elongation after Fracture: {elongation_after_fracture:.2f} %\n")
                    else:
                        f.write("Elongation after Fracture: - (data not available)\n")
                    
                    # Handle reduction of area - may be None
                    reduction_of_area = self.results.get('reduction_of_area')
                    if reduction_of_area is not None:
                        f.write(f"Reduction of Area: {reduction_of_area:.2f} %\n\n")
                    else:
                        f.write("Reduction of Area: - (data not available)\n\n")
                    
                    f.write("Energy Properties:\n")
                    f.write(f"Modulus of Toughness: {self.results['modulus_of_toughness']:.2f} MJ/m³\n")
                    f.write(f"Modulus of Resilience: {self.results['modulus_of_resilience']:.3f} MJ/m³\n\n")
                    
                    if 'ramberg_osgood' in self.results:
                        f.write("Ramberg-Osgood Parameters:\n")
                        f.write(f"K (Strength Coefficient): {self.results['ramberg_osgood']['K']:.1f} MPa\n")
                        f.write(f"n (Hardening Exponent): {self.results['ramberg_osgood']['n']:.3f}\n")
                        f.write(f"R² (Fit Quality): {self.results['ramberg_osgood']['r_squared']:.4f}\n")
                        
                        # Add elastic and plastic components if available
                        if 'elastic_component_at_yield' in self.results['ramberg_osgood']:
                            f.write(f"Elastic Component at Yield: {self.results['ramberg_osgood']['elastic_component_at_yield']:.6f} mm/mm\n")
                            f.write(f"Plastic Component at Yield: {self.results['ramberg_osgood']['plastic_component_at_yield']:.6f} mm/mm\n")
                        f.write("\n")
                    
                    if 'hollomon' in self.results:
                        f.write("Hollomon Equation Parameters:\n")
                        f.write(f"K (Strength Coefficient): {self.results['hollomon']['K']:.2f} MPa\n")
                        f.write(f"n (Strain Hardening Exponent): {self.results['hollomon']['n']:.3f}\n")
                        f.write(f"R² (Fit Quality): {self.results['hollomon']['r_squared']:.4f}\n")
                
                messagebox.showinfo("Success", f"Results exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")
    
    def show_formulae(self):
        """Show formulae used for mechanical property calculations"""
        formulae_text = """Mechanical Properties Calculation Formulae
(According to ASTM E8 and E112 Standards)

═══════════════════════════════════════════════════════════════

TEST INFORMATION:

Maximum Force:
Fₘₐₓ = max(F(t))
Where: F(t) = Force as function of time

Maximum Displacement:
δₘₐₓ = max(δ(t))
Where: δ(t) = Displacement as function of time

Test Duration:
tₜₑₛₜ = tₘₐₓ - tₘᵢₙ
Where: t = Time values from data

═══════════════════════════════════════════════════════════════

STRESS AND STRAIN CALCULATIONS:

Engineering Stress (σ):
σ = F / A₀
Where: F = Applied force (N), A₀ = Original cross-sectional area (mm²)

Engineering Strain (ε):
ε = ΔL / L₀ = (L - L₀) / L₀
Where: L = Current length, L₀ = Original gauge length

True Stress (σₜ):
σₜ = σ(1 + ε) = F(1 + ε) / A₀

True Strain (εₜ):
εₜ = ln(1 + ε) = ln(L / L₀)

Cross-sectional Area (Circular):
A₀ = π(d₀/2)² = πr₀²
Where: d₀ = Original diameter, r₀ = Original radius

═══════════════════════════════════════════════════════════════

MECHANICAL PROPERTIES:

Yield Strength (YS) (σᵧ) - 0.2% Offset Method:

Ultimate Tensile Strength (UTS) = σₘₐₓ = Maximum stress

Fracture Strength (σf) = Stress at final data point (fracture)

Elastic Modulus (E) - Secant Method:
E = Δσ / Δε (in middle 50% of elastic region)

═══════════════════════════════════════════════════════════════

ELONGATION CALCULATIONS:

Uniform Elongation (%) = εᵤₜₛ × 100
Where: εᵤₜₛ = Engineering strain at ultimate tensile strength

Elongation at Fracture (%) = εf × 100
Where: εf = Engineering strain at maximum strain

Elongation after Fracture (%) = ((Lf - L₀) / L₀) × 100
Where: Lf = Final gauge length, L₀ = Original gauge length

Reduction of Area (RA) (%) = ((A₀ - Af) / A₀) × 100
Where: A₀ = Original area, Af = Final area at fracture

═══════════════════════════════════════════════════════════════

ENERGY CALCULATIONS:

Modulus of Resilience (Ur):
Ur = ∫₀^εᵧ σ dε ≈ σᵧ²/(2E)
• Area under stress-strain curve up to yield point

Modulus of Toughness (Ut):
Ut = ∫₀^εf σ dε
• Total area under stress-strain curve to fracture

═══════════════════════════════════════════════════════════════

RAMBERG-OSGOOD EQUATION:

Standard Form:
ε = σ/E + (σ/K)ⁿ
Where: ε = Total strain, σ = Stress, E = Young's modulus,
       K = Strength coefficient (MPa), n = Hardening exponent

═══════════════════════════════════════════════════════════════

HOLLOMON EQUATION:

True Stress-Strain Relationship:
σₜ = K × εₜⁿ
Where: σₜ = True stress, εₜ = True strain
       K = Strength coefficient, n = Strain hardening exponent

Logarithmic Form:
ln(σₜ) = ln(K) + n × ln(εₜ)

═══════════════════════════════════════════════════════════════

NOTES:
• All calculations follow ASTM E8 (tensile testing) standards
• Elastic modulus follows ASTM E112 recommendations
• Units: Stress in MPa, Strain dimensionless, Energy in MJ/m³
• Smoothing applied using Savitzky-Golay filter to reduce noise
• Hollomon equation provides better fit for true stress-strain data
• Ramberg-Osgood equation useful for engineering stress-strain data"""
        
        # Create a new window to display formulae
        formulae_window = tk.Toplevel(self.root)
        formulae_window.title("Mechanical Properties Formulae")
        formulae_window.geometry("800x700")
        formulae_window.transient(self.root)
        formulae_window.grab_set()
        
        # Create text widget with scrollbar
        frame = ttk.Frame(formulae_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Text widget
        text_widget = tk.Text(frame, wrap=tk.WORD, font=("Consolas", 10))
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Insert formulae text
        text_widget.insert(tk.END, formulae_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Close button
        close_button = ttk.Button(formulae_window, text="Close", 
                                 command=formulae_window.destroy)
        close_button.pack(pady=10)
        
        # Center the window
        formulae_window.update_idletasks()
        x = (formulae_window.winfo_screenwidth() // 2) - (formulae_window.winfo_width() // 2)
        y = (formulae_window.winfo_screenheight() // 2) - (formulae_window.winfo_height() // 2)
        formulae_window.geometry(f"+{x}+{y}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """Tensile Analyzer Program
        
Version: 1.5
Compliance: ASTM E8, ASTM E112

This program analyzes tensile test data to mechanical properties including:
• Yield Strength (YS) (0.2% offset method)
• Ultimate Tensile Strength (UTS) 
• Elastic Modulus (EM) (from middle 50% of elastic region)
• Elongation at and after fracture
• Reduction of Area (RA)
• Modulus of Toughness and Resilience
• Ramberg-Osgood curve fitting parameters
• Hollomon curve fitting parameters

Created: August 2025"""
        
        messagebox.showinfo("About", about_text)


def main():
    root = tk.Tk()
    app = TensileAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

# Make classes available for import when used as a module
__all__ = ['DataLoader', 'TensileAnalyzer', 'TensileAnalysisGUI']
