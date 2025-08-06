# Tensile Analyzer
This program processes and analyzes tensile test data in accordance with ASTM standards. Such data are often analyzed using spreadsheets and lack a standard methodology, which is prone to user error and tedious. This program aims to reduce such user errors and improve the accuracy and precision of the calculated mechanical properties more efficiently. 

The user can import their data file from a universal testing machine (UTM) containing force, displacement, and strain data. The program then analyzes this data to determine key mechanical properties such as ultimate tensile strength (UTS), yield strength (YS), elongation, reduction of area (RA), elastic modulus (EM), and monotonic properties in accordance with ASTM E8/E8M[^1] and ASTM E111[^2] requirements.

The following plots can be generated and exported:
- Stress-strain plot with annotated YS, UTS, and EM.
- Stress-strain plot with highlighted modulus of resilience and modulus of toughness areas.
- Engineering stress-strain plot overlayed with true stress-strain plot.
- Ramberg-Osgood curve fitting and parameters.

## Table of Contents

## Features

## Installation Instructions
To run the Tensile Analyzer program, the following prerequisite Python libraries must be installed:
```
pip install tkinter, pandas, numpy, matplotlib, scipy, os, warnings
```
After installing these prerequisites, the tensile_analyzer_XXX.py (where XXX is the relevant version of the program) file can be run in your IDE of choice, and the GUI will appear.

## Data File Format
Data files exported from the UTM testing software can be imported into the Tensile Analyzer program in the following formats:
- CSV (.csv)
- XLSM (.xlsm)
- TXT (.txt)

The data file must be formatted to contain data for columns in the format and associated metric units defined below.

### Circular cross-section specimens
| Time (s) | Extension (mm)	| Load (kN)	| Strain (mm/mm)	| Original Length (mm)	| Original Diameter (mm)	| Final Length (mm)	| Final Diameter (mm) |
### Rectangular cross-section specimens
| Time (s) | Extension (mm)	| Load (kN)	| Strain (mm/mm)	| Original Length (mm)	| Original Thickness (mm)	| Original Width (mm)	| Final Length (mm)	| Final Thickness (mm) | Final Width (mm) |

## Examples and Testing

## License and Citation

## References

[^1]: ASTM International. Standard Test Methods for Tension Testing of Metallic Materials. ASTM E8/E8M, 2024.
[^2]: ASTM International. Standard Test Method for Young’s Modulus, Tangent Modulus, and Chord Modulus. ASTM E111, 2017.
