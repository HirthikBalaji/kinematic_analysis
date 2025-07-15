# Four-Bar Function Generator

A Python implementation for designing and analyzing four-bar linkages using Freudenstein's equation for function generation problems.

## Overview

This library provides tools for synthesizing four-bar linkages that can approximate desired input-output relationships. It uses Freudenstein's equation, a fundamental relationship in mechanism design, to solve both synthesis and analysis problems.

## Features

- **Three-Point Synthesis**: Design linkages to pass through three precision points
- **Forward Kinematics**: Calculate output angles for given input angles
- **Function Approximation**: Approximate mathematical functions using mechanical linkages
- **Motion Analysis**: Analyze complete linkage motion and characteristics
- **Visualization**: Plot input-output relationships and linkage parameters

## Installation

```bash
git clone https://github.com/HirthikBalaji/kinematic_analysis.git
cd kinematic analysis
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
```

## Quick Start

```python
from fourbar_freudenstein import FourBarFunctionGenerator
import numpy as np

# Create a function generator
fg = FourBarFunctionGenerator()

# Define three precision points (input, output) in radians
phi_points = [np.pi/6, np.pi/3, np.pi/2]  # 30°, 60°, 90°
psi_points = [np.pi/4, np.pi/3, 5*np.pi/12]  # 45°, 60°, 75°

# Synthesize the linkage
params = fg.synthesize_three_point(phi_points, psi_points)
print(f"Linkage parameters: {params}")

# Analyze the resulting motion
results = fg.analyze_motion()

# Plot the function relationship
fg.plot_function()
```

## Theory

### Freudenstein's Equation

The core of this implementation is Freudenstein's equation:

```
K₁cos(φ) - K₂cos(ψ) + K₃ = cos(ψ - φ)
```

Where:
- **φ** = input angle (crank angle)
- **ψ** = output angle (rocker angle)
- **K₁ = d/a** (ratio of ground link to input link)
- **K₂ = d/c** (ratio of ground link to output link)
- **K₃ = (a² - b² + c² + d²)/(2ac)** (geometric constant)

### Link Nomenclature

```
    B -------- C
   /|          |
  / |          |
 /  |          |
A   |          D
    |__________|
    
a = AB (input link/crank)
b = BC (coupler link)
c = CD (output link/rocker)
d = AD (ground link)
```

## Examples

### Example 1: Basic Three-Point Synthesis

```python
import numpy as np
from fourbar_freudenstein import FourBarFunctionGenerator

# Create function generator
fg = FourBarFunctionGenerator()

# Define precision points
phi_points = [0.5, 1.0, 1.5]  # Input angles (radians)
psi_points = [0.3, 0.8, 1.2]  # Output angles (radians)

# Synthesize linkage
try:
    params = fg.synthesize_three_point(phi_points, psi_points)
    print("Synthesis successful!")
    print(f"a (crank): {params['a']:.3f}")
    print(f"b (coupler): {params['b']:.3f}")
    print(f"c (rocker): {params['c']:.3f}")
    print(f"d (ground): {params['d']:.3f}")
except ValueError as e:
    print(f"Synthesis failed: {e}")
```

### Example 2: Function Approximation

```python
# Approximate y = x² function over [0,1] range
x_vals = np.linspace(0, 1, 3)
y_vals = x_vals**2

# Map to angular ranges
phi_approx = x_vals * np.pi/2      # [0,1] → [0,π/2]
psi_approx = y_vals * np.pi/4      # [0,1] → [0,π/4]

# Synthesize approximating linkage
fg_approx = FourBarFunctionGenerator()
params = fg_approx.synthesize_three_point(phi_approx, psi_approx)

# Test approximation quality
test_x = 0.7
phi_test = test_x * np.pi/2
psi_solutions = fg_approx.solve_output_angle(phi_test)
if psi_solutions:
    y_approx = psi_solutions[0] / (np.pi/4)
    y_exact = test_x**2
    error = abs(y_approx - y_exact)
    print(f"At x={test_x}: exact={y_exact:.3f}, approx={y_approx:.3f}, error={error:.3f}")
```

### Example 3: Forward Analysis

```python
# Given a designed linkage, find output for various inputs
fg = FourBarFunctionGenerator(a=1.0, b=2.5, c=1.8, d=2.0)

input_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
for phi in input_angles:
    psi_solutions = fg.solve_output_angle(phi)
    print(f"φ = {np.degrees(phi):5.1f}° → ", end="")
    if psi_solutions:
        psi_str = ", ".join([f"{np.degrees(psi):5.1f}°" for psi in psi_solutions])
        print(f"ψ = {psi_str}")
    else:
        print("No solution (dead position)")
```

## API Reference

### FourBarFunctionGenerator Class

#### Constructor
```python
FourBarFunctionGenerator(a=1.0, b=None, c=None, d=None)
```

#### Methods

**`synthesize_three_point(phi_values, psi_values)`**
- Synthesizes linkage dimensions for three precision points
- **Parameters**: Lists of 3 input and output angles (radians)
- **Returns**: Dictionary with linkage parameters `{a, b, c, d}`

**`solve_output_angle(phi)`**
- Solves for output angle given input angle
- **Parameters**: Input angle φ (radians)
- **Returns**: List of output angles ψ (radians)

**`analyze_motion(phi_range=None, num_points=100)`**
- Analyzes complete linkage motion
- **Parameters**: Angular range tuple, number of analysis points
- **Returns**: Dictionary with motion data

**`plot_function(phi_range=None, num_points=100)`**
- Plots input-output relationship and linkage parameters
- **Parameters**: Angular range tuple, number of plot points

**`freudenstein_equation(phi, psi)`**
- Evaluates Freudenstein's equation
- **Parameters**: Input angle φ, output angle ψ (radians)
- **Returns**: Equation residual (should be 0 for valid positions)

## Applications

This library is useful for:

- **Mechanical Engineering**: Designing linkages for specific motion requirements
- **Robotics**: Creating mechanical function generators
- **Education**: Teaching mechanism synthesis and kinematic analysis
- **Research**: Prototyping and testing linkage designs

## Limitations

- Currently supports only planar four-bar linkages
- Three-point synthesis only (not five-point or optimization-based)
- Does not include dynamic analysis or force calculations
- Assumes ideal joints without friction or clearance

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

### Development Setup

```bash
git clone https://github.com/hirthik_balaji/kinematic_analysis.git
cd fourbar-function-generator
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Freudenstein, F. (1954). "An analytical approach to the design of four-link mechanisms." *Transactions of the ASME*, 76, 483-492.
2. Hartenberg, R. S., & Denavit, J. (1964). *Kinematic synthesis of linkages*. McGraw-Hill.
3. Norton, R. L. (2011). *Design of machinery: an introduction to the synthesis and analysis of mechanisms and machines*. McGraw-Hill.

## Acknowledgments

- Based on fundamental work by Ferdinand Freudenstein
- Inspired by classical mechanism design literature
- Built with Python scientific computing stack (NumPy, SciPy, Matplotlib)

---

**Author**: [Your Name]  
**Email**: [your.email@example.com]  
**Project Link**: [https://github.com/yourusername/fourbar-function-generator](https://github.com/yourusername/fourbar-function-generator)