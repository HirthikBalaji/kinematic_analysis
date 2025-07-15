import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math

class FourBarFunctionGenerator:
    """
    Four-bar linkage function generator using Freudenstein's equation.
    
    Freudenstein's equation: K1*cos(φ) - K2*cos(ψ) + K3 = cos(ψ - φ)
    where:
    - K1 = d/a
    - K2 = d/c  
    - K3 = (a² - b² + c² + d²)/(2ac)
    - φ = input angle (crank angle)
    - ψ = output angle (rocker angle)
    """
    
    def __init__(self, a=1.0, b=None, c=None, d=None):
        """
        Initialize four-bar linkage parameters.
        
        Parameters:
        a: input link length (crank)
        b: coupler link length
        c: output link length (rocker)
        d: fixed link length (ground)
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        if all(x is not None for x in [a, b, c, d]):
            self.update_constants()
    
    def update_constants(self):
        """Update Freudenstein constants K1, K2, K3"""
        self.K1 = self.d / self.a
        self.K2 = self.d / self.c
        self.K3 = (self.a**2 - self.b**2 + self.c**2 + self.d**2) / (2 * self.a * self.c)
    
    def freudenstein_equation(self, phi, psi):
        """
        Evaluate Freudenstein's equation.
        
        Parameters:
        phi: input angle (radians)
        psi: output angle (radians)
        
        Returns:
        Value of Freudenstein's equation (should be 0 for valid positions)
        """
        return (self.K1 * np.cos(phi) - self.K2 * np.cos(psi) + 
                self.K3 - np.cos(psi - phi))
    
    def solve_output_angle(self, phi):
        """
        Solve for output angle given input angle.
        
        Parameters:
        phi: input angle (radians)
        
        Returns:
        psi: output angle (radians)
        """
        def equation(psi):
            return self.freudenstein_equation(phi, psi)
        
        # Try multiple initial guesses to find all solutions
        solutions = []
        initial_guesses = np.linspace(0, 2*np.pi, 8)
        
        for guess in initial_guesses:
            try:
                sol = fsolve(equation, guess)[0]
                if abs(equation(sol)) < 1e-10:  # Check if solution is valid
                    # Normalize to [0, 2π)
                    sol = sol % (2 * np.pi)
                    # Check if this solution is already found
                    if not any(abs(sol - s) < 1e-6 for s in solutions):
                        solutions.append(sol)
            except:
                continue
        
        return solutions
    
    def synthesize_three_point(self, phi_values, psi_values):
        """
        Synthesize four-bar linkage for three precision points using Freudenstein's equation.
        
        Parameters:
        phi_values: list of 3 input angles (radians)
        psi_values: list of 3 output angles (radians)
        
        Returns:
        Dictionary with linkage parameters {a, b, c, d}
        """
        if len(phi_values) != 3 or len(psi_values) != 3:
            raise ValueError("Exactly 3 precision points required")
        
        phi1, phi2, phi3 = phi_values
        psi1, psi2, psi3 = psi_values
        
        # Set up system of equations from Freudenstein's equation
        # K1*cos(φ) - K2*cos(ψ) + K3 = cos(ψ - φ)
        
        # Coefficient matrix for [K1, K2, K3]
        A = np.array([
            [np.cos(phi1), -np.cos(psi1), 1],
            [np.cos(phi2), -np.cos(psi2), 1],
            [np.cos(phi3), -np.cos(psi3), 1]
        ])
        
        # Right-hand side
        b = np.array([
            np.cos(psi1 - phi1),
            np.cos(psi2 - phi2),
            np.cos(psi3 - phi3)
        ])
        
        # Solve for K1, K2, K3
        try:
            K_values = np.linalg.solve(A, b)
            K1, K2, K3 = K_values
        except np.linalg.LinAlgError:
            raise ValueError("System is singular - precision points may be collinear")
        
        # Given a = 1 (normalized), solve for b, c, d
        a = self.a  # Use existing value or default
        
        # From K1 = d/a and K2 = d/c
        d = K1 * a
        c = d / K2
        
        # From K3 = (a² - b² + c² + d²)/(2ac)
        # Solve for b: b² = a² + c² + d² - 2ac*K3
        b_squared = a**2 + c**2 + d**2 - 2*a*c*K3
        
        if b_squared < 0:
            raise ValueError("Invalid solution - b² < 0")
        
        b = np.sqrt(b_squared)
        
        # Update object parameters
        self.a, self.b, self.c, self.d = a, b, c, d
        self.update_constants()
        
        return {'a': a, 'b': b, 'c': c, 'd': d}
    
    def analyze_motion(self, phi_range=None, num_points=100):
        """
        Analyze the complete motion of the four-bar linkage.
        
        Parameters:
        phi_range: tuple (start, end) for input angle range in radians
        num_points: number of analysis points
        
        Returns:
        Dictionary with motion analysis results
        """
        if phi_range is None:
            phi_range = (0, 2*np.pi)
        
        phi_values = np.linspace(phi_range[0], phi_range[1], num_points)
        psi_values = []
        valid_positions = []
        
        for phi in phi_values:
            solutions = self.solve_output_angle(phi)
            if solutions:
                psi_values.append(solutions[0])  # Take first solution
                valid_positions.append(True)
            else:
                psi_values.append(np.nan)
                valid_positions.append(False)
        
        return {
            'phi': phi_values,
            'psi': np.array(psi_values),
            'valid': valid_positions,
            'transmission_angle': self.calculate_transmission_angle(phi_values, psi_values)
        }
    
    def calculate_transmission_angle(self, phi_values, psi_values):
        """Calculate transmission angle for given positions"""
        # This is a simplified calculation
        # In practice, you'd need full kinematic analysis
        return np.ones_like(phi_values) * np.pi/2  # Placeholder
    
    def plot_function(self, phi_range=None, num_points=100):
        """
        Plot the input-output function relationship.
        
        Parameters:
        phi_range: tuple (start, end) for input angle range
        num_points: number of plot points
        """
        results = self.analyze_motion(phi_range, num_points)
        
        plt.figure(figsize=(10, 6))
        
        # Plot function relationship
        plt.subplot(1, 2, 1)
        valid_mask = ~np.isnan(results['psi'])
        plt.plot(np.degrees(results['phi'][valid_mask]), 
                np.degrees(results['psi'][valid_mask]), 'b-', linewidth=2)
        plt.xlabel('Input Angle φ (degrees)')
        plt.ylabel('Output Angle ψ (degrees)')
        plt.title('Four-Bar Function Generation')
        plt.grid(True, alpha=0.3)
        
        # Plot linkage parameters
        plt.subplot(1, 2, 2)
        plt.text(0.1, 0.8, f'Link Parameters:', fontsize=12, fontweight='bold')
        plt.text(0.1, 0.7, f'a (crank): {self.a:.3f}', fontsize=10)
        plt.text(0.1, 0.6, f'b (coupler): {self.b:.3f}', fontsize=10)
        plt.text(0.1, 0.5, f'c (rocker): {self.c:.3f}', fontsize=10)
        plt.text(0.1, 0.4, f'd (ground): {self.d:.3f}', fontsize=10)
        plt.text(0.1, 0.2, f'Freudenstein Constants:', fontsize=12, fontweight='bold')
        plt.text(0.1, 0.1, f'K1: {self.K1:.3f}', fontsize=10)
        plt.text(0.1, 0.0, f'K2: {self.K2:.3f}', fontsize=10)
        plt.text(0.1, -0.1, f'K3: {self.K3:.3f}', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(-0.2, 1)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
if __name__ == "__main__":
    # Create function generator
    fg = FourBarFunctionGenerator()
    
    print("Four-Bar Function Generation with Freudenstein's Equation")
    print("=" * 55)
    
    # Example 1: Synthesis for three precision points
    print("\nExample 1: Three-Point Synthesis")
    print("-" * 30)
    
    # Define three precision points (φ, ψ) in radians
    phi_points = [np.pi/6, np.pi/3, np.pi/2]  # 30°, 60°, 90°
    psi_points = [np.pi/4, np.pi/3, 5*np.pi/12]  # 45°, 60°, 75°
    
    print(f"Precision points:")
    for i, (phi, psi) in enumerate(zip(phi_points, psi_points)):
        print(f"  Point {i+1}: φ = {np.degrees(phi):.1f}°, ψ = {np.degrees(psi):.1f}°")
    
    try:
        # Synthesize linkage
        params = fg.synthesize_three_point(phi_points, psi_points)
        
        print(f"\nSynthesized linkage parameters:")
        print(f"  a (crank): {params['a']:.4f}")
        print(f"  b (coupler): {params['b']:.4f}")
        print(f"  c (rocker): {params['c']:.4f}")
        print(f"  d (ground): {params['d']:.4f}")
        
        print(f"\nFreudenstein constants:")
        print(f"  K1: {fg.K1:.4f}")
        print(f"  K2: {fg.K2:.4f}")
        print(f"  K3: {fg.K3:.4f}")
        
        # Verify precision points
        print(f"\nVerification at precision points:")
        for i, (phi, psi) in enumerate(zip(phi_points, psi_points)):
            error = fg.freudenstein_equation(phi, psi)
            print(f"  Point {i+1}: Freudenstein error = {error:.2e}")
        
    except Exception as e:
        print(f"Synthesis failed: {e}")
    
    # Example 2: Forward analysis
    print(f"\n\nExample 2: Forward Kinematic Analysis")
    print("-" * 35)
    
    # Test forward analysis for a few input angles
    test_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    
    print("Input → Output mapping:")
    for phi in test_angles:
        psi_solutions = fg.solve_output_angle(phi)
        print(f"  φ = {np.degrees(phi):5.1f}° → ψ = ", end="")
        if psi_solutions:
            psi_str = ", ".join([f"{np.degrees(psi):5.1f}°" for psi in psi_solutions])
            print(psi_str)
        else:
            print("No solution")
    
    # Example 3: Function approximation
    print(f"\n\nExample 3: Function Approximation")
    print("-" * 32)
    
    # Approximate y = x² function over a limited range
    # Map input range [0, π/2] to approximate x² behavior
    x_vals = np.linspace(0, 1, 3)
    y_vals = x_vals**2
    
    # Convert to angles
    phi_approx = x_vals * np.pi/2  # Map [0,1] to [0,π/2]
    psi_approx = y_vals * np.pi/4  # Map [0,1] to [0,π/4]
    
    print(f"Approximating y = x² function:")
    print(f"Input range: [0, 1] → [0, {np.degrees(np.pi/2):.1f}°]")
    print(f"Output range: [0, 1] → [0, {np.degrees(np.pi/4):.1f}°]")
    
    try:
        fg_approx = FourBarFunctionGenerator()
        params_approx = fg_approx.synthesize_three_point(phi_approx, psi_approx)
        
        print(f"\nApproximation linkage parameters:")
        for key, value in params_approx.items():
            print(f"  {key}: {value:.4f}")
        
        # Test approximation quality
        print(f"\nApproximation quality check:")
        test_x = np.linspace(0, 1, 5)
        for x in test_x:
            phi_test = x * np.pi/2
            psi_actual = fg_approx.solve_output_angle(phi_test)
            if psi_actual:
                y_approx = psi_actual[0] / (np.pi/4)  # Convert back to [0,1]
                y_exact = x**2
                error = abs(y_approx - y_exact)
                print(f"  x = {x:.2f}: y_exact = {y_exact:.3f}, y_approx = {y_approx:.3f}, error = {error:.3f}")
    
    except Exception as e:
        print(f"Function approximation failed: {e}")
    
    # Plotting (if matplotlib is available)
    try:
        print(f"\nGenerating plots...")
        fg.plot_function()
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print(f"\nAnalysis complete!")
