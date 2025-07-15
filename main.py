import math
import numpy as np

print("Freudenstein Equation Solver + Link Length Calculator")

def get_angles():
    theta = float(input("Enter θ (in degrees): "))
    phi = float(input("Enter φ (in degrees): "))
    return math.radians(theta), math.radians(phi)

print("Enter angle pair 1:")
theta1, phi1 = get_angles()

print("Enter angle pair 2:")
theta2, phi2 = get_angles()

print("Enter angle pair 3:")
theta3, phi3 = get_angles()

A = [
    [math.cos(phi1), -math.cos(theta1), 1],
    [math.cos(phi2), -math.cos(theta2), 1],
    [math.cos(phi3), -math.cos(theta3), 1]
]

B = [
    math.cos(theta1 - phi1),
    math.cos(theta2 - phi2),
    math.cos(theta3 - phi3)
]

K1, K2, K3 = np.linalg.solve(A, B)

d = float(input("\nEnter the fixed link length d (in mm or any unit): "))

a = d / K1
b = d / K2
c_squared = d**2 - a**2 - b**2 + 2 * a * b * K3

if c_squared < 0:
    print("\nError: Negative value under square root. Check input angles.")
    c = None
else:
    c = math.sqrt(c_squared)

print("\nResults:")
print(f"K1 = {K1:.4f}")
print(f"K2 = {K2:.4f}")
print(f"K3 = {K3:.4f}")

print(f"\na (input link)  = {a:.4f}")
print(f"b (output link) = {b:.4f}")
if c is not None:
    print(f"c (coupler)     = {c:.4f}")
else:
    print("c (coupler)     = Cannot compute (invalid geometry)")

print(f"d (fixed link)  = {d:.4f}")
