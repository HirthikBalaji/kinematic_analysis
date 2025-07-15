# import streamlit as st
# import numpy as np
# import math
#
# st.set_page_config(page_title="Freudenstein Equation Solver", layout="centered")
# st.title("🔧 Freudenstein Equation & Link Length Calculator")
# st.markdown("A tool for analyzing 4-bar mechanisms using angle input and the Freudenstein equation.")
#
# # Sidebar input for fixed link length
# st.sidebar.header("Link Parameters")
# d = st.sidebar.number_input("Fixed Link Length (d)", min_value=0.01, value=100.0)
#
#
# # Function to input angle pair and convert to radians
# def angle_input(index):
#     col1, col2 = st.columns(2)
#     with col1:
#         theta = st.number_input(f"θ{index} (degrees)", value=30.0 * index, key=f"theta{index}")
#     with col2:
#         phi = st.number_input(f"φ{index} (degrees)", value=20.0 * index, key=f"phi{index}")
#     return math.radians(theta), math.radians(phi)
#
#
# st.subheader("🔢 Enter 3 Sets of θ and φ Angles")
#
# theta1, phi1 = angle_input(1)
# theta2, phi2 = angle_input(2)
# theta3, phi3 = angle_input(3)
#
# # Create matrix A and vector B
# A = [
#     [math.cos(phi1), -math.cos(theta1), 1],
#     [math.cos(phi2), -math.cos(theta2), 1],
#     [math.cos(phi3), -math.cos(theta3), 1]
# ]
#
# B = [
#     math.cos(theta1 - phi1),
#     math.cos(theta2 - phi2),
#     math.cos(theta3 - phi3)
# ]
#
# try:
#     # Solve for K1, K2, K3
#     K1, K2, K3 = np.linalg.solve(A, B)
#
#     # Calculate link lengths
#     a = d / K1
#     b = d / K2
#     c_squared = d ** 2 - a ** 2 - b ** 2 + 2 * a * b * K3
#
#     st.subheader("✅ Freudenstein Coefficients")
#     st.write(f"K1 = {K1:.4f}")
#     st.write(f"K2 = {K2:.4f}")
#     st.write(f"K3 = {K3:.4f}")
#
#     st.subheader("📐 Link Lengths")
#     st.write(f"a (Input link)  = {a:.4f}")
#     st.write(f"b (Output link) = {b:.4f}")
#
#     if c_squared < 0:
#         st.error("⚠️ Invalid geometry: negative value under square root for link c.")
#     else:
#         c = math.sqrt(c_squared)
#         st.write(f"c (Coupler link) = {c:.4f}")
#
#     st.write(f"d (Fixed link)   = {d:.4f}")
#
# except np.linalg.LinAlgError:
#     st.error("❌ Could not solve equations. Make sure the input angles are not linearly dependent.")
import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go

st.set_page_config(page_title="Freudenstein Linkage Animator", layout="centered")
st.title("🔧 4-Bar Linkage Solver & Animator")

# Sidebar: fixed link
st.sidebar.header("Link Length")
d = st.sidebar.number_input("Fixed link length d", min_value=1.0, value=100.0)

# Function to get angle inputs in radians
def angle_input(index, default_theta, default_phi):
    c1, c2 = st.columns(2)
    with c1:
        theta = st.number_input(f"θ{index} (°)", value=default_theta, key=f"theta{index}")
    with c2:
        phi = st.number_input(f"φ{index} (°)", value=default_phi, key=f"phi{index}")
    return math.radians(theta), math.radians(phi)

st.subheader("🔢 Angle Sets")
t1, p1 = angle_input(1, 30, 10)
t2, p2 = angle_input(2, 60, 20)
t3, p3 = angle_input(3, 90, 30)

# Solve for K values
A = [
    [math.cos(p1), -math.cos(t1), 1],
    [math.cos(p2), -math.cos(t2), 1],
    [math.cos(p3), -math.cos(t3), 1]
]
B = [math.cos(t1 - p1), math.cos(t2 - p2), math.cos(t3 - p3)]
try:
    K1, K2, K3 = np.linalg.solve(A, B)
    a = d / K1             # input
    b = d / K2             # output
    c2 = d**2 - a**2 - b**2 + 2*a*b*K3
    if c2 < 0:
        st.error("Invalid linkage geometry (c² < 0). Adjust inputs.")
        st.stop()
    c = math.sqrt(c2)
except Exception as e:
    st.error(f"Cannot solve: {e}")
    st.stop()

# Display link data
st.subheader("🔎 Link Lengths")
st.write(f"a = {a:.2f}, b = {b:.2f}, c = {c:.2f}, d = {d:.2f}")

# Animation controls
st.subheader("🎥 Animate Mechanism")
num_frames = st.slider("Number of frames", 20, 200, 60)
input_start = st.number_input("Start θ (°)", value=0.0)
input_end = st.number_input("End θ (°)", value=360.0)

thetas = np.linspace(math.radians(input_start), math.radians(input_end), num_frames)

# Compute positions for each frame
px, py, qx, qy, cx, cy = [], [], [], [], [], []
for th in thetas:
    # Coordinates: O fixed at (0,0), Ground link OD with length d along x-axis
    ox, oy = 0, 0
    dx, dy = d, 0
    # Input crank end P
    px.append( a * math.cos(th) )
    py.append( a * math.sin(th) )
    # Cosine law for φ (output) from Freudenstein rearrangement:
    expr = (a*px[-1] + d*dx + b**2 - a**2 - d**2)/(2*b)
    # Actually directly compute φ using Freudenstein eqn:
    val = K1 * math.cos(th) - K2 * math.cos(th) + K3
    val = max(-1, min(1, val))  # Clamp to [-1, 1]
    phi = math.acos(val) # approximate
    qx.append( d - b*math.cos(phi) )
    qy.append( b*math.sin(phi) )
    # Coupler C is at joint between P and Q:
    cx.append(px[-1] + (qx[-1]-px[-1]) * (c / math.hypot(qx[-1]-px[-1], qy[-1]-py[-1])))
    cy.append(py[-1] + (qy[-1]-py[-1]) * (c / math.hypot(qx[-1]-px[-1], qy[-1]-py[-1])))

# Build Plotly animation
fig = go.Figure(
    data=[
        go.Scatter(x=[ox, px[0], cx[0], qx[0], dx],
                   y=[oy, py[0], cy[0], qy[0], dy],
                   mode="lines+markers",
                   marker=dict(size=8),
                   line=dict(width=4))
    ],
    layout=go.Layout(
        xaxis=dict(range=[-1.1*d, 1.1*d], zeroline=False),
        yaxis=dict(range=[-1.1*d, 1.1*d], zeroline=False),
        title="4-Bar Mechanism",
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, {"frame": {"duration": 50, "redraw": True},
                                                     "fromcurrent": True}])])]
    ),
    frames=[go.Frame(
        data=[go.Scatter(x=[ox, px[i], cx[i], qx[i], dx],
                         y=[oy, py[i], cy[i], qy[i], dy],
                         mode="lines+markers")]
    ) for i in range(num_frames)]
)

st.plotly_chart(fig, use_container_width=True)
