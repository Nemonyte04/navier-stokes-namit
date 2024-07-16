import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Function to solve the Navier-Stokes equations with an optional obstacle
def solve_navier_stokes_with_obstacle(rey, u_w, steps, obstacle=False, obs_type="rectangular", obs_x0=0.4, obs_y0=0.4, obs_width=0.2, obs_height=0.2, obs_radius=0.1, obs_x_center=0.8, obs_y_center=0.7):
    # Setting up the domain
    Lx, Ly = 1, 1
    nx, ny = 201, 201
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x, y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    # Constants
    nu = 1 / rey

    # Setting up arrays
    psi = np.zeros([ny, nx])
    omega = np.zeros([ny, nx])
    U = np.zeros([ny, nx])
    V = np.zeros([ny, nx])
    U[ny - 1, 0:] = u_w

    if obstacle:
        if obs_type == "rectangular":
            # Rectangular obstacle dimensions and position
            obstacle_x0 = obs_x0
            obstacle_y0 = obs_y0
            obstacle_width = obs_width
            obstacle_height = obs_height

            # Identify grid points within the rectangular obstacle
            obstacle_mask = (x >= obstacle_x0) & (x <= obstacle_x0 + obstacle_width) & (y >= obstacle_y0) & (y <= obstacle_y0 + obstacle_height)

        elif obs_type == "circular":
            # Circular obstacle parameters
            obstacle_radius = obs_radius
            obstacle_x_center = obs_x_center
            obstacle_y_center = obs_y_center

            # Identify grid points within the circular obstacle
            obstacle_mask = (x - obstacle_x_center) ** 2 + (y - obstacle_y_center) ** 2 < obstacle_radius ** 2

        # Set boundary conditions for the obstacle
        U[obstacle_mask] = 0.0
        V[obstacle_mask] = 0.0

    # Time
    dt = 0.0005  # Time step

    for i in range(steps):
        omegaprime = np.copy(omega)

        # Update omega at the top and bottom boundaries
        omega[ny - 1, 1:-1] = (2 / dx ** 2) * (psi[ny - 1, 1:-1] - psi[ny - 2, 1:-1]) - (2 / dx) * u_w
        omega[0, 1:-1] = (2 / dx ** 2) * (psi[0, 1:-1] - psi[1, 1:-1])

        # Update omega at the left and right boundaries
        omega[1:-1, 0] = (2 / dy ** 2) * (psi[1:-1, 0] - psi[1:-1, 1])
        omega[1:-1, nx - 1] = (2 / dy ** 2) * (psi[1:-1, nx - 1] - psi[1:-1, nx - 2])

        if obstacle:
            # Update omega at the obstacle's boundaries
            omega[obstacle_mask] = 0.0

        # Update Omega (interior points)
        omega[1:-1, 1:-1] = omega[1:-1, 1:-1] + 0.25 * dt / (dx * dy) * (
                    -(psi[2:, 1:-1] - psi[0:-2, 1:-1]) * (omegaprime[1:-1, 2:] - omegaprime[1:-1, 0:-2]) + (
                        psi[1:-1, 2:] - psi[1:-1, 0:-2]) * (omegaprime[2:, 1:-1] - omegaprime[0:-2, 1:-1])) + dt * nu * (
                                      (omegaprime[1:-1, 2:] - 2 * omegaprime[1:-1, 1:-1] + omegaprime[1:-1, 0:-2]) / dx ** 2 + (
                                          omegaprime[2:, 1:-1] - 2 * omegaprime[1:-1, 1:-1] + omegaprime[0:-2, 1:-1]) / dy ** 2)

        psiprime = np.copy(psi)

        # Update Psi (interior points)
        psi[1:-1, 1:-1] = (-omegaprime[1:-1, 1:-1] - ((psiprime[2:, 1:-1] + psiprime[0:-2, 1:-1]) / dy ** 2 + (
                    psiprime[1:-1, 2:] + psiprime[1:-1, 0:-2]) / dx ** 2)) * (-0.5 / ((1 / dx ** 2) + (1 / dy ** 2)))

        if obstacle:
            # Set Psi at the obstacle's boundaries to zero
            psi[obstacle_mask] = 0.0

        U[1:-1, 1:-1] = -(psi[1:-1, 1:-1] - psi[2:, 1:-1]) / dy
        V[1:-1, 1:-1] = -(psi[1:-1, 1:-1] - psi[1:-1, 0:-2]) / dx

    return x, y, U, V, dt * steps

# Streamlit sliders

st.title("2D Lid Driven Cavity Solver with Obstacle")

rey = st.sidebar.slider("Reynolds Number", min_value=1, max_value=1200, value=200)
u_w = st.sidebar.slider("Initial Speed of Cavity", min_value=0.1, max_value=10.0, value=2.0)
steps = st.sidebar.slider("Number of Steps", min_value=100, max_value=50000, value=20000)
obstacle = st.sidebar.checkbox("Include Obstacle", value=True)

if obstacle:
    obs_type = st.sidebar.radio("Obstacle Type", ("rectangular", "circular"))
    if obs_type == "rectangular":
        obs_x0 = st.sidebar.slider("Obstacle X Position", min_value=0.0, max_value=0.8, value=0.4)
        obs_y0 = st.sidebar.slider("Obstacle Y Position", min_value=0.0, max_value=0.8, value=0.4)
        obs_width = st.sidebar.slider("Obstacle Width", min_value=0.1, max_value=0.5, value=0.2)
        obs_height = st.sidebar.slider("Obstacle Height", min_value=0.1, max_value=0.5, value=0.2)
        obs_radius, obs_x_center, obs_y_center = 0, 0, 0
    elif obs_type == "circular":
        obs_radius = st.sidebar.slider("Obstacle Radius", min_value=0.05, max_value=0.5, value=0.1)
        obs_x_center = st.sidebar.slider("Obstacle X Center", min_value=0.0, max_value=1.0, value=0.8)
        obs_y_center = st.sidebar.slider("Obstacle Y Center", min_value=0.0, max_value=1.0, value=0.7)
        obs_x0, obs_y0, obs_width, obs_height = 0, 0, 0, 0
else:
    obs_type, obs_x0, obs_y0, obs_width, obs_height, obs_radius, obs_x_center, obs_y_center = "", 0, 0, 0, 0, 0, 0, 0

if st.sidebar.button("Solve"):
    x, y, U, V, time_passed = solve_navier_stokes_with_obstacle(rey, u_w, steps, obstacle, obs_type, obs_x0, obs_y0, obs_width, obs_height, obs_radius, obs_x_center, obs_y_center)

    # Plotting the results
    fig, ax = plt.subplots()
    ax.streamplot(x, y, U, V, density=2.5)
    
    st.pyplot(fig)

    st.write("The amount of time that has passed =", time_passed)