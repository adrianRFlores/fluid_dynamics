import numpy as np
import math
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Grid size and parameters
nx, ny = 50, 50  # Number of grid points in the x and y directions
dx, dy = 0.001, 0.001  # Grid spacing in the x and y directions
dt = 1/120  # Time step
iterations = 100  # Number of iterations for Gauss-Seidel solver
overrelaxation = 1.7  # Overrelaxation parameter for Gauss-Seidel
g = 0  # Gravitational acceleration
density = 1  # Density of the fluid (assumed constant)
in_velocity = 0.5  # Initial velocity

# Initialize velocity, pressure
u = np.zeros((nx, ny))  # Horizontal velocity field
v = np.zeros((nx, ny))  # Vertical velocity field
s = np.zeros((nx, ny))  # Obstacle field (0 means no obstacle, 1 means obstacle)
p = np.zeros((nx, ny))  # Pressure field

# Function to update the velocity based on the source term
def update_velocity():
    # Loop over the grid excluding boundaries
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Only update if there is a source term at this grid point
            if s[i, j] != 0 and s[i, j - 1] != 0:
                u[i, j] += g * dt  # Update horizontal velocity based on gravity

# Gauss-Seidel solver to update the velocities
def gauss_seidel():
    cp = density * dx * dy / dt  # Constant related to the grid spacing and time step

    for _ in range(iterations):
        # Loop over all internal grid points (excluding boundaries)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # Skip cells without source terms
                if s[i, j] == 0:
                    continue
                
                # Neighbors of the current cell
                sx0 = s[i - 1, j]
                sx1 = s[i + 1, j]
                sy0 = s[i, j - 1]
                sy1 = s[i, j + 1]

                # Sum of neighboring source terms
                neighbor_sum = sx0 + sx1 + sy0 + sy1

                if neighbor_sum == 0:
                    continue

                # Calculate the divergence term (d) for velocity correction
                d = overrelaxation * (-(u[i+1, j] - u[i, j] + v[i, j+1] - v[i, j])) / neighbor_sum
                
                # Apply the correction to the velocities
                u[i, j] -= sx0 * d
                u[i + 1, j] += sx1 * d
                v[i, j] -= sy0 * d
                v[i, j + 1] += sy1 * d

                # Update the pressure field
                p[i, j] += d * cp

# Function to define obstacles in the grid
def obstacle(x, y, r):
    for i in range(nx - 2):
        for j in range(ny - 2):
            s[i, j] = 1.0  # Initially, all cells are set to source (no obstacle)
            # Check if the cell is within the circular obstacle
            delta_x = (i + 0.5) * dx - x
            delta_y = (j + 0.5) * dx - y
            if delta_x ** 2 + delta_y ** 2 < r ** 2:  # If inside obstacle radius
                s[i, j] = 0  # Set to 0 (obstacle)

# Function to compute the divergence of the velocity field
def check_divergence():
    divergence = np.zeros((nx, ny))  # Initialize a divergence field
    # Loop over all grid points
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # Compute divergence using finite difference
            divergence[i, j] = (u[i + 1, j] - u[i, j]) / dx + (v[i, j + 1] - v[i, j]) / dy
    return divergence

# Function to sample the velocity field at a given (x, y) position
def getSample(x, y, field_type):
    n = ny  # Number of grid points in the y direction
    h = dx  # Grid spacing
    h1 = 1 / h  # Inverse of the grid spacing
    h2 = 0.5 * h  # Half of the grid spacing

    # Bound the input (x, y) coordinates within the grid limits
    x_sample = max(min(x, nx * h), h)
    y_sample = max(min(y, ny * h), h)

    sample_dx = 0
    sample_dy = 0

    field = None

    # Set the field to sample from based on the field type ('u' for u-velocity, 'v' for v-velocity)
    if field_type == 'u':
        field = u
        sample_dy = h2  # Adjust for vertical offset when sampling u-velocity
    elif field_type == 'v':
        field = v
        sample_dx = h2  # Adjust for horizontal offset when sampling v-velocity

    # Convert (x_sample, y_sample) into grid indices (x0, y0)
    x0 = min(math.floor((x_sample - sample_dx) * h1), nx - 1)
    tx = ((x_sample - sample_dx) - x0 * h) * h1
    x1 = min(x0 + 1, nx - 1)

    y0 = min(math.floor((y_sample - sample_dy) * h1), ny - 1)
    ty = ((y_sample - sample_dy) * h) * h1
    y1 = min(y0 + 1, ny - 1)

    # Bilinear interpolation to get the velocity value at the given sample point
    sx = 1 - tx
    sy = 1 - ty

    return sx * sy * field[x0, y0] + tx * sy * field[x1, y0] + tx * ty * field[x1, y1] + sx * ty * field[x0, y1]

# Function to advect (transport) the velocity field
def advect():
    global u, v
    tempU = u.copy()  # Create temporary arrays to store the updated velocity fields
    tempV = v.copy()

    n = ny
    h = dx
    h2 = 0.5 * h

    # Loop over all grid points (excluding boundaries)
    for i in range(1, nx):
        for j in range(1, ny):

            # Only advect if the cell is not an obstacle
            if s[i, j] != 0 and s[i - 1, j] != 0 and j < ny - 1:
                x = i * h
                y = j * h + h2 
                u_sample = u[i, j]  # Sample u-velocity at the current position
                v_sample = (v[i, j] + v[i - 1, j] + v[i - 1, j + 1] + v[i, j + 1]) * 0.25  # Average v-velocity from neighbors
                x -= dt * u_sample  # Move x position backward by the velocity * time step
                y -= dt * v_sample  # Move y position backward by the velocity * time step

                # Interpolate the velocity at the new position and store it
                tempU[i, j] = getSample(x, y, 'u')

            # Repeat for vertical velocity
            if s[i, j] != 0 and s[i, j - 1] != 0 and i < nx - 1:
                x = i * h + h2
                y = j * h
                u_sample = (u[i, j] + u[i, j - 1] + u[i + 1, j - 1] + u[i + 1, j]) * 0.25  # Average u-velocity from neighbors
                v_sample = v[i, j]  # Sample v-velocity at the current position

                x -= dt * u_sample  # Move x position backward
                y -= dt * v_sample  # Move y position backward

                # Interpolate the velocity at the new position and store it
                tempV[i, j] = getSample(x, y, 'v')

    # Update the velocity fields with the advected values
    u = tempU
    v = tempV

def extrapolate():
    for i in range(nx):
        u[i, 0] = u[i, 1]
        u[i, ny - 1] = u[i, ny - 2]
    for j in range(ny):
        v[0, j] = v[1, j]
        v[nx - 1, j] = v[nx - 2, j]

# Simulation Setup
for i in range(nx):
    for j in range(ny):
        state = 1 # Fluid
        if i == 0 or j == 0 or j == ny - 1: # Border
            state = 0 # Solid
        s[i, j] = state

        if i in range(20, 30) and j == 1:
            v[i, j] = in_velocity

def step():

    # Apply the Gauss-Seidel method to enforce incompressibility
    update_velocity()

    global p
    p = np.zeros((nx, ny))

    gauss_seidel()
    extrapolate()
    advect()

obstacle(25, 25, 5)

# Set up an empty list to store frames
frames = []

# Simulate and save each frame
for _ in tqdm(range(50)):
    step()  # Your simulation step function
    div = check_divergence()
    non_div_count = 0
    for i in range(nx):
        for j in range(ny):
            if div[i, j] > 0.01 or div[i, j] < -0.01:
                non_div_count += 1
    fig, ax = plt.subplots()
    im = ax.imshow(p, cmap='turbo')
    plt.colorbar(im, ax=ax)
    plt.title(f'Step {_}')
    
    # Save the current frame to a numpy array
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(Image.fromarray(frame))
    plt.close(fig)  # Close the figure to save memory

# Save all frames as a GIF
frames[0].save("simulation.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)