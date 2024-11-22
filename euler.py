import numpy as np
import math
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Grid size and parameters
nx, ny = 50, 50
dx, dy = 1, 1
dt = 1/120
nu = 0.1  # Viscosity
iterations = 100
overrelaxation = 1.9
g = 0
density = 1000
in_velocity = 0.5

# Initialize velocity and pressure
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
s = np.zeros((nx, ny))
p = np.zeros((nx, ny))

def obstacle(x, y, r):
    for i in range(nx - 2):
        for j in range(ny - 2):
            s[i, j] = 1.0
            delta_x = (i + 0.5) * dx - x
            delta_y = (j + 0.5) * dx - y
            if delta_x ** 2 + delta_y ** 2 < r ** 2:
                s[i, j] = 0

def update_velocity():
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            if s[i, j] != 0 and s[i, j - 1] != 0:
                u[i, j] += g * dt

def gauss_seidel():

    cp = density * dx * dy / dt

    for _ in range(iterations):
        for i in range(1, nx-1):
            for j in range(1, ny-1):

                if s[i, j] == 0:
                    continue
                
                sx0 = s[i - 1, j]
                sx1 = s[i + 1, j]
                sy0 = s[i, j - 1]
                sy1 = s[i, j + 1]

                # Calculate the summation of neighboring s values
                neighbor_sum  = sx0 + sx1 + sy0 + sy1

                if neighbor_sum == 0:
                    continue

                # Calculate the divergence term d
                d = overrelaxation * (-(u[i+1, j] - u[i, j] + v[i, j+1] - v[i, j])) / neighbor_sum
                
                # Update u and v based on Gauss-Seidel correction
                u[i, j] -= sx0 * d
                u[i + 1, j] += sx1 * d
                v[i, j] -= sy0 * d
                v[i, j + 1] += sy1 * d

                p[i, j] += d * cp

def getSample(x, y, field_type):
    n = ny
    h = dx
    h1 = 1 / h
    h2 = 0.5 * h

    x_sample = max(min(x, nx * h), h)
    y_sample = max(min(y, ny * h), h)

    sample_dx = 0
    sample_dy = 0

    field = None

    if field_type == 'u':
        field = u
        sample_dy = h2
    elif field_type == 'v':
        field = v
        sample_dx = h2

    x0 = min(math.floor((x_sample - sample_dx) * h1), nx - 1)
    tx = ((x_sample - sample_dx) - x0 * h) * h1
    x1 = min(x0 + 1, nx - 1)

    y0 = min(math.floor((y_sample - sample_dy) * h1), ny - 1)
    ty = ((y_sample - sample_dy) * h) * h1
    y1 = min(y0 + 1, ny - 1)

    sx = 1 - tx
    sy = 1 - ty

    return sx * sy * field[x0, y0] + tx * sy * field[x1, y0] + tx * ty * field[x1, y1] + sx * ty * field[x0, y1]

def advect():
    global u, v
    tempU = u.copy()
    tempV = v.copy()

    n = ny
    h = dx
    h2 = 0.5 * h

    for i in range(1, nx):
        for j in range(1, ny):

            if s[i, j] != 0 and s[i - 1, j] != 0 and j < ny - 1:
                x = i * h
                y = j * h + h2 
                u_sample = u[i, j]
                v_sample = (v[i, j] + v[i - 1, j] + v[i - 1, j + 1] + v[i, j + 1]) * 0.25
                x -= dt * u_sample
                y -= dt * v_sample

                tempU[i, j] = getSample(x, y, 'u')

            if s[i, j] != 0 and s[i, j - 1] != 0 and i < nx - 1:
                x = i * h + h2
                y = j * h
                u_sample = (u[i, j] + u[i, j - 1] + u[i + 1, j - 1] + u[i + 1, j]) * 0.25
                v_sample = v[i, j]

                x -= dt * u_sample
                y -= dt * v_sample

                tempV[i, j] = getSample(x, y, 'v')

    u = tempU
    v = tempV

for i in range(nx):
    for j in range(ny):
        state = 1 # Fluid
        if i == 0 or j == 0 or j == ny - 1: # Border
            state = 0 # Solid
        s[i, j] = state

def extrapolate():
    for i in range(nx):
        u[i, 0] = u[i, 1]
        u[i, ny - 1] = u[i, ny - 2]
    for j in range(ny):
        v[0, j] = v[1, j]
        v[nx - 1, j] = v[nx - 2, j]

obstacle(25, 25, 5)

def step():

    #for j in range(int(nx / 4), int(nx - nx / 4)):
        #u[1, j] = in_velocity

    # Apply the Gauss-Seidel method to enforce incompressibility
    update_velocity()

    global p
    p = np.zeros((nx, ny))

    gauss_seidel()
    extrapolate()
    advect()

    pass

# Set up an empty list to store frames
frames = []

# Simulate and save each frame
for _ in tqdm(range(50)):
    step()  # Your simulation step function
    fig, ax = plt.subplots()
    im = ax.imshow(p, cmap='turbo')
    plt.colorbar(im, ax=ax)
    #ax.quiver(u, v)
    
    # Save the current frame to a numpy array
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(Image.fromarray(frame))
    plt.close(fig)  # Close the figure to save memory

# Save all frames as a GIF
frames[0].save("simulation.gif", save_all=True, append_images=frames[1:], duration=300, loop=0)

