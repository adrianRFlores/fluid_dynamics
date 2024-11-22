import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Lattice dimensions
nx, ny = 300, 100

# Fluid parameters
viscosity = 0.53
steps = 6000

# Store frames for the simulation
frames = []

# Initialize obstacle map
obstacle_map = np.full((ny, nx), False)

class Obstacle:
    """
    Represents an obstacle in the fluid domain.
    Supports circular and square shapes.
    """
    def __init__(self, x, y, r, objType):
        """
        Initialize an obstacle.
        :param x: x-coordinate of the obstacle center
        :param y: y-coordinate of the obstacle center
        :param r: radius or half-side length of the obstacle
        :param objType: type of obstacle ('circle' or 'cube')
        """
        self.x = x
        self.y = y
        self.r = r
        self.type = objType

    def testPoint(self, x, y):
        """
        Test whether a point (x, y) is inside the obstacle.
        :param x: x-coordinate of the point
        :param y: y-coordinate of the point
        :return: True if the point is inside, False otherwise
        """
        if self.type == 'circle':
            return self.circleTest(x, y)
        elif self.type == 'cube':
            return self.cubeTest(x, y)

    def circleTest(self, x, y):
        """ Check if point (x, y) is within a circular obstacle. """
        return np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2) < self.r

    def cubeTest(self, x, y):
        """ Check if point (x, y) is within a square obstacle. """
        return (self.x - self.r <= x <= self.x + self.r) and (self.y - self.r <= y <= self.y + self.r)

def setObstacle(obstacle):
    """
    Mark the locations of an obstacle on the obstacle map.
    :param obstacle: Obstacle object to add
    """
    for j in range(ny):
        for i in range(nx):
            if obstacle.testPoint(i, j):
                obstacle_map[j, i] = True

# Lattice velocities (9 directions for D2Q9 lattice)
cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])  # x components
cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])  # y components

# Weight factors for the lattice directions
weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

# Initialize particle distribution functions (F) with small random perturbations
F = np.ones((ny, nx, 9)) + 0.01 * np.random.randn(ny, nx, 9)

# Add initial flow in one direction
F[:, :, 3] = 2.3

# Add obstacles
obstacle1 = Obstacle(75, 40, 10, 'cube')
obstacle2 = Obstacle(150, 70, 15, 'circle')
obstacle3 = Obstacle(230, 30, 8, 'circle')
setObstacle(obstacle1)
setObstacle(obstacle2)
setObstacle(obstacle3)

# Main simulation loop
for step in tqdm(range(steps)):
    # Apply periodic boundary conditions
    F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
    F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

    # Stream step: shift distribution functions along their velocity directions
    for i, cx, cy in zip(range(9), cxs, cys):
        F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)  # Shift in x-direction
        F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)  # Shift in y-direction

    # Handle collisions with obstacles
    obstacleF = F[obstacle_map, :]  # Extract values at obstacle locations
    obstacleF = obstacleF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]  # Reverse directions
    F[obstacle_map, :] = obstacleF  # Update obstacle values

    # Compute density and velocity from distribution functions
    density = np.sum(F, axis=2)
    momentum_x = np.sum(F * cxs, axis=2) / density
    momentum_y = np.sum(F * cys, axis=2) / density

    # Zero velocity inside obstacles
    momentum_x[obstacle_map] = 0
    momentum_y[obstacle_map] = 0

    # Compute equilibrium distribution
    F_equilibrium = np.zeros(F.shape)
    for i, cx, cy, w in zip(range(9), cxs, cys, weights):
        eq_term = cx * momentum_x + cy * momentum_y
        F_equilibrium[:, :, i] = density * w * (
            1 + 3 * eq_term + 9 * eq_term ** 2 / 2 - 3 * (momentum_x ** 2 + momentum_y ** 2) / 2
        )

    # Collision step
    F = F - (1 / viscosity) * (F - F_equilibrium)

    # Save frames at regular intervals
    if step % 60 == 0:
        velocity_vector = np.sqrt(momentum_x ** 2 + momentum_y ** 2)
        
        # Plot the velocity field
        fig, ax = plt.subplots()
        im = ax.imshow(velocity_vector, cmap='turbo')
        plt.colorbar(im, ax=ax)
        plt.title(f'Step {step}')
        
        # Save the current frame to an image
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(Image.fromarray(frame))
        plt.close(fig)  # Close the figure to save memory

# Save all frames as a GIF
frames[0].save("simulation_LBM.gif", save_all=True, append_images=frames[1:], duration=10, loop=0)
