import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

nx, ny = 300, 100
viscosity = 0.53
steps = 6000
frames = []

obstacle_map = np.full((ny, nx), False)

class Obstacle:
    def __init__(self, x, y, r, objType):
        self.x = x
        self.y = y
        self.r = r
        self.type = objType

    def testPoint(self, x, y):
        if self.type == 'circle':
            return self.circleTest(x, y)
        elif self.type == 'cube':
            return self.cubeTest(x, y)

    def circleTest(self, x, y):
        return np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2) < self.r

    def cubeTest(self, x, y):
        return (self.x - self.r <= x <= self.x + self.r) and (self.y - self.r <= y <= self.y + self.r)

def setObstacle(obstacle):
    for j in range(ny):
        for i in range(nx):
            if obstacle.testPoint(i, j):
                obstacle_map[j, i] = True

cxs = np.array(
    [ 0,  0,  1,
      1,  1,  0,
     -1, -1, -1 ]
)
cys = np.array(
    [ 0,  1,  1,
      0, -1, -1,
     -1,  0,  1 ]
)

weights = np.array(
    [ 4/9,  1/9,  1/36,
      1/9,  1/36, 1/9,
      1/36, 1/9,  1/36 ]
)

F = np.ones((ny, nx, 9)) + 0.01 * np.random.randn(ny, nx, 9)

F[:, :, 3] = 2.3

obstacle = Obstacle(75, 40, 10, 'cube')
obstacle2 = Obstacle(150, 70, 15, 'circle')
obstacle3 = Obstacle(230, 30, 8, 'circle')
setObstacle(obstacle)
setObstacle(obstacle2)
setObstacle(obstacle3)

for _ in tqdm(range(steps)):

    F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
    F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

    for i, cx, cy in zip(range(9), cxs, cys):
        F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
        F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

    obstacleF = F[obstacle_map, :]
    obstacleF = obstacleF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

    density = np.sum(F, 2)
    momentum_x = np.sum(F * cxs, 2) / density
    momentum_y = np.sum(F * cys, 2) / density

    F[obstacle_map, :] = obstacleF
    momentum_x[obstacle_map] = 0
    momentum_y[obstacle_map] = 0

    F_equillibrium = np.zeros(F.shape)
    for i, cx, cy, w in zip(range(9), cxs, cys, weights):
        eq_term = cx * momentum_x + cy * momentum_y
        F_equillibrium[:, :, i] = density * w * (
            1 + 3 * eq_term + 9 * eq_term ** 2 / 2 - 3 * (momentum_x ** 2 + momentum_y ** 2) / 2
        )

    F = F - (1 / viscosity) * (F - F_equillibrium)

    if _ % 60 == 0:

        velocity_vector = np.sqrt(momentum_x ** 2 + momentum_y ** 2)
        
        fig, ax = plt.subplots()
        im = ax.imshow(velocity_vector, cmap='turbo')
        plt.colorbar(im, ax=ax)
        plt.title(f'Step {_}')
        #ax.quiver(u, v)
        
        # Save the current frame to a numpy array
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(Image.fromarray(frame))
        plt.close(fig)  # Close the figure to save memory

# Save all frames as a GIF
frames[0].save("simulation_LBM.gif", save_all=True, append_images=frames[1:], duration=10, loop=0)