import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

nx, ny = 300, 100
viscosity = 0.53
steps = 6000

obstacle_map = np.full((ny, nx), False)

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

for _ in tqdm(range(steps)):

    F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
    F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

    for i, cx, cy in zip(range(9), cxs, cys):
        F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
        F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

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