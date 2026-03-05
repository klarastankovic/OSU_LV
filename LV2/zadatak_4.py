import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50, 50), dtype=np.uint8)
white = np.ones((50, 50), dtype=np.uint8) * 255

first_row = np.hstack((black, white, black, white))
second_row = np.hstack((white, black, white, black))
third_row = np.hstack((black, white, black, white))
forth_row = np.hstack((white, black, white, black))

image = np.vstack((first_row, second_row, third_row, forth_row))

plt.figure()
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.show()