import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 2.0, 1.5, 1.0])
y = np.array([1.0, 1.0, 1 + np.sqrt(3)/2, 1.0])

A = np.array(x[0],y[0])
B = np.array((x[1],y[1]))
C = np.array(x[2], y[2])

if np.linalg.norm(A-B) and np.linalg.norm(B-C) and np.linalg.norm(A-C):
    print(str(np.linalg.norm(A-B)))
else:
    print('nisu')

plt.figure()

plt.plot(x, y, 'b', linewidth=1.5, marker='.', markersize=10)

plt.axis([0, 4, 0, 4])
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer')

plt.show()

