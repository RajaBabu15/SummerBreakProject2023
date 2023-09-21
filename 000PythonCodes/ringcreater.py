import numpy as np

# Create an empty 10x10 array
t = np.zeros((10, 10))

# Define the center of the array
center = [len(t) // 2, len(t[0]) // 2]

# Iterate over the array and assign values based on the distance from the center
for i in range(len(t)):
    for j in range(len(t[0])):
        dist = max(abs(i - center[0]), abs(j - center[1]))
        if dist < 1:
            t[i][j] = 5
        elif dist < 2:
            t[i][j] = 4
        elif dist < 3:
            t[i][j] = 3
        elif dist < 4:
            t[i][j] = 2
        else:
            t[i][j] = 1

print(t)
