import numpy as np
x = np.load('ai8x-synthesis/tests/sample_indoorenvironment.npy')
print("Original sample shape:", x.shape)

# Convert to CHW (1, 101, 2)
if x.shape == (101, 2):
    # Add channel dimension first
    x = x[np.newaxis, :, :]
    print("Added channel dimension:", x.shape)
elif x.shape == (101, 2, 1):
    # Transpose from HWC to CHW
    x = np.transpose(x, (2, 0, 1))
    print("Transposed from HWC to CHW:", x.shape)
elif x.shape == (1, 101, 2):
    print("Already in CHW format:", x.shape)
else:
    print("Unexpected shape, saving as is:", x.shape)

np.save('ai8x-synthesis/tests/sample_indoorenvironment.npy', x)
print("Final sample shape:", x.shape)
