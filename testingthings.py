
import numpy as np
import matplotlib.pyplot as plt

# plotting buffer sample

numbers = np.arange(10)

print(numbers)

for number in numbers:

    data = np.load("experiments/15_DASDL/buffer/denoised_buffer" + str(number) + ".npy")

    plt.imshow(data, extent=(0, 5, 0, 5))
    plt.show()


