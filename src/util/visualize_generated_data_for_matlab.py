from scipy.io import loadmat
import matplotlib.pyplot as plt


a = loadmat('data_association_task-1000samples-seed0.mat')

measurements = a['measurements'][0]
ground_truths = a['ground_truths'][0]
n_samples = len(measurements)

for i in range(100):
    # Deal with weird way that saving to .mat file works with jagged arrays
    gts = ground_truths[i]

    # Plot the trajectory of all ground-truth objects in the scene
    plt.scatter(gts[:, 0], gts[:, 1], s=100, c='r')

    # Plot all measurements
    m = measurements[i]
    plt.scatter(m[:, 0], m[:, 1], marker='x', c='k', s=30)

    plt.xlim([-11, 11])
    plt.ylim([-11, 11])
    plt.grid()
    plt.show()
