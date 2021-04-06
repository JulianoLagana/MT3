import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_generation.mot_data_generation import MotDataGenerator as DataGenerator
from util.load_config_files import dotdict


def generate_traj():
    args = dotdict()
    args.prob_add_obj = 0.2
    args.prob_remove_obj = 0.05
    args.n_average_starting_objects = 4
    args.start_position_mu, args.start_position_std = [0], [[3]]
    args.start_velocity_mu, args.start_velocity_std = [0], [[1]]
    args.delta_t = 0.1
    args.process_noise_intens = 2

    args.prob_measure = 0.9
    args.n_average_false_measurements = 1
    args.measure_noise_intens = 0.1
    args.false_measure_lb = -10
    args.false_measure_ub = 10

    seed = 1


    np.random.seed(seed)
    data_gen = DataGenerator(args)

    for i in range(50):
        data_gen.step()
    data_gen.finish()
    trajectories = data_gen.trajectories

    # Plot trajectories
    for traj in trajectories:
        xs = [float(s[0]) for s in traj]
        ts = [float(s[2]) for s in traj]
        plt.plot(ts, xs, 'o-')

    # Plot measurements taken
    for meas in data_gen.measurements:
        plt.scatter(meas[1], meas[0], color='b', marker='x')

    plt.show()

def generate_traj2d():
    args = dotdict()
    args.prob_add_obj = 0.2
    args.prob_remove_obj = 0.05
    args.n_average_starting_objects = 4
    args.start_position_mu, args.start_position_std = [0, 0], [[3, 0], [0, 3]]
    args.start_velocity_mu, args.start_velocity_std = [0, 0], [[3, 0], [0, 3]]
    args.delta_t = 0.1
    args.process_noise_intens = 1

    args.prob_measure = 0.9
    args.n_average_false_measurements = 0.5
    args.measure_noise_intens = 0.1
    args.false_measure_lb = -10
    args.false_measure_ub = 10

    seed = 0


    np.random.seed(seed)
    data_gen = DataGenerator(args)

    for i in range(50):
        data_gen.step()
    data_gen.finish()

    trajectories = data_gen.trajectories

    # Plot trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for traj in trajectories:
        xs = [float(s[0][0]) for s in traj]
        ys = [float(s[0][1]) for s in traj]
        ts = [float(s[2]) for s in traj]
        ax.plot(ts, xs, ys, 'o-')

    # Plot measurements taken
    for meas in data_gen.measurements:
        ax.scatter(meas[-1], meas[0], meas[1], color='b', marker='x')

    ax.set_xlabel('time')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    plt.show()





if __name__=="__main__":
    generate_traj2d()




