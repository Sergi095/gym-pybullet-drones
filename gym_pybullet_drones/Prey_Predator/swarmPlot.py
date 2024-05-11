import matplotlib.pyplot as plt
import numpy as np


class SwarmPlotter:
    def __init__(self, n_agents, n_preys, x_lim, y_lim, z_lim):
        self.n_agents = n_agents
        self.n_preys = n_preys
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        # self.trajectories = [[] for _ in range(n_agents)]
        # self.trajectories_preys = [[] for _ in range(n_preys)]
        self.trajectory = []
        self.trajectory_preys = []
        # Initialize scatter plot
        self.scat = self.ax.scatter([], [], [], color="red", label="Predator")
        self.scat2 = self.ax.scatter([], [], [], color="blue", label="Prey")
        # Initialize quiver plot
        self.quiv  = self.ax.quiver([], [], [], [], [], [], color="red")
        self.quiv2 = self.ax.quiver([], [], [], [], [], [], color="blue")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.legend()

    def update_plot(self, pos_xs, pos_ys, pos_zs, pos_hxs, pos_hys, pos_hzs, pos_preys_xs, pos_preys_ys, pos_preys_zs, pos_preys_hxs, pos_preys_hys, pos_preys_hzs):
        # Update scatter plot data
        # for i in range(self.n_agents):
        #     self.trajectories[i].append((pos_xs[i], pos_ys[i], pos_zs[i]))
        #     self.ax.plot(*zip(*self.trajectories[i]), color='red')

        # for i in range(self.n_preys):
        #     self.trajectories_preys[i].append((pos_preys_xs[i], pos_preys_ys[i], pos_preys_zs[i]))
        #     self.ax.plot(*zip(*self.trajectories_preys[i]), color='blue')
        mean_pos = (np.mean(pos_xs), np.mean(pos_ys), np.mean(pos_zs))
        self.trajectory.append(mean_pos)
        self.ax.plot(*zip(*self.trajectory), color='red')

        mean_pos_preys = (np.mean(pos_preys_xs), np.mean(pos_preys_ys), np.mean(pos_preys_zs))
        self.trajectory_preys.append(mean_pos_preys)
        self.ax.plot(*zip(*self.trajectory_preys), color='blue')


        # Update the plot limits to move with the positions
        padding = 10  # Adjust as needed
        all_xs = np.concatenate([pos_xs, pos_preys_xs])
        all_ys = np.concatenate([pos_ys, pos_preys_ys])
        all_zs = np.concatenate([pos_zs, pos_preys_zs])
        self.ax.set_xlim(all_xs.min() - padding, all_xs.max() + padding)
        self.ax.set_ylim(all_ys.min() - padding, all_ys.max() + padding)
        self.ax.set_zlim(all_zs.min() - padding, all_zs.max() + padding)

        self.scat._offsets3d = (pos_xs, pos_ys, pos_zs)
        self.scat2._offsets3d = (pos_preys_xs, pos_preys_ys, pos_preys_zs)
        # Update quiver plot data
        self.quiv.remove()  # Currently necessary due to Matplotlib's limitations with 3D quiver updates
        self.quiv2.remove()
        self.quiv = self.ax.quiver(pos_xs, pos_ys, pos_zs, np.cos(pos_hxs), np.cos(pos_hys), np.cos(pos_hzs), color="red", length=0.08)
        self.quiv2 = self.ax.quiver(pos_preys_xs, pos_preys_ys, pos_preys_zs, np.cos(pos_preys_hxs), np.cos(pos_preys_hys), np.cos(pos_preys_hzs), color="blue", length=0.1)
        plt.pause(0.0000001)  # Adjust this value as needed for your visualization needs