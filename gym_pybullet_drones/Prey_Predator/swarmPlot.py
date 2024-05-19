import matplotlib.pyplot as plt
import numpy as np


class SwarmPlotter:
    def __init__(self, n_agents, n_preys, x_lim, y_lim, z_lim, drones_no_sensors, no_sensor_percentage,boundless=False):

        title = f"Swarm of {n_agents} agents and {n_preys} preys with {no_sensor_percentage*100}% no distance sensors predators"
        self.n_agents = n_agents
        self.n_preys = n_preys
        self.boundless = boundless

        self.drones_no_sensors = drones_no_sensors
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        # self.trajectories = [[] for _ in range(n_agents)]
        # self.trajectories_preys = [[] for _ in range(n_preys)]
        self.trajectory = []
        self.trajectory_preys = []
        # Initialize scatter plot
        self.scat = self.ax.scatter([], [], [], color="red", label="Predator")
        self.scat2 = self.ax.scatter([], [], [], color="blue", label="Prey")
        self.scat3 = self.ax.scatter([], [], [], color="green", label="No sensors")
        # Initialize quiver plot
        self.quiv  = self.ax.quiver([], [], [], [], [], [], color="red")
        self.quiv2 = self.ax.quiver([], [], [], [], [], [], color="blue")
        self.quiv3 = self.ax.quiver([], [], [], [], [], [], color="green")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        if not self.boundless:
            self.ax.set_xlim(0,x_lim)
            self.ax.set_ylim(0,y_lim)
            self.ax.set_zlim(0,z_lim)
        plt.title(title)
        plt.legend()
        self.sphere_plots = []

    def update_plot(self, pos_xs, pos_ys, pos_zs, pos_hxs, pos_hys, pos_hzs, pos_preys_xs, pos_preys_ys, pos_preys_zs, pos_preys_hxs, pos_preys_hys, pos_preys_hzs):

        mean_pos = (np.mean(pos_xs), np.mean(pos_ys), np.mean(pos_zs))
        self.trajectory.append(mean_pos)
        self.ax.plot(*zip(*self.trajectory), color='purple', label="Trajectory Predators")

        mean_pos_preys = (np.mean(pos_preys_xs), np.mean(pos_preys_ys), np.mean(pos_preys_zs))
        self.trajectory_preys.append(mean_pos_preys)
        self.ax.plot(*zip(*self.trajectory_preys), color='cyan', label="Trajectory Preys")


        # Update the plot limits to move with the positions
        if self.boundless:
            padding = 10  # Adjust as needed
            all_xs = np.concatenate([pos_xs, pos_preys_xs])
            all_ys = np.concatenate([pos_ys, pos_preys_ys])
            all_zs = np.concatenate([pos_zs, pos_preys_zs])
            self.ax.set_xlim(all_xs.min() - padding, all_xs.max() + padding)
            self.ax.set_ylim(all_ys.min() - padding, all_ys.max() + padding)
            self.ax.set_zlim(all_zs.min() - padding, all_zs.max() + padding)

        if len(self.drones_no_sensors) != 0:
            self.scat._offsets3d = (pos_xs[~self.drones_no_sensors], pos_ys[~self.drones_no_sensors], pos_zs[~self.drones_no_sensors])
            self.scat2._offsets3d = (pos_preys_xs, pos_preys_ys, pos_preys_zs)
            self.scat3._offsets3d = (pos_xs[self.drones_no_sensors], pos_ys[self.drones_no_sensors], pos_zs[self.drones_no_sensors])

            # Update quiver plot data
            self.quiv.remove()  # Currently necessary due to Matplotlib's limitations with 3D quiver updates
            self.quiv2.remove()
            self.quiv3.remove()
            self.quiv = self.ax.quiver(pos_xs[~self.drones_no_sensors], pos_ys[~self.drones_no_sensors], pos_zs[~self.drones_no_sensors], np.cos(pos_hxs[~self.drones_no_sensors]), np.cos(pos_hys[~self.drones_no_sensors]), np.cos(pos_hzs[~self.drones_no_sensors]), color="red", length=0.8)
            self.quiv2 = self.ax.quiver(pos_preys_xs, pos_preys_ys, pos_preys_zs, np.cos(pos_preys_hxs), np.cos(pos_preys_hys), np.cos(pos_preys_hzs), color="blue", length=0.8)
            self.quiv3 = self.ax.quiver(pos_xs[self.drones_no_sensors], pos_ys[self.drones_no_sensors], pos_zs[self.drones_no_sensors], np.cos(pos_hxs[self.drones_no_sensors]), np.cos(pos_hys[self.drones_no_sensors]), np.cos(pos_hzs[self.drones_no_sensors]), color="green", length=0.8)
        else:
            self.scat._offsets3d = (pos_xs, pos_ys, pos_zs)
            self.scat2._offsets3d = (pos_preys_xs, pos_preys_ys, pos_preys_zs)
            # Update quiver plot data
            self.quiv.remove()
            self.quiv2.remove()
            self.quiv = self.ax.quiver(pos_xs, pos_ys, pos_zs, np.cos(pos_hxs), np.cos(pos_hys), np.cos(pos_hzs), color="red", length=0.8)
            self.quiv2 = self.ax.quiver(pos_preys_xs, pos_preys_ys, pos_preys_zs, np.cos(pos_preys_hxs), np.cos(pos_preys_hys), np.cos(pos_preys_hzs), color="blue", length=0.8)
       
       
        plt.pause(0.0000001)  # Adjust this value as needed for your visualization needs