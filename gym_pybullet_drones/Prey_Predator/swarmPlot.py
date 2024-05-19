
import matplotlib.pyplot as plt
import numpy as np

class SwarmPlotter:
    def __init__(self, n_agents, n_preys, x_lim, y_lim, z_lim, drones_no_sensors, no_sensor_percentage, boundless=False):
        title = f"Swarm of {n_agents} agents and {n_preys} preys with {no_sensor_percentage * 100}% no distance sensors predators"
        self.n_agents = n_agents
        self.n_preys = n_preys
        self.boundless = boundless
        self.drones_no_sensors = np.array(drones_no_sensors)
        print(self.drones_no_sensors)   
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.trajectory = []
        self.trajectory_preys = []
        self.scat = self.ax.scatter([], [], [], color="red", label="Predator")
        self.scat2 = self.ax.scatter([], [], [], color="blue", label="Prey")
        self.scat3 = self.ax.scatter([], [], [], color="green", label="No sensors")
        self.quiv = self.ax.quiver([], [], [], [], [], [], color="red")
        self.quiv2 = self.ax.quiver([], [], [], [], [], [], color="blue")
        self.quiv3 = self.ax.quiver([], [], [], [], [], [], color="green")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        if not self.boundless:
            self.ax.set_xlim(0, x_lim)
            self.ax.set_ylim(0, y_lim)
            self.ax.set_zlim(0, z_lim)
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

        if self.boundless:
            padding = 5
            all_xs = np.concatenate([pos_xs, pos_preys_xs])
            all_ys = np.concatenate([pos_ys, pos_preys_ys])
            all_zs = np.concatenate([pos_zs, pos_preys_zs])
            self.ax.set_xlim(all_xs.min() - padding, all_xs.max() + padding)
            self.ax.set_ylim(all_ys.min() - padding, all_ys.max() + padding)
            self.ax.set_zlim(all_zs.min() - padding, all_zs.max() + padding)

        boolean_mask = np.zeros(len(pos_xs), dtype=bool)
        boolean_mask[self.drones_no_sensors] = True

        pos_xs_no_sensors = pos_xs[boolean_mask]
        pos_ys_no_sensors = pos_ys[boolean_mask]
        pos_zs_no_sensors = pos_zs[boolean_mask]
        pos_hxs_no_sensors = pos_hxs[boolean_mask]
        pos_hys_no_sensors = pos_hys[boolean_mask]
        pos_hzs_no_sensors = pos_hzs[boolean_mask]

        pos_xs = pos_xs[~boolean_mask]
        pos_ys = pos_ys[~boolean_mask]
        pos_zs = pos_zs[~boolean_mask]
        pos_hxs = pos_hxs[~boolean_mask]
        pos_hys = pos_hys[~boolean_mask]
        pos_hzs = pos_hzs[~boolean_mask]

        self.scat3._offsets3d = (pos_xs_no_sensors, pos_ys_no_sensors, pos_zs_no_sensors)
        self.quiv3.remove()

        self.scat._offsets3d = (pos_xs, pos_ys, pos_zs)
        self.scat2._offsets3d = (pos_preys_xs, pos_preys_ys, pos_preys_zs)

        self.quiv.remove()
        self.quiv2.remove()

        self.quiv = self.ax.quiver(pos_xs, pos_ys, pos_zs, np.cos(pos_hxs), np.cos(pos_hys), np.cos(pos_hzs), color="red", length=0.8)
        self.quiv2 = self.ax.quiver(pos_preys_xs, pos_preys_ys, pos_preys_zs, np.cos(pos_preys_hxs), np.cos(pos_preys_hys), np.cos(pos_preys_hzs), color="blue", length=0.8)
        self.quiv3 = self.ax.quiver(pos_xs_no_sensors, pos_ys_no_sensors, pos_zs_no_sensors, np.cos(pos_hxs_no_sensors), np.cos(pos_hys_no_sensors), np.cos(pos_hzs_no_sensors), color="green", length=0.8)




        plt.pause(0.0001)  # Adjust this value as needed for your visualization needs

