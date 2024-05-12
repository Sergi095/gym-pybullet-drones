import numpy as np
from swarmPlot import SwarmPlotter

class FlockingUtils:
    def __init__(self, n_predators, preys, center_x, center_y, center_z, spacing, center_x_preys, center_y_preys, center_z_preys, drones_ids, perc_no_sensor, boundless):
        self.n_agents = n_predators
        self.n_preys = preys
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z

        self.center_x_preys = center_x_preys
        self.center_y_preys = center_y_preys
        self.center_z_preys = center_z_preys
        self.spacing = spacing

        self.drones_ids = drones_ids
        self.no_sensor_predators = self.no_sensor_predators(perc_no_sensor, self.drones_ids)

        self.boun_thresh = 0.5
        self.boun_x = 20
        self.boun_y = 20 
        self.boun_z = 20

        self.Dp = 3.0
        self.Dp_preys = 3.0

        self.sensing_range = 2.0 # sensing range for preys
        self.sensor_range  = 5.0 # sensing range for predators


        self.sigma = 1.4
        self.sigma_no_sensor = 1.7
        self.sigma_prey = 0.7
        # self.sigma = 0.3
        self.sigmas = self.sigma * np.ones(self.n_agents)
        self.sigmas_b = 0.05 * np.ones(self.n_agents) # for boundary repulsion (not planning on using this)

        self.sigmas_preys = self.sigma_prey * np.ones(self.n_preys)
        self.sigmas_b_preys = 0.05 * np.ones(self.n_preys) # for boundary repulsion (not planning on using this)

        self.epsilon = 12.0
        self.alpha = 2.0
        self.alpha_preys = 1.0# adding alpha for preys
        self.beta = 2.0
        self.kappa = 2.0
        self.k1 = 0.5
        # self.k1 = 0.3
        self.k2 = 0.1
        self.umax_const = 0.15
        # self.umax_const = 0.1
        self.wmax = np.pi/3.0
        self.h_alignment = False
        # self.dt = 0.042
        self.dt = 0.05
        self.noise_pos = 0.05
        self.noise_h = np.pi / 72
        self.rng = np.random.default_rng(1234)
        self.mean_noise = 0.5




        self.d_ij = np.zeros([self.n_agents, self.n_agents])
        self.pos_h_xc = np.zeros(self.n_agents)
        self.pos_h_yc = np.zeros(self.n_agents)
        self.pos_h_zc = np.zeros(self.n_agents)
        self.ij_ang_x = np.zeros([self.n_agents, self.n_agents])
        self.ij_ang_y = np.zeros([self.n_agents, self.n_agents])
        self.ij_ang_z = np.zeros([self.n_agents, self.n_agents])
        self.f_x = np.zeros(self.n_agents)
        self.f_y = np.zeros(self.n_agents)
        self.f_z = np.zeros(self.n_agents)
        self.fa_x = np.zeros(self.n_agents)
        self.fa_y = np.zeros(self.n_agents)
        self.fa_z = np.zeros(self.n_agents)
        self.u = np.zeros(self.n_agents)
        self.w = np.zeros(self.n_agents)

        self.d_ij_from_preys = np.zeros([self.n_agents, self.n_preys])
        
        self.d_ij_from_predators = np.zeros([self.n_preys, self.n_agents])

        self.d_ij_preys = np.zeros([self.n_preys, self.n_preys])
        self.pos_h_xc_preys = np.zeros(self.n_preys)
        self.pos_h_yc_preys = np.zeros(self.n_preys)
        self.pos_h_zc_preys = np.zeros(self.n_preys)
        self.ij_ang_x_preys = np.zeros([self.n_preys, self.n_preys])
        self.ij_ang_y_preys = np.zeros([self.n_preys, self.n_preys])
        self.ij_ang_z_preys = np.zeros([self.n_preys, self.n_preys])
        self.f_x_preys = np.zeros(self.n_preys)
        self.f_y_preys = np.zeros(self.n_preys)
        self.f_z_preys = np.zeros(self.n_preys)
        self.fa_x_preys = np.zeros(self.n_preys)
        self.fa_y_preys = np.zeros(self.n_preys)
        self.fa_z_preys = np.zeros(self.n_preys)
        self.u_preys = np.zeros(self.n_preys)
        self.w_preys = np.zeros(self.n_preys)

        map_x, map_y, map_z = [np.linspace(-1, 1, 150) for _ in range(3)]
        X, Y, Z = np.meshgrid(map_x, map_y, map_z)
        self.map_3d = 255 * np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * self.sigma ** 2))
        self.grad_const_x, self.grad_const_y, self.grad_const_z = [150 / boun for boun in (self.boun_x, self.boun_y, self.boun_z)]

        self.plotter = SwarmPlotter(self.n_agents, self.n_preys, self.boun_x, self.boun_y, self.boun_z, self.no_sensor_predators, no_sensor_percentage=perc_no_sensor, boundless=boundless)


    def initialize_positions(self, preds: bool = True):
        """
        Place agents in a 3D grid around an initialization point with specified spacing and noise.

        Parameters:
            num_agents (int): Number of agents to place.
            init_pos (tuple): Initialization point (x, y, z).
            spacing (float): Spacing between agents.
            mean_noise (float): Mean value of noise to apply to positions.

        Returns:
            np.array: 3D positions of agents.
        """


        if preds:
            # Approximate cube root to start searching for dimensions
            num_agents = self.n_agents
            spacing = self.spacing
            init_pos = (self.center_x, self.center_y, self.center_z)
            mean_noise = self.mean_noise

            cube_root = round(num_agents ** (1 / 3))

            # Find dimensions that fill a space as equally as possible, even if some agents are left out
            best_diff = float('inf')
            for x in range(cube_root, 0, -1):
                for y in range(x, 0, -1):
                    z = int(np.ceil(num_agents / (x * y)))
                    total_agents = x * y * z
                    diff = max(abs(x - y), abs(y - z), abs(x - z))
                    if diff < best_diff and total_agents >= num_agents:
                        best_diff = diff
                        dimensions = (x, y, z)

            # Generate grid positions
            grid_positions = np.mgrid[0:dimensions[0], 0:dimensions[1], 0:dimensions[2]].reshape(3, -1).T
            grid_positions = grid_positions * spacing

            # Center the grid around the init_pos
            offset = np.array(init_pos) - (np.array(dimensions) * spacing / 2)
            grid_positions += offset

            # Apply noise
            noise = np.random.normal(loc=mean_noise, scale=mean_noise / 3, size=grid_positions.shape)
            grid_positions += noise

            theta = np.random.uniform(0, 2 * np.pi, self.n_agents)
            phi = np.random.uniform(0, np.pi, self.n_agents)

            self.pos_h_xc = np.sin(phi) * np.cos(theta)
            self.pos_h_yc = np.sin(phi) * np.sin(theta)
            self.pos_h_zc = np.cos(phi)

            return (grid_positions[:num_agents, 0], grid_positions[:num_agents, 1], grid_positions[:num_agents, 2],
                    self.pos_h_xc, self.pos_h_yc, self.pos_h_zc)
        else: # preys
            num_preys = self.n_preys
            spacing = self.spacing
            init_pos = (self.center_x_preys, self.center_y_preys, self.center_z_preys)
            mean_noise = self.mean_noise

            cube_root = round(num_preys ** (1 / 3))

            # Find dimensions that fill a space as equally as possible, even if some agents are left out
            best_diff = float('inf')
            for x in range(cube_root, 0, -1):
                for y in range(x, 0, -1):
                    z = int(np.ceil(num_preys / (x * y)))
                    total_agents = x * y * z
                    diff = max(abs(x - y), abs(y - z), abs(x - z))
                    if diff < best_diff and total_agents >= num_preys:
                        best_diff = diff
                        dimensions = (x, y, z)

            # Generate grid positions
            grid_positions = np.mgrid[0:dimensions[0], 0:dimensions[1], 0:dimensions[2]].reshape(3, -1).T
            grid_positions = grid_positions * spacing

            # Center the grid around the init_pos
            offset = np.array(init_pos) - (np.array(dimensions) * spacing / 2)
            grid_positions += offset

            # Apply noise
            noise = np.random.normal(loc=mean_noise, scale=mean_noise / 3, size=grid_positions.shape)
            grid_positions += noise

            theta = np.random.uniform(0, 2 * np.pi, self.n_preys)
            phi = np.random.uniform(0, np.pi, self.n_preys)

            self.pos_h_xc_preys = np.sin(phi) * np.cos(theta)
            self.pos_h_yc_preys = np.sin(phi) * np.sin(theta)
            self.pos_h_zc_preys = np.cos(phi)

            return (grid_positions[:num_preys , 0], grid_positions[:num_preys , 1], grid_positions[:num_preys , 2],
                    self.pos_h_xc_preys, self.pos_h_yc_preys, self.pos_h_zc_preys)

    def no_sensor_predators(self, no_sensor_percentage: float, Ids: list):
        no_sensor_agents = int(no_sensor_percentage * self.n_agents)
        no_sensor_agents = np.random.choice(Ids, no_sensor_agents, replace=False)
        no_sensor_predators = no_sensor_agents
        return no_sensor_predators
        # print(f"Predators with no sensors: {no_sensor_agents}")



    def calculate_rotated_vector_batch(self, X1, Y1, Z1, X2, Y2, Z2, wdt):
        # Convert inputs to NumPy arrays if they aren't already
        # X1, Y1, Z1, X2, Y2, Z2, wdt = [np.asarray(a) for a in [X1, Y1, Z1, X2, Y2, Z2, wdt]]

        # Stack the original and target vectors for batch processing
        vector1 = np.stack([X1, Y1, Z1], axis=-1)
        vector2 = np.stack([X2, Y2, Z2], axis=-1)

        # Calculate magnitudes for normalization
        original_magnitude = np.linalg.norm(vector1, axis=1, keepdims=True)
        vector2_magnitude = np.linalg.norm(vector2, axis=1, keepdims=True)

        # Normalize and scale vector2
        vector2_normalized_scaled = vector2 * (original_magnitude / vector2_magnitude)

        # Calculate the normal vector for each pair
        normal_vector = np.cross(vector1, vector2_normalized_scaled)
        normal_magnitude = np.linalg.norm(normal_vector, axis=1, keepdims=True)

        # Avoid division by zero by ensuring non-zero magnitude
        normal_vector /= np.where(normal_magnitude > 0, normal_magnitude, 1)

        # Rodrigues' rotation formula for batch
        k_cross_vector1 = np.cross(normal_vector, vector1)
        cos_theta = np.cos(wdt)[:, np.newaxis]
        sin_theta = np.sin(wdt)[:, np.newaxis]
        one_minus_cos_theta = (1 - cos_theta)

        dot_product = np.sum(normal_vector * vector1, axis=1, keepdims=True)
        v_rot = vector1 * cos_theta + k_cross_vector1 * sin_theta + normal_vector * dot_product * one_minus_cos_theta

        return v_rot.T

    def calculate_av_heading(self, x_components, y_components, z_components):
        # Normalize each vector and sum them to get an average direction
        normalized_vectors = []
        for x, y, z in zip(x_components, y_components, z_components):
            vec = np.array([x, y, z])
            norm = np.linalg.norm(vec)
            if norm != 0:  # Avoid division by zero
                normalized_vectors.append(vec / norm)
            else:
                normalized_vectors.append(vec)  # Keep zero vectors as is

        # Calculate the average vector (sum of normalized vectors)
        sum_of_normalized_vectors = np.sum(normalized_vectors, axis=0)

        # Normalize the sum to get the unit vector with the average direction
        unit_vector_average_direction = sum_of_normalized_vectors / np.linalg.norm(sum_of_normalized_vectors)

        return unit_vector_average_direction

    def detect_bounds(self, pos_x, pos_y, pos_z):
        result_x = np.zeros_like(pos_x)
        result_y = np.zeros_like(pos_y)
        result_z = np.zeros_like(pos_z)

        result_x[pos_x < self.boun_thresh] = 1
        result_x[pos_x > self.boun_x - self.boun_thresh] = -1

        result_y[pos_y < self.boun_thresh] = 1
        result_y[pos_y > self.boun_y - self.boun_thresh] = -1

        result_z[pos_z < self.boun_thresh] = 1
        result_z[pos_z > self.boun_z - self.boun_thresh] = -1

        return result_x, result_y, result_z

    def calc_dij(self, pos_xs, pos_ys, pos_zs, pos_x_preys, pos_y_preys, pos_z_preys):
        self.d_ij = np.hypot(np.hypot(pos_xs[:, None] - pos_xs, pos_ys[:, None] - pos_ys), pos_zs[:, None] - pos_zs)
        self.d_ij[(self.d_ij > self.Dp) | (self.d_ij == 0)] = np.inf
        self.d_ij_noise = self.d_ij + self.rng.uniform(-self.noise_pos, self.noise_pos, (self.n_agents, self.n_agents)) * self.dt

        self.d_ij_preys = np.hypot(np.hypot(pos_x_preys[:, None] - pos_x_preys, pos_y_preys[:, None] - pos_y_preys), pos_z_preys[:, None] - pos_z_preys)
        self.d_ij_preys[(self.d_ij_preys > self.Dp_preys) | (self.d_ij_preys == 0)] = np.inf
        self.d_ij_noise_preys = self.d_ij_preys + self.rng.uniform(-self.noise_pos, self.noise_pos, (self.n_preys, self.n_preys)) * self.dt

    def calc_ang_ij(self, pos_xs, pos_ys, pos_zs, pos_x_preys, pos_y_preys, pos_z_preys):
        # print((pos_xs - pos_xs[:, None]) / self.d_ij)
        self.ij_ang_x = np.arccos((pos_xs - pos_xs[:, None]) / self.d_ij)
        self.ij_ang_y = np.arccos((pos_ys - pos_ys[:, None]) / self.d_ij)
        self.ij_ang_z = np.arccos((pos_zs - pos_zs[:, None]) / self.d_ij)

        self.ij_ang_x_preys = np.arccos((pos_x_preys - pos_x_preys[:, None]) / self.d_ij_preys)
        self.ij_ang_y_preys = np.arccos((pos_y_preys - pos_y_preys[:, None]) / self.d_ij_preys)
        self.ij_ang_z_preys = np.arccos((pos_z_preys - pos_z_preys[:, None]) / self.d_ij_preys)


    def calc_grad_vals(self, pos_xs, pos_ys, pos_zs, pos_x_preys, pos_y_preys, pos_z_preys):
        '''
        THIS METHOD IS THE DISTANCE MODULATION, I NEED TO MODIFY THIS TO FOLLOW MY PREYS
        '''

        # Calculate the 3D distance from each predator to each prey
        self.sigmas = self.sigma * np.ones(self.n_agents)
        self.d_ij_from_preys = np.hypot(np.hypot(pos_xs - pos_x_preys[:, None], pos_ys - pos_y_preys[:, None]), pos_zs - pos_z_preys[:, None])
        self.d_ij_from_preys[(self.d_ij_from_preys > self.sensor_range) | (self.d_ij_from_preys == 0)] = np.inf
        # print(self.d_ij_from_preys)
        self.d_ij_from_preys_noise = self.d_ij_from_preys + self.rng.uniform(-self.noise_pos, self.noise_pos, (self.n_preys, self.n_agents)) * self.dt
        # print(self.d_ij_from_preys_noise)
        # closest prey to each predator
        closest_prey_index = np.argmin(self.d_ij_from_preys_noise, axis=0)
        closest_prey_distance = self.d_ij_from_preys_noise[closest_prey_index, np.arange(self.n_agents)]
        closest_prey_distance[self.no_sensor_predators] = np.inf
        # print(closest_prey_distance)
        # self.sigmas = self.sigma + 1/closest_prey_distance
        self.sigmas = np.where(closest_prey_distance == np.inf, self.sigma_no_sensor, self.sigma +1/ closest_prey_distance)
        print(self.sigmas)


    def calc_repulsion_predator_forces(self, pos_xs, pos_ys, pos_zs, pos_x_preys, pos_y_preys, pos_z_preys):

        self.d_ij_from_predators = np.hypot(np.hypot(pos_x_preys - pos_xs, pos_y_preys- pos_ys), pos_z_preys - pos_zs)
        # self.d_ij_from_predators = np.sqrt((pos_xs - pos_x_preys[:, None]) ** 2 + (pos_ys - pos_y_preys[:, None]) ** 2 + (pos_zs - pos_z_preys[:, None]) ** 2)
        self.d_ij_from_predators[(self.d_ij_from_predators > self.sensing_range) | (self.d_ij_from_predators == 0)] = np.inf
        self.d_ij_from_predators_noise = self.d_ij_from_predators + self.rng.uniform(-self.noise_pos, self.noise_pos, (self.n_preys, self.n_agents)) * self.dt

        # closest predator
        closest_predator_index = np.argmin(self.d_ij_from_predators_noise, axis=1)
        closest_predator_distance = self.d_ij_from_predators_noise[np.arange(self.n_preys), closest_predator_index]

        dx = pos_x_preys - pos_xs[closest_predator_index]
        dy = pos_y_preys - pos_ys[closest_predator_index]
        dz = pos_z_preys - pos_zs[closest_predator_index]

        dx_avg = np.mean(dx)
        dy_avg = np.mean(dy)
        dz_avg = np.mean(dz)

        self.f_x_preys += self.kappa * dx_avg
        self.f_y_preys += self.kappa * dy_avg
        self.f_z_preys += self.kappa * dz_avg

        


    def calc_p_forcesADM(self): #ONLY FOR PREDATORS
        # -((self.epsilon /1.0 )* ( (self.sigma_i / distance ) - np.sqrt( self.sigma_i / distance )))
        forces = - self.epsilon * (self.sigmas[:, np.newaxis] / self.d_ij_noise - np.sqrt(self.sigmas[:, np.newaxis] / self.d_ij_noise))
        # forces = -self.epsilon * (np.sqrt(self.sigmas[:, np.newaxis] / self.d_ij_noise) - self.sigmas[:, np.newaxis] / self.d_ij_noise)
        
        # forces = -self.epsilon * (2 * (self.sigmas[:, np.newaxis] ** 4 / self.d_ij_noise ** 5) -  
        #                             (self.sigmas[:, np.newaxis] ** 2 / self.d_ij_noise ** 3))

        cos_ij_ang_x = np.cos(self.ij_ang_x)
        cos_ij_ang_y = np.cos(self.ij_ang_y)
        cos_ij_ang_z = np.cos(self.ij_ang_z)

        self.f_x = self.alpha * np.sum(forces * cos_ij_ang_x, axis=1)
        self.f_x = np.where(self.f_x == 0, 0.00001, self.f_x)

        self.f_y = self.alpha * np.sum(forces * cos_ij_ang_y, axis=1)
        self.f_y = np.where(self.f_y == 0, 0.00001, self.f_y)

        self.f_z = self.alpha * np.sum(forces * cos_ij_ang_z, axis=1)
        self.f_z = np.where(self.f_z == 0, 0.00001, self.f_z)

    def calc_p_forces(self):
        forces = -self.epsilon * (2 * (self.sigmas_preys[:, np.newaxis] ** 4 / self.d_ij_noise_preys ** 5) -
                                  (self.sigmas_preys[:, np.newaxis] ** 2 / self.d_ij_noise_preys ** 3))

        cos_ij_ang_x = np.cos(self.ij_ang_x_preys)
        cos_ij_ang_y = np.cos(self.ij_ang_y_preys)
        cos_ij_ang_z = np.cos(self.ij_ang_z_preys)

        self.f_x_preys = self.alpha_preys * np.sum(forces * cos_ij_ang_x, axis=1)
        self.f_x_preys = np.where(self.f_x_preys == 0, 0.00001, self.f_x_preys)

        self.f_y_preys = self.alpha_preys * np.sum(forces * cos_ij_ang_y, axis=1)
        self.f_y_preys = np.where(self.f_y_preys == 0, 0.00001, self.f_y_preys)

        self.f_z_preys = self.alpha_preys * np.sum(forces * cos_ij_ang_z, axis=1)
        self.f_z_preys = np.where(self.f_z_preys == 0, 0.00001, self.f_z_preys)

    def calc_alignment_forces(self):
        av_heading = self.calculate_av_heading(self.pos_h_xc, self.pos_h_yc, self.pos_h_zc)
        av_heading_preys = self.calculate_av_heading(self.pos_h_xc_preys, self.pos_h_yc_preys, self.pos_h_zc_preys)

        self.fa_x = int(self.h_alignment) * self.beta * av_heading[0]
        self.fa_y = int(self.h_alignment) * self.beta * av_heading[1]
        self.fa_z = int(self.h_alignment) * self.beta * av_heading[2]

        self.fa_x_preys = int(self.h_alignment) * self.beta * av_heading_preys[0]
        self.fa_y_preys = int(self.h_alignment) * self.beta * av_heading_preys[1]
        self.fa_z_preys = int(self.h_alignment) * self.beta * av_heading_preys[2]
    def calc_boun_rep_preds(self, pos_xs, pos_ys, pos_zs):
        d_bxi = np.minimum(np.abs(self.boun_x - pos_xs), pos_xs)
        d_byi = np.minimum(np.abs(self.boun_y - pos_ys), pos_ys)
        d_bzi = np.minimum(np.abs(self.boun_z - pos_zs), pos_zs)

        close_to_bound_x = np.logical_or(pos_xs < self.boun_thresh, pos_xs > (self.boun_x - self.boun_thresh))
        close_to_bound_y = np.logical_or(pos_ys < self.boun_thresh, pos_ys > (self.boun_y - self.boun_thresh))
        close_to_bound_z = np.logical_or(pos_zs < self.boun_thresh, pos_zs > (self.boun_z - self.boun_thresh))

        if np.any(close_to_bound_x) or np.any(close_to_bound_y) or np.any(close_to_bound_z):
            db_bxi, db_byi, db_bzi = self.detect_bounds(pos_xs, pos_ys, pos_zs)

            boundary_effect_x = -self.epsilon * 5 * (2 * (self.sigmas_b ** 4 / d_bxi ** 5) -
                                                (self.sigmas_b ** 2 / d_bxi ** 3))
            boundary_effect_y = -self.epsilon * 5 * (2 * (self.sigmas_b ** 4 / d_byi ** 5) -
                                                (self.sigmas_b ** 2 / d_byi ** 3))
            boundary_effect_z = -self.epsilon * 5 * (2 * (self.sigmas_b ** 4 / d_bzi ** 5) -
                                                (self.sigmas_b ** 2 / d_bzi ** 3))

            boundary_effect_x[boundary_effect_x < 0] = 0.0
            boundary_effect_y[boundary_effect_y < 0] = 0.0
            boundary_effect_z[boundary_effect_z < 0] = 0.0

            self.f_x += self.fa_x + boundary_effect_x * db_bxi
            self.f_y += self.fa_y + boundary_effect_y * db_byi
            self.f_z += self.fa_z + boundary_effect_z * db_bzi
        else:
            self.f_x += self.fa_x
            self.f_y += self.fa_y
            self.f_z += self.fa_z

    def calc_boun_rep_preys(self, pos_x_preys, pos_y_preys, pos_z_preys):
        d_bxi_preys = np.minimum(np.abs(self.boun_x - pos_x_preys), pos_x_preys)
        d_byi_preys = np.minimum(np.abs(self.boun_y - pos_y_preys), pos_y_preys)
        d_bzi_preys = np.minimum(np.abs(self.boun_z - pos_z_preys), pos_z_preys)

        close_to_bound_x_preys = np.logical_or(pos_x_preys < self.boun_thresh, pos_x_preys > (self.boun_x - self.boun_thresh))
        close_to_bound_y_preys = np.logical_or(pos_y_preys < self.boun_thresh, pos_y_preys > (self.boun_y - self.boun_thresh))
        close_to_bound_z_preys = np.logical_or(pos_z_preys < self.boun_thresh, pos_z_preys > (self.boun_z - self.boun_thresh))

        if np.any(close_to_bound_x_preys) or np.any(close_to_bound_y_preys) or np.any(close_to_bound_z_preys):
            db_bxi_preys, db_byi_preys, db_bzi_preys = self.detect_bounds(pos_x_preys, pos_y_preys, pos_z_preys)

            boundary_effect_x_preys = -self.epsilon * 5 * (2 * (self.sigmas_b_preys ** 4 / d_bxi_preys ** 5) -
                                                (self.sigmas_b_preys ** 2 / d_bxi_preys ** 3))
            boundary_effect_y_preys = -self.epsilon * 5 * (2 * (self.sigmas_b_preys ** 4 / d_byi_preys ** 5) -
                                                (self.sigmas_b_preys ** 2 / d_byi_preys ** 3))
            boundary_effect_z_preys = -self.epsilon * 5 * (2 * (self.sigmas_b_preys ** 4 / d_bzi_preys ** 5) -
                                                (self.sigmas_b_preys ** 2 / d_bzi_preys ** 3))

            boundary_effect_x_preys[boundary_effect_x_preys < 0] = 0.0
            boundary_effect_y_preys[boundary_effect_y_preys < 0] = 0.0
            boundary_effect_z_preys[boundary_effect_z_preys < 0] = 0.0

            self.f_x_preys += self.fa_x_preys + boundary_effect_x_preys * db_bxi_preys
            self.f_y_preys += self.fa_y_preys + boundary_effect_y_preys * db_byi_preys
            self.f_z_preys += self.fa_z_preys + boundary_effect_z_preys * db_bzi_preys
        else:
            self.f_x_preys += self.fa_x_preys
            self.f_y_preys += self.fa_y_preys
            self.f_z_preys += self.fa_z_preys

    def calc_u_w(self):
        f_mag = np.sqrt(np.square(self.f_x) + np.square(self.f_y) + np.square(self.f_z))
        f_mag_preys = np.sqrt(np.square(self.f_x_preys) + np.square(self.f_y_preys) + np.square(self.f_z_preys))

        f_mag = np.where(f_mag == 0, 0.00001, f_mag)
        f_mag_preys = np.where(f_mag_preys == 0, 0.00001, f_mag_preys)

        dot_f_h = self.f_x * self.pos_h_xc + self.f_y * self.pos_h_yc + self.f_z * self.pos_h_zc
        dot_f_h_preys = self.f_x_preys * self.pos_h_xc_preys + self.f_y_preys * self.pos_h_yc_preys + self.f_z_preys * self.pos_h_zc_preys

        cos_dot_f_h = np.clip(dot_f_h / f_mag, -1.0, 1.0)
        cos_dot_f_h_preys = np.clip(dot_f_h_preys / f_mag_preys, -1.0, 1.0)

        ang_f_h = np.arccos(cos_dot_f_h)
        ang_f_h_preys = np.arccos(cos_dot_f_h_preys)
        # ang_f_h += self.rng.uniform(-self.noise_h, self.noise_h, self.n_agents) * self.dt

        self.u = self.k1 * f_mag * np.cos(ang_f_h) + 0.05
        self.w = self.k2 * f_mag * np.sin(ang_f_h)

        self.u_preys = self.k1 * f_mag_preys * np.cos(ang_f_h_preys) + 0.05
        self.w_preys = self.k2 * f_mag_preys * np.sin(ang_f_h_preys)

        self.u = np.clip(self.u, 0, self.umax_const)
        self.w = np.clip(self.w, -self.wmax, self.wmax)

        self.u_preys = np.clip(self.u_preys, 0, self.umax_const)
        self.w_preys = np.clip(self.w_preys, -self.wmax, self.wmax)

        return self.u, self.u_preys

    def get_heading(self):
        pos_h_m       = np.sqrt(np.square(self.pos_h_xc)       + np.square(self.pos_h_yc)       + np.square(self.pos_h_zc))
        pos_h_m_preys = np.sqrt(np.square(self.pos_h_xc_preys) + np.square(self.pos_h_yc_preys) + np.square(self.pos_h_zc_preys))

        pos_hxs = np.arccos(self.pos_h_xc / pos_h_m)
        pos_hys = np.arccos(self.pos_h_yc / pos_h_m)
        pos_hzs = np.arccos(self.pos_h_zc / pos_h_m)


        pos_hxs_preys = np.arccos(self.pos_h_xc_preys / pos_h_m_preys)
        pos_hys_preys = np.arccos(self.pos_h_yc_preys / pos_h_m_preys)
        pos_hzs_preys = np.arccos(self.pos_h_zc_preys / pos_h_m_preys)

        return pos_hxs, pos_hys, pos_hzs, pos_hxs_preys, pos_hys_preys, pos_hzs_preys

    def update_heading(self):
        v_rot = self.calculate_rotated_vector_batch(
            self.pos_h_xc, self.pos_h_yc, self.pos_h_zc, self.f_x, self.f_y, self.f_z, self.w * self.dt)
        v_rot_preys = self.calculate_rotated_vector_batch(
            self.pos_h_xc_preys, self.pos_h_yc_preys, self.pos_h_zc_preys, self.f_x_preys, self.f_y_preys, self.f_z_preys, self.w_preys * self.dt)

        self.pos_h_xc = v_rot[0, :]
        self.pos_h_yc = v_rot[1, :]
        self.pos_h_zc = v_rot[2, :]
        
        self.pos_h_xc_preys = v_rot_preys[0, :]
        self.pos_h_yc_preys = v_rot_preys[1, :]
        self.pos_h_zc_preys = v_rot_preys[2, :]

    def plot_swarm(self, pos_xs, pos_ys, pos_zs, pos_hxs, pos_hys, pos_hzs, pos_preys_xs, pos_preys_ys, pos_preys_zs, pos_preys_hxs, pos_preys_hys, pos_preys_hzs):
        self.plotter.update_plot(pos_xs, pos_ys, pos_zs, pos_hxs, pos_hys, pos_hzs, pos_preys_xs, pos_preys_ys, pos_preys_zs, pos_preys_hxs, pos_preys_hys, pos_preys_hzs)