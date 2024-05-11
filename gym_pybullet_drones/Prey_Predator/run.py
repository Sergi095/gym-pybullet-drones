"""CrazyFlie software-in-the-loop control example.

Setup
-----
Step 1: Clone pycffirmware from https://github.com/utiasDSL/pycffirmware
Step 2: Follow the install instructions for pycffirmware in its README

Example
-------
In terminal, run:
python gym_pybullet_drones/examples/cf.py

"""

###
# Logging is added!
###
# gym-pybullet-drones\gym_pybullet_drones\envs\CrtlAviary_prey_preds.py

import time
import argparse
import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CrtlAviary_prey_preds import CtrlAviary_
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool
from flocking import FlockingUtils
import pybullet as p
from datetime import datetime


DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_OUTPUT_FOLDER = 'results'
DURATION_SEC = 200

NS = 5

NUM_DRONES = NS
PREYS = NS
if_cube = True
init_center_x = 5
init_center_y = 3
init_center_z = 1

min_distance = 1
init_center_x_preys = init_center_x + min_distance
init_center_y_preys = init_center_y + min_distance
init_center_z_preys = init_center_z + 0
spacing = 0.1
self_log = False
save_dir = "./self_logs/"

f_util = FlockingUtils(NUM_DRONES, PREYS, init_center_x, init_center_y, init_center_z, spacing, init_center_x_preys, init_center_y_preys, init_center_z_preys)
pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc = f_util.initialize_positions(preds=True)
pos_xs_preys, pos_ys_preys, pos_zs_preys, pos_h_xc_preys, pos_h_yc_preys, pos_h_zc_preys = f_util.initialize_positions(preds=False)

INIT_XYZ = np.zeros([NUM_DRONES, 3])
INIT_XYZ[:, 0] = pos_xs
INIT_XYZ[:, 1] = pos_ys
INIT_XYZ[:, 2] = pos_zs
INIT_RPY = np.array([[.0, .0, .0] for _ in range(NUM_DRONES)])

INIT_XYZ_PREYS = np.zeros([PREYS, 3])
INIT_XYZ_PREYS[:, 0] = pos_xs_preys
INIT_XYZ_PREYS[:, 1] = pos_ys_preys
INIT_XYZ_PREYS[:, 2] = pos_zs_preys
INIT_RPY_PREYS = np.array([[.0, .0, .0] for _ in range(PREYS)])





def run(
        drone=DEFAULT_DRONES,
        num_drones=NUM_DRONES,
        preys=PREYS,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        duration_sec=DURATION_SEC
):
    env = CtrlAviary_(drone_model=ARGS.drone,
                     num_drones=ARGS.num_drones,
                     preys=ARGS.preys,
                     initial_xyzs=INIT_XYZ,
                     initial_rpys=INIT_RPY,
                     initial_xyzs_preys=INIT_XYZ_PREYS,
                     initial_rpys_preys=INIT_RPY_PREYS,
                     physics=ARGS.physics,
                     # physics=Physics.PYB_DW,
                     neighbourhood_radius=10,
                     pyb_freq=ARGS.simulation_freq_hz,
                     ctrl_freq=ARGS.control_freq_hz,
                     gui=ARGS.gui,
                     user_debug_gui=ARGS.user_debug_gui
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    ctrl       = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]
    ctrl_preys = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.preys)]

    if self_log:
        log_pos_xs = np.zeros([NUM_DRONES, DURATION_SEC*DEFAULT_CONTROL_FREQ_HZ])
        log_pos_ys = np.zeros([NUM_DRONES, DURATION_SEC*DEFAULT_CONTROL_FREQ_HZ])
        log_pos_zs = np.zeros([NUM_DRONES, DURATION_SEC*DEFAULT_CONTROL_FREQ_HZ])
        log_pos_hxs = np.zeros([NUM_DRONES, DURATION_SEC*DEFAULT_CONTROL_FREQ_HZ])
        log_pos_hys = np.zeros([NUM_DRONES, DURATION_SEC * DEFAULT_CONTROL_FREQ_HZ])
        log_pos_hzs = np.zeros([NUM_DRONES, DURATION_SEC * DEFAULT_CONTROL_FREQ_HZ])

    START = time.time()
    action = np.zeros((NUM_DRONES, 4))
    action_preys = np.zeros((PREYS, 4))

    pos_x = np.zeros(NUM_DRONES)
    pos_y = np.zeros(NUM_DRONES)
    pos_z = np.zeros(NUM_DRONES)

    pos_x_preys = np.zeros(PREYS)
    pos_y_preys = np.zeros(PREYS)
    pos_z_preys = np.zeros(PREYS)

    for i in range(0, int(ARGS.duration_sec * env.CTRL_FREQ)):
        obs, obs_preys, reward, reward_preys, done, done_preys, info, info_preys, _, _ = env.step(action, action_preys)
        
        for j, k in zip(range(NUM_DRONES), range(PREYS)):
            states = env._getDroneStateVector(j)
            # for k in range(PREYS):
            states_preys = env._getDroneStateVectorPreys(k)
            pos_x[j] = states[0]
            pos_y[j] = states[1]
            pos_z[j] = states[2]
            pos_x_preys[k] = states_preys[0]
            pos_y_preys[k] = states_preys[1]
            pos_z_preys[k] = states_preys[2]

        # obs_preys, reward_preys, done_preys, info_preys, _ = env.step_preys(action_preys)


        midpoint_x = (pos_x[0] + pos_x[1] + pos_x[2]) / 3
        midpoint_y = (pos_y[0] + pos_y[1] + pos_y[2]) / 3
        midpoint_z = (pos_z[0] + pos_z[1] + pos_z[2]) / 3

        midpoint_x_preys = (pos_x_preys[0] + pos_x_preys[1] + pos_x_preys[2]) / 3
        midpoint_y_preys = (pos_y_preys[0] + pos_y_preys[1] + pos_y_preys[2]) / 3
        midpoint_z_preys = (pos_z_preys[0] + pos_z_preys[1] + pos_z_preys[2]) / 3

        midpoint_x = (midpoint_x + midpoint_x_preys) / 2
        midpoint_y = (midpoint_y + midpoint_y_preys) / 2
        midpoint_z = (midpoint_z + midpoint_z_preys) / 2

        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-25, cameraPitch=-20,
                                    cameraTargetPosition=[midpoint_x, midpoint_y, midpoint_z])

        # p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-30, cameraPitch=-40,
        #                                  cameraTargetPosition=[pos_x[1], pos_y[1], pos_z[1] - 1])

        # p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-30, cameraPitch=-40,
        #                               cameraTargetPosition=[pos_x_preys[0], pos_y_preys[0], pos_z_preys[0] - 1])

        f_util.calc_dij(pos_x, pos_y, pos_z, pos_x_preys, pos_y_preys, pos_z_preys)
        f_util.calc_ang_ij(pos_x, pos_y, pos_z, pos_x_preys, pos_y_preys, pos_z_preys)
        f_util.calc_grad_vals()
        f_util.calc_p_forces()
        f_util.calc_p_forcesADM()
        f_util.calc_repulsion_predator_forces()
        # f_util.calc_alignment_forces()
        # f_util.calc_boun_rep_preds(pos_xs, pos_ys, pos_zs)
        # f_util.calc_boun_rep_preys(pos_x_preys, pos_y_preys, pos_z_preys)
        u, u_preys = f_util.calc_u_w()
        pos_hxs, pos_hys, pos_hzs, pos_h_xc_preys, pos_h_yc_preys, pos_h_zc_preys = f_util.get_heading()
        f_util.update_heading()

        if i % (env.CTRL_FREQ*1) == 0:
            f_util.plot_swarm(pos_x, 
                              pos_y, 
                              pos_z, 
                              pos_hxs, 
                              pos_hys,
                              pos_hzs,
                                pos_x_preys,
                                pos_y_preys,
                                pos_z_preys,
                                pos_h_xc_preys,
                                pos_h_yc_preys,
                                pos_h_zc_preys)

        for j, k in zip(range(NUM_DRONES), range(PREYS)):
            # for k in range(PREYS):
            # print(j,k)
            vel_cmd = np.array([u[j]*np.cos(pos_hxs[j]), u[j]*np.cos(pos_hys[j]), u[j]*np.cos(pos_hzs[j])])
            pos_cmd = np.array([pos_x[j], pos_y[j], pos_z[j]])

            action[j], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                state=obs[j],
                                                                target_pos=pos_cmd,
                                                                target_vel=vel_cmd,
                                                                target_rpy=np.array([0, 0, 0])
                                                                )
            vel_cmd_preys = np.array([u_preys[k]*np.cos(pos_h_xc_preys[k]), u_preys[k]*np.cos(pos_h_yc_preys[k]), u_preys[k]*np.cos(pos_h_zc_preys[k])])
            pos_cmd_preys = np.array([pos_x_preys[k], pos_y_preys[k], pos_z_preys[k]])

            action_preys[k], _, _ = ctrl_preys[k].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                            state=obs_preys[k],
                                                                            target_pos=pos_cmd_preys,
                                                                            target_vel=vel_cmd_preys,
                                                                            target_rpy=np.array([0, 0, 0])
                                                                            )

     


        #### Log the simulation ####################################
        if self_log:
            log_pos_xs[:, i] = pos_x[:]
            log_pos_ys[:, i] = pos_y[:]
            log_pos_zs[:, i] = pos_z[:]
            log_pos_hxs[:, i] = pos_hxs[:]
            log_pos_hys[:, i] = pos_hys[:]
            log_pos_hzs[:, i] = pos_hzs[:]

        #### Printout ##############################################
        # env.render()

        #### Sync the simulation ###################################
        if gui:
            pass
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    if self_log:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y|%H_%M_%S")
        filename_posx = save_dir + "log_pos_xs_" + dt_string + ".npy"
        filename_posy = save_dir + "log_pos_ys_" + dt_string + ".npy"
        filename_posz = save_dir + "log_pos_zs_" + dt_string + ".npy"
        filename_pos_hxs = save_dir + "log_pos_hxs_" + dt_string + ".npy"
        filename_pos_hys = save_dir + "log_pos_hys_" + dt_string + ".npy"
        filename_pos_hzs = save_dir + "log_pos_hzs_" + dt_string + ".npy"
        np.save(filename_posx, log_pos_xs)
        np.save(filename_posy, log_pos_ys)
        np.save(filename_posz, log_pos_zs)
        np.save(filename_pos_hxs, log_pos_hxs)
        np.save(filename_pos_hys, log_pos_hys)
        np.save(filename_pos_hzs, log_pos_hzs)


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel, help='Drone model (default: BETA)',
                        metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=NUM_DRONES, type=int, help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--preys', default=PREYS, type=int, help='Number of preys (default: 3)', metavar='')
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics, help='Physics updates (default: PYB)',
                        metavar='', choices=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
                        metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                        help='Simulation frequency in Hz (default: 500)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                        help='Control frequency in Hz (default: 25)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--duration_sec',default=DURATION_SEC, type=int,
                        help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
