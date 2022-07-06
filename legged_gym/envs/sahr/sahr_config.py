from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
from isaacgym import gymapi

class SahrRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 1024
        num_actions = 12
        num_observations = 48
        episode_length_s = 20.0
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' 
        measure_heights = False
        static_friction = 1.0
        dynamic_friction = 1.0
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.39] # x,y,z [m]
        rot = [0.0, 0.09, 0.0, 1.0] # x,y,z,w [quat]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw': 0.79*np.pi/180,   # [rad]
            'left_hip_roll': 6.06*np.pi/180,   # [rad]
            'left_hip_pitch': -31.99*np.pi/180,  # [rad]
            'left_knee': 0.7362,  # [rad]
            'left_ankle_pitch': -21.53*np.pi/180,
            'left_ankle_roll': -6.18*np.pi/180,     # [rad]
            
            'right_hip_yaw': 0.79*np.pi/180,   # [rad]
            'right_hip_roll': -6.06*np.pi/180,     # [rad]
            'right_hip_pitch': -31.99*np.pi/180 ,  # [rad]
            'right_knee': 0.7362,  # [rad]
            'right_ankle_pitch': -21.53*np.pi/180,
            'right_ankle_roll': 6.18*np.pi/180,

            'head_yaw': 0.0,
            'head_pitch': 0.0,
            
            'right_shoulder_pitch': 0.0,
            'right_shoulder_roll': 0.0,
            'right_elbow': 0.0,

            'left_shoulder_pitch': 0.0,
            'left_shoulder_roll': 0.0,
            'left_elbow': 0.0
        }


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = { 'left_hip_yaw': 20.0,
            'left_hip_roll': 20.0, 'left_hip_pitch': 20.0, 'left_knee': 20.0, 
            'left_ankle_pitch': 20.0, 'left_ankle_roll': 20.0, 'right_hip_yaw': 20.0,
            'right_hip_roll': 20.0, 'right_hip_pitch': 20.0, 'right_knee': 20.0,
            'right_ankle_pitch': 20.0, 'right_ankle_roll': 20.0, 'head_yaw': 20.0,
            'head_pitch': 20.0,
            
            'right_shoulder_pitch': 20.0,
            'right_shoulder_roll': 20.0,
            'right_elbow': 20.0,

            'left_shoulder_pitch': 20.0,
            'left_shoulder_roll': 20.0,
            'left_elbow': 20.0}
        damping = { 'left_knee': 1.23, 'left_hip_yaw': 0.65,
            'left_hip_roll': 1.23, 'left_hip_pitch': 1.23, 
            'left_ankle_pitch': 1.23,  'left_ankle_roll': 1.23, 'right_knee': 1.23, 'right_hip_yaw': 0.65,
            'right_hip_roll': 1.23, 'right_hip_pitch': 1.23, 
            'right_ankle_pitch': 1.23, 'right_ankle_roll': 1.23, 'head_yaw': 1.23,
            'head_pitch': 1.23,
            
            'right_shoulder_pitch': 0.65,
            'right_shoulder_roll': 0.65,
            'right_elbow': 0.65,

            'left_shoulder_pitch': 0.65,
            'left_shoulder_roll': 0.65,
            'left_elbow': 0.65 }   # [N*m*s/rad]
        
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/sahr/urdf/sahr_only_legs.urdf'
        name = "sahr"
        foot_name = "foot"
        penalize_contacts_on = ["trunk_1", "u_shoulder_1", "u_shoulder_2",
                                "right_humerus_1", "left_humerus_1",
                                "left_boxing_glove", "right_boxing_glove"
                                "right_knee_1", "left_knee_1", "head_1"]
        terminate_after_contacts_on = ["trunk_1", "u_shoulder_1", "u_shoulder_2",
                                       "right_humerus_1", "left_humerus_1",
                                       "left_boxing_glove", "right_boxing_glove"
                                       "right_knee_1", "left_knee_1", "head_1", 
                                       "u_block_1", "u_block_2", "left_radius_1",
                                       "right_radius_1"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        max_angular_velocity = 3.
        max_linear_velocity = 3.
        default_dof_drive_mode = 1
        flip_visual_attachments = False
        mesh_normal_mode = gymapi.FROM_ASSET
        override_inertia = False
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            # dof_pos_limits = 0.0
            lin_vel_z = -3.0
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.5
            ang_vel_xy = -0.01
            # orientation = -0.
            # dof_vel = -0.001
            # dof_acc = -2.5e-7
            # base_height = -0.0 
            # feet_air_time =  1.0
            # collision = -1.
            # feet_stumble = -0.0 
            action_rate = -0.001
            # stand_still = -0.0
           
        
        time_of_step = 0.3 #longer steps get positive reward
        soft_dof_pos_limit = 0.99
        base_height_target = 0.35        
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 1.0
        max_contact_force = 250.0 # forces above this value are penalized
        only_positive_rewards = False

    class sim (LeggedRobotCfg.sim ):
        dt =  0.0025

class SahrRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_sahr'
        # load_run = "Jul05_15-23-43_"