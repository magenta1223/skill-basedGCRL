from .maze_vis import draw_maze

import sys
sys.path.append("/home/magenta1223/skill-based/SiMPL/proposed")

from LVD.envs import *

env = Maze_GC(**maze_config)
train_tasks = [ MazeTask_Custom(task)  for task in MAZE_META_TASKS]

config = dict(
    policy=dict(hidden_dim=256, n_hidden=3, prior_state_dim = 4),
    qf=dict(hidden_dim=256, n_hidden=3),
    n_qf=2,
    encoder=dict(hidden_dim=128, n_hidden=2, init_scale=1, prior_scale=1),
    simpl=dict(init_enc_prior_reg=1e-5, target_enc_prior_kl=1,
               init_enc_post_reg=3e-6, target_enc_post_kl=10,
               init_policy_prior_reg=3e-3, target_policy_prior_kl=0.5,
               init_policy_post_reg=1e-3, target_policy_post_kl=5, kl_clip=5, prior_state_dim = 4, policy_lr = 3e-6),

    # simpl=dict(init_enc_prior_reg=1e-3, target_enc_prior_kl=1,
    #            init_enc_post_reg=3e-4, target_enc_post_kl=10,
    #            init_policy_prior_reg=5e-2, target_policy_prior_kl=0.5,
    #            init_policy_post_reg=3e-2, target_policy_post_kl=5, kl_clip=10, prior_state_dim = 4),

    enc_buffer_size=20000,
    buffer_size=20000,
    e_dim = 5,
    time_limit=4000,
    n_epoch=500,
    train=dict(batch_size=1024, reuse_rate=256,
               n_prior_batch=4, n_post_batch=26,
               prior_enc_size=4, post_enc_size=8192)
)
visualize_env = draw_maze
    
