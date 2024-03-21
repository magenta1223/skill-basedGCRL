import cv2
import matplotlib.pyplot as plt
import numpy as np

from d4rl.pointmaze.maze_model import WALL
from copy import deepcopy



def save_video(path, imgs, frame = (400,400), verbose = False):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, frame)
    
    for i in range(len(imgs)):
        # writing to a image array
        out.write(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
    out.release()         

    if verbose:
        print(f"Rendered : {path}")


def render_from_env(*args, **kwargs):
    if kwargs['env'].name == "kitchen":
        imgs = render_kitchen(*args, **kwargs)
    elif kwargs['env'].name == "maze":
        imgs = draw_maze(*args, **kwargs)
        
    return imgs


def render_kitchen(env, task, states = None, actions = None, c = 0, text = None, size = (400, 400)):
    assert states is not None or actions is not None, "States or actions must not be None."
    if actions is not None:
        mode = "action"
    else:
        mode = "state"

    imgs = []
    with env.set_task(task):
        env.reset()
        init_qvel = env.init_qvel        
        if mode == "state":
            for i, state in enumerate(states):
                env.set_state(state, init_qvel)
                img = env.render(mode= "rgb_array")
                img = img.copy()
                if text is not None:
                    x, y = size 
                    cv2.putText(img = img,  text = text, color = (255,0,0),  org = (x // 2, y // 2), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 2, lineType= cv2.LINE_AA)

                if i > 280:
                    cv2.putText(img = img,  text = "END", color = (255,0,0),  org = (x // 4, y // 4), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 2, lineType= cv2.LINE_AA)

                imgs.append(img)

        else:
            for i, action in enumerate(actions):
                # if i ==  c:
                #     qvel = env.data.qvel
                #     env.set_state(states[i], qvel)
                env.step(action)
                img = env.render(mode= "rgb_array")
                img = img.copy()
                if text is not None:
                    x, y = size 
                    cv2.putText(img = img,  text = text, color = (255,0,0),  org = (x // 2, y // 2), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 2, lineType= cv2.LINE_AA)
                if i > 280:
                    cv2.putText(img = img,  text = "END", color = (255,0,0),  org = (x // 4, y // 4), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 2, lineType= cv2.LINE_AA)

                
                imgs.append(img)

    return imgs


def draw_maze(env, episodes, markers, color = "royalblue"):
    plt.cla() 
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    # axis options
    ax.axis('off')        
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


    img = np.rot90(env.maze_arr != WALL)
    extent = [
        -0.5, env.maze_arr.shape[0]-0.5,
        -0.5, env.maze_arr.shape[1]-0.5
    ]
    
    ax.imshow((1-img)/5, extent=extent, cmap='Reds', alpha=0.2)
    ax.set_xlim(0, env.maze_size+1)
    ax.set_ylim(0, env.maze_size+1)

    # speical markers 
    for marker in markers:
        ax.scatter(*marker.data, **marker.params)
    

    for episode in episodes:
        states = deepcopy(np.array(episode.states))
        ax.plot(*states[:, :2].T  , color=color, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    

    return fig