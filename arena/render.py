import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def render_episode(frames):
    fig, ax = plt.subplots(1, len(frames), figsize=(6 * len(frames), 6), layout="tight")
    if len(frames) == 1:
        ax = [ax]
    fig.subplots_adjust(wspace=0, hspace=0)

    vid_frames = {}
    for k, f in frames.items():
        vid_frames[k] = np.concat(f, 0)

    imagexn = {k: np.transpose(vid_frames[k][0], (1, 2, 0)) for k in frames}
    plotxn = {k: ax[0].imshow(img) for k, img in imagexn.items()}
    for i in range(len(plotxn)):
        ax[i].axis("off")

    def update(frame):
        for k in plotxn:
            frame_image = np.transpose(vid_frames[k][frame], (1, 2, 0))
            plotxn[k].set_array(frame_image)
        return plotxn

    return animation.FuncAnimation(
        fig, update, frames=min([f.shape[0] for f in vid_frames.values()]), interval=100
    )
