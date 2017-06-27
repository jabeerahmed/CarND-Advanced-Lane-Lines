import imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def noop(frame, i): return frame

def play_video(process=noop, filename='project_video.mp4', start=0, num_frames=100000000000):
    pass

    vid = imageio.get_reader(filename)
    nframes = vid.get_length()
    fig, axs = plt.subplots(1, 1)
    im = -1

    def foo(i):
        img = process(vid.get_data(i), i)

#        global im
#        if im == -1:
#         axs.imshow(img, cmap='gray')
#        return im.set_data(img)
        return img
    return animation.FuncAnimation(fig, foo, nframes, repeat=False, blit=False)

# if __name__ == '__main__':
#     anim = play_video()

#%%
anim = play_video()

plt.show()

#
# import matplotlib
# print(str(matplotlib.use('MacOSX')))
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro', animated=True)
#
# def init():
#     ax.set_xlim(0, 2*np.pi)
#     ax.set_ylim(-1, 1)
#     return ln,
#
# def update(frame):
#     xdata.append(frame)
#     ydata.append(np.sin(frame))
#     ln.set_data(xdata, ydata)
#     return ln,
#
#
# ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
#                     init_func=init, blit=False)
# plt.show()
