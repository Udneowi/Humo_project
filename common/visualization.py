# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

def render_animation(data, skeleton, fps, output='interactive', bitrate=1000):
    """
    Render or show an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    x = 0
    y = 1
    z = 2
    radius = torch.max(skeleton.offsets()).item() * 10

    skeleton_parents = skeleton.parents()

    plt.ioff()
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20., azim=30)

    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5

    lines = []
    initialized = False

    trajectory = data[:, 0, [0, 2]]
    avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
    draw_offset = int(25/avg_segment_length)
    spline_line, = ax.plot(*trajectory.T)
    camera_pos = trajectory
    height_offset = np.min(data[:, :, 1]) # Min height
    data = data.copy()
    data[:, :, 1] -= height_offset

    def update(frame):
        nonlocal initialized
        ax.set_xlim3d([-radius/2 + camera_pos[frame, 0], radius/2 + camera_pos[frame, 0]])
        ax.set_ylim3d([-radius/2 + camera_pos[frame, 1], radius/2 + camera_pos[frame, 1]])

        positions_world = data[frame]
        for i in range(positions_world.shape[0]):
            if skeleton_parents[i] == -1:
                continue
            if not initialized:
                col = 'red' if i in skeleton.joints_right() else 'black' # As in audio cables :)
                lines.append(ax.plot([positions_world[i, x], positions_world[skeleton_parents[i], x]],
                        [positions_world[i, y], positions_world[skeleton_parents[i], y]],
                        [positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y', c=col))
            else:
                lines[i-1][0].set_xdata([positions_world[i, x], positions_world[skeleton_parents[i], x]])
                lines[i-1][0].set_ydata([positions_world[i, y], positions_world[skeleton_parents[i], y]])
                lines[i-1][0].set_3d_properties([positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y')
        l = max(frame-draw_offset, 0)
        r = min(frame+draw_offset, trajectory.shape[0])
        spline_line.set_xdata(trajectory[l:r, 0])
        spline_line.set_ydata(np.zeros_like(trajectory[l:r, 0]))
        spline_line.set_3d_properties(trajectory[l:r, 1], zdir='y')
        initialized = True
        if output == 'interactive' and frame == data.shape[0] - 1:
            plt.close('all')

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=np.arange(0, data.shape[0]), interval=1000/fps, repeat=False)
    if output == 'interactive':
        plt.show()
        return anim
    elif output == 'html':
        return anim.to_html5_video()
    elif output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only html, .mp4, and .gif are supported)')
    plt.close()


def render_animation_dual(data_real, data_gen, skeleton, fps, output='interactive', bitrate=1000):
    """
    Render or show an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    x = 0
    y = 1
    z = 2
    radius = torch.max(skeleton.offsets()).item() * 10

    skeleton_parents = skeleton.parents()

    plt.ioff()
    fig = plt.figure(figsize=(8, 4))
    ax_real = fig.add_subplot(1, 2, 1, projection='3d')
    ax_real.view_init(elev=20., azim=30)

    ax_real.set_xlim3d([-radius/2, radius/2])
    ax_real.set_zlim3d([0, radius])
    ax_real.set_ylim3d([-radius/2, radius/2])
    ax_real.set_aspect('equal')
    ax_real.set_xticklabels([])
    ax_real.set_yticklabels([])
    ax_real.set_zticklabels([])
    ax_real.dist = 7.5

    lines_real = []
    initialized_real = False

    trajectory_real = data_real[:, 0, [0, 2]]
    avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory_real, axis=0), axis=1)) + 1e-3
    draw_offset = int(25/avg_segment_length)
    spline_line_real, = ax_real.plot(*trajectory_real.T)
    height_offset_real = np.min(data_real[:, :, 1]) # Min height
    data_real = data_real.copy()
    data_real[:, :, 1] -= height_offset_real

    ax_gen = fig.add_subplot(1, 2, 2, projection='3d')
    ax_gen.view_init(elev=20., azim=30)

    ax_gen.set_xlim3d([-radius/2, radius/2])
    ax_gen.set_zlim3d([0, radius])
    ax_gen.set_ylim3d([-radius/2, radius/2])
    ax_gen.set_aspect('equal')
    ax_gen.set_xticklabels([])
    ax_gen.set_yticklabels([])
    ax_gen.set_zticklabels([])
    ax_gen.dist = 7.5

    lines_gen = []
    initialized_gen = False

    trajectory_gen = data_gen[:, 0, [0, 2]]
    avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory_gen, axis=0), axis=1)) + 1e-3
    draw_offset = int(25/avg_segment_length)
    spline_line_gen, = ax_gen.plot(*trajectory_gen.T)
    height_offset_gen = np.min(data_gen[:, :, 1]) # Min height
    data_gen = data_gen.copy()
    data_gen[:, :, 1] -= height_offset_gen

    def update_single(frame, ax, data, spline_line, lines, trajectory, initialized):
        ax.set_xlim3d([-radius/2 + trajectory[frame, 0], radius/2 + trajectory[frame, 0]])
        ax.set_ylim3d([-radius/2 + trajectory[frame, 1], radius/2 + trajectory[frame, 1]])

        positions_world = data[frame]
        for i in range(positions_world.shape[0]):
            if skeleton_parents[i] == -1:
                continue
            if not initialized:
                col = 'red' if i in skeleton.joints_right() else 'black' # As in audio cables :)
                lines.append(ax.plot([positions_world[i, x], positions_world[skeleton_parents[i], x]],
                        [positions_world[i, y], positions_world[skeleton_parents[i], y]],
                        [positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y', c=col))
            else:
                lines[i-1][0].set_xdata([positions_world[i, x], positions_world[skeleton_parents[i], x]])
                lines[i-1][0].set_ydata([positions_world[i, y], positions_world[skeleton_parents[i], y]])
                lines[i-1][0].set_3d_properties([positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y')
        l = max(frame-draw_offset, 0)
        r = min(frame+draw_offset, trajectory.shape[0])
        spline_line.set_xdata(trajectory[l:r, 0])
        spline_line.set_ydata(np.zeros_like(trajectory[l:r, 0]))
        spline_line.set_3d_properties(trajectory[l:r, 1], zdir='y')
        if output == 'interactive' and frame == data.shape[0] - 1:
            plt.close('all')

    def update(frame):
        nonlocal initialized_real
        nonlocal initialized_gen
        update_single(frame, ax_real, data_real, spline_line_real, lines_real, trajectory_real, initialized_real)
        update_single(frame, ax_gen, data_gen, spline_line_gen, lines_gen, trajectory_gen, initialized_gen)

        initialized_real = True
        initialized_gen = True

    fig.tight_layout()
    num_frames = np.min([data_real.shape[0], data_gen.shape[0]])
    anim = FuncAnimation(fig, update, frames=np.arange(0, num_frames), interval=1000/fps, repeat=False)
    if output == 'interactive':
        plt.show()
        return anim
    elif output == 'html':
        return anim.to_html5_video()
    elif output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only html, .mp4, and .gif are supported)')
    plt.close()
