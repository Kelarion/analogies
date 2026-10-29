import os, sys
import pickle

import torch
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gsp
import matplotlib.colors as mpc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as ani
from cycler import cycler
import scipy
import scipy.linalg as la
from scipy.spatial.distance import pdist, squareform
from itertools import permutations, combinations
import itertools as itt
import warnings
import networkx as nx

from plotting import set_axes_equal, draw_multigraph


#####################################

class AnimeCollection(object):
    def __init__(self, *animes):
        self.animes = list(animes)

        self.fig = self.animes[0].fig

        self.ani = ani.FuncAnimation(self.fig, self.update, interval=self.animes[0].interval, 
            init_func=self.setup_plot, blit=False, frames=self.animes[0].frames)

    def setup_plot(self):

        self.patches = []
        for an in self.animes:
            self.patches.append(an.setup_plot())

        return self.patches

    def update(self, i):
        for an in self.animes:
            an.update(i)

        return self.patches

    def save(self,save_dir, fps=30):
        # self.ani.save(save_dir, writer=ani.writers['ffmpeg'](fps=fps))
        self.ani.save(save_dir, writer=ani.FFMpegWriter(fps=fps))


class Anime(object):
    """
    Base class: subclasses draw in setup_plot() and change things in update(i).

    In an IPython console it plays with a GUI backend (%matplotlib qt); keep the object
    (a = SomeAnime(...)), or it's garbage-collected and stops. In Jupyter, a.html() gives a
    player. Spyder's inline plots can't animate at all.
    """

    def __init__(self, frames, interval, blit=False, ax=None, fig=None, repeat=True,
                 repeat_delay=0):

        self.frames = frames
        self.interval = interval
        _check_backend()

        # Setup the figure and axes...
        if fig is None and ax is None:
            self.fig, self.ax = plt.subplots()
        elif ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif fig is not None:
            self.fig = fig
            self.ax = fig.add_subplot(111)
        
        # Then setup FuncAnimation.
        self.ani = ani.FuncAnimation(self.fig, self.update, interval=interval, 
            init_func=self.setup_plot, blit=blit, frames=frames, repeat=repeat,
            repeat_delay=repeat_delay)

    def save(self, save_dir, fps=30):
        """Write to a file: .gif with pillow, anything else with ffmpeg."""
        if str(save_dir).endswith('.gif'):
            self.ani.save(save_dir, writer=ani.PillowWriter(fps=fps))
        else:
            self.ani.save(save_dir, writer=ani.FFMpegWriter(fps=fps))

    def html(self):
        """An HTML player, for a Jupyter notebook (closes the figure, or the inline
        backend shows a still of it as well)."""
        from IPython.display import HTML
        plt.close(self.fig)
        return HTML(self.ani.to_jshtml())

    def setup_plot(self):
        raise NotImplementedError

    def update(self, i):
        raise NotImplementedError


def _check_backend():
    if 'inline' in mpl.get_backend() and 'spyder_kernels' in sys.modules:
        warnings.warn("Spyder's inline plots can't animate: run %matplotlib qt first, "
                      "or .save('name.gif')")


class WalkAnime(Anime):
    """
    A walk on a graph: draws it with plotting.draw_multigraph(G, cols, **draw_args), then
    rings the nodes of `walk` in turn, `speed` steps per second, with the last `trail` nodes
    fading out behind and the edge(s) just traversed drawn over in `color` (every parallel
    edge between the two nodes, since a walk of nodes doesn't say which).

    walk:  a sequence of nodes of G (for graphs labelled 0..n-1, their indices)
    pos:   node positions, as for nx.draw; if not given, spring_layout with seed 0, since
           every frame needs the same layout

        a = WalkAnime(G, walk, cols='action', speed=3, with_labels=True)
        a.save('walk.gif')      # at `speed` steps per second
        a.html()                # in Jupyter
    """

    def __init__(self, G, walk, cols='weight', speed=2, trail=3, color='tab:red', pos=None,
                 ax=None, fig=None, repeat=True, **draw_args):

        self.walk = list(walk)
        missing = [v for v in self.walk if v not in G]
        if missing:
            raise ValueError(f"not nodes of G: {missing[:5]}")
        self.G, self.cols, self.draw_args = G, cols, draw_args
        self.speed, self.trail = speed, trail
        self.rgba = np.array(mpl.colors.to_rgba(color))
        self.pos = pos or nx.spring_layout(G, seed=0)
        self.xy = np.array([self.pos[v] for v in self.walk])
        self.drawn = False

        super().__init__(frames=len(self.walk), interval=1000 / speed, ax=ax, fig=fig,
                         repeat=repeat, repeat_delay=1000)

    def save(self, save_dir, fps=None):
        super().save(save_dir, fps=self.speed if fps is None else fps)

    def setup_plot(self):
        # FuncAnimation calls this again on saving and on resizing, so draw only once
        if not self.drawn:
            n = len(self.ax.patches)
            draw_multigraph(self.G, cols=self.cols, pos=self.pos, ax=self.ax, **self.draw_args)

            # nx.draw adds one arrow per edge, in the order of G.edges(keys=True)
            drawn = self.ax.patches[n:]
            self.edges = {}
            if len(drawn) == self.G.number_of_edges():
                for (u, v, _), p in zip(self.G.edges(keys=True), drawn):
                    self.edges.setdefault((u, v), []).append(p)
            else:
                warnings.warn("can't match the drawn edges to G's (arrows=False?); "
                              "not highlighting edges")
            self.original = {p: (p.get_edgecolor(), p.get_facecolor(), p.get_linewidth())
                             for ps in self.edges.values() for p in ps}
            self.lit = []

            ring = dict(s=self.draw_args.get('node_size', 300) * 2.5, facecolors='none',
                        linewidths=3, zorder=3, edgecolors=[self.rgba])
            self.past = self.ax.scatter(self.xy[:1, 0], self.xy[:1, 1], **ring)
            self.now = self.ax.scatter(self.xy[:1, 0], self.xy[:1, 1], **ring)
            self.drawn = True

        return self.update(0)

    def update(self, i):

        for p in self.lit:                                  # restore the last step's edges
            ec, fc, lw = self.original[p]
            p.set_edgecolor(ec), p.set_facecolor(fc), p.set_linewidth(lw)
        self.lit = []
        if i > 0:
            for p in self.edges.get((self.walk[i - 1], self.walk[i]), []):
                p.set_color(self.rgba)
                p.set_linewidth(3 * self.original[p][2])
                self.lit.append(p)

        lo = max(i - self.trail, 0)
        fade = np.tile(self.rgba, (i - lo, 1))
        fade[:, 3] = np.linspace(0.15, 0.6, i - lo)
        self.past.set_offsets(self.xy[lo:i] if i > lo else np.empty((0, 2)))
        self.past.set_edgecolors(fade)
        self.now.set_offsets(self.xy[i:i + 1])
        self.ax.set_title(f'step {i} of {len(self.walk) - 1}: node {self.walk[i]}')

        self.patches = [self.past, self.now, *self.lit]
        return self.patches

class MatrixAnime(Anime):
    """
    Animate a series of matrices, i.e. frames with either imshow or colormesh
    """

    def __init__(self, X, 
        ax=None, 
        grid=False,
        color='k', 
        linewidth=1, 
        axis='both',
        linestyle='-',
        **im_args):

        super().__init__(frames=len(X), interval=10, ax=ax)

        self.X = X

        self.im_args = im_args

        self.grid = grid
        self.grid_args = {'color':color,
                            'linewidth':linewidth,
                            'axis':axis,
                            'linestyle':linestyle}

    def setup_plot(self):

        self.patches = []

        self.im = self.matshow(self.X[0])
        self.patches = [self.im,]

        return self.patches

    def update(self, i):

        self.im.set_data(self.X[i])

        return self.patches

    def matshow(self,X):

        extent = (0, X.shape[1], X.shape[0], 0)
        im = self.ax.imshow(X, extent=extent, **self.im_args)

        if self.grid:
            self.ax.set_xticks(range(X.shape[1]))
            self.ax.set_yticks(range(X.shape[0]))
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.grid(**self.grid_args)
        else:
            self.ax.set_xticks([])
            self.ax.set_yticks([])

        return im



class ScatterAnime(object):
    """ 
    Makes a scatter plot, and animates it to update along a particular axis 

    i hate matplotlib animation so much 
    """
    def __init__(self, x, y, auto_zoom=False, trail=False, interval=10, 
        fig=None, ax=None, **scat_args):
        ''' 
        x and y are shape (num_data, num_time) 
        '''

        self.x = x
        self.y = y
        self.auto_zoom = auto_zoom
        self.leave_trail = trail
        self.kwargs = scat_args

        self.frames = x.shape[-1]
        self.interval = interval
        
        # Setup the figure and axes...
        if fig is None and ax is None:
            self.fig, self.ax = plt.subplots()
        elif ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif fig is not None:
            self.fig = fig
            self.ax = fig.add_subplot(111)
        # Then setup FuncAnimation.
        self.ani = ani.FuncAnimation(self.fig, self.update, interval=interval, 
            init_func=self.setup_plot, blit=False, frames=x.shape[-1])

    def save(self,save_dir, fps=30):
        self.ani.save(save_dir, writer=ani.FFMpegWriter(fps=fps))

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        self.patches = []

        # self.ax.plot(self.x,self.y)
        self.scat = self.ax.scatter(self.x[...,0], self.y[...,0], **self.kwargs)

        self.patches += [self.scat,]
        # if not self.auto_zoom:
        #     self.ax.set_xlim([1.1*self.x.min(), 1.1*self.x.max()])
        #     self.ax.set_ylim([1.1*self.y.min(), 1.1*self.y.max()])

        if self.leave_trail:
            self.trail, = self.ax.plot(self.x[...,0], self.y[...,0])
            self.patches += [self.trail,]
    
        return self.patches


    def update(self, i):
        """Update the scatter plot."""

        # Set x and y data...
        self.scat.set_offsets([self.x[...,i], self.y[...,i]])

        if self.auto_zoom and i>0:
            # self.ax.set_xlim([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
            # self.ax.set_ylim([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])
            plt.xlim([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
            plt.ylim([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])

        if self.leave_trail and i>0:
            self.trail.set_data([self.x[...,:i].flatten(), self.y[...,:i].flatten()])

        return self.patches

class RotationAnime(object):
    """ 
    Simply rotate a 3d plot 

    i hate matplotlib animation so much 
    """
    def __init__(self, ax, interval=10, rotation_period=100, 
        frames=None, blit=False,
        init_elev=30, init_azim=30, rot_elev=0, rot_azim=1):
        ''' 
        x and y are shape (num_lines, num_data, num_time) 
        '''

        self.rot_speed = rotation_period
        self.init_elev = init_elev # initial elevation
        self.init_azim = init_azim # initial azimuth
        self.rot_e = rot_elev # how much to rotat3 elevation
        self.rot_a = rot_azim # how much to rotate azimuth

        self.ax = ax
        self.fig = ax.get_figure()

        if frames is None:
            frames = rotation_period*2
        
        # Then setup FuncAnimation.
        self.ani = ani.FuncAnimation(self.fig, self.update, interval=interval, 
            init_func=self.setup_plot, blit=blit, frames=frames)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        self.ax.view_init(self.init_elev, self.init_azim)

    def save(self,save_dir, fps=30):
        self.ani.save(save_dir, writer=ani.FFMpegWriter(fps=fps))

    def update(self, i):
        """Update the scatter plot."""
        self.ax.view_init(self.init_elev + self.rot_e*(i/self.rot_speed)*360,
                            self.init_azim + self.rot_a*(i/self.rot_speed)*360)

class LineAnime(object):
    """ 
    Makes a (multiple) line plot(s), and animates 

    i hate matplotlib animation so much 
    """
    def __init__(self, x, y, auto_zoom=False, interval=10, colors=None,
        fig=None, ax=None, **ln_args):
        ''' 
        x and y are shape (num_lines, num_data, num_time) 
        '''

        self.x = x
        self.y = y
        self.kwargs = ln_args
        self.auto_zoom = auto_zoom

        if fig is None and ax is None:
            self.fig, self.ax = plt.subplots()
        elif ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif fig is not None:
            self.fig = fig
            self.ax = fig.add_subplot(111)

        if colors is None:
            self.colors = [(0.5,0.5,0.5),]*len(x)
        else:
            self.colors = colors

        self.frames = x.shape[-1]
        self.interval = interval
        
        # Then setup FuncAnimation.
        # self.ani = anime.FuncAnimation(self.fig, self.update, interval=interval, 
        #     init_func=self.setup_plot, blit=False, frames=x.shape[-1])

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        self.patches = []

        # self.ax.plot(self.x,self.y)
        self.lines = []
        for i in range(len(self.x)):
            ln, = self.ax.plot(self.x[i,...,:1].T, self.y[i,...,:1].T,
                color=self.colors[i], **self.kwargs)
            self.lines.append(ln)

        self.patches += self.lines
        if self.auto_zoom:
            self.ax.set_xlim([1.1*self.x.min(), 1.1*self.x.max()])
            self.ax.set_ylim([1.1*self.y.min(), 1.1*self.y.max()])
    
        return self.patches

    def save(self,save_dir, fps=30):
        self.ani.save(save_dir, writer=ani.FFMpegWriter(fps=fps))

    def update(self, i):
        """Update the scatter plot."""

        # Set x and y data...
        for j in range(len(self.lines)):
            self.lines[j].set_data([self.x[j,...,:i].T, self.y[j,...,:i].T])

        if self.auto_zoom and i>0:
            # self.ax.set_xlim([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
            # self.ax.set_ylim([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])
            plt.xlim([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
            plt.ylim([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])

        return self.patches

class LineAnime3D(object):
    """ 
    Makes a (multiple) line plot(s), and animates 

    i hate matplotlib animation so much 
    """
    def __init__(self, x, y, z, auto_zoom=False, interval=10, 
        rotation_period=1000, colors=None, dot=False, scat_colors=None,
        focus_on=None, focus_period=100,
        frames=None, blit=False, view_period=0,
        init_elev=30, init_azim=30, rot_elev=0, rot_azim=1,
        fig=None, ax=None,
        **kwargs):
        ''' 
        x, y and z are shape (num_lines, num_data, num_time) 
        ''' 

        self.x = x
        self.y = y
        self.z = z
        self.kwargs = kwargs
        self.auto_zoom = auto_zoom
        self.view_period = view_period
        
        self.rot_speed = rotation_period
        self.init_elev = init_elev # initial elevation
        self.init_azim = init_azim # initial azimuth
        self.rot_e = rot_elev # how much to rotat3 elevation
        self.rot_a = rot_azim # how much to rotate azimuth

        if scat_colors is None:
            self.scat_colors = colors
        else:
            self.scat_colors = scat_colors

        if colors is None:
            self.colors = [(0.5,0.5,0.5),]*len(x)
        else:
            self.colors = colors

        self.do_scat = dot
        
        if focus_on is None:
            self.focus = False
        else:
            self.focus = True
            self.focus_on = focus_on
            self.focus_period = focus_period

        if frames is None:
            frames = x.shape[-1] + 2*view_period

        # Setup the figure and axes...
        if fig is None and ax is None:
            self.fig = plt.figure()
            self.ax = plt.subplot(111, projection='3d')
        elif ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif fig is not None:
            self.fig = fig
            self.ax = fig.add_subplot(111, projection='3d')
            
        # Then setup FuncAnimation.
        self.ani = ani.FuncAnimation(self.fig, self.update, interval=interval, 
            init_func=self.setup_plot, blit=blit, frames=frames)

    def save(self,save_dir, fps=30):
        self.ani.save(save_dir, writer=ani.FFMpegWriter(fps=fps))

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        self.patches = []

        if self.focus:
            idx = self.focus_on
        else:
            idx = np.arange(len(self.x))

        # self.ax.plot(self.x,self.y)
        self.lines = []
        for i in idx:
            ln, = self.ax.plot3D(self.x[i,...,:1].T, self.y[i,...,:1].T, self.z[i,...,:1].T, 
                color=self.colors[i], **self.kwargs)
            self.lines.append(ln)

        self.patches += self.lines

        if self.do_scat:
            self.scat = self.ax.scatter(self.x[idx,...,0], self.y[idx,...,0], self.z[idx,...,0], 
                edgecolors=self.scat_colors, facecolors=self.scat_colors)
            self.patches += [self.scat,]

        if not self.auto_zoom:
            self.ax.set_xlim3d([1.1*self.x[idx].min(), 1.1*self.x[idx].max()])
            self.ax.set_ylim3d([1.1*self.y[idx].min(), 1.1*self.y[idx].max()])
            self.ax.set_zlim3d([1.1*self.z[idx].min(), 1.1*self.z[idx].max()])
        else:
            self.ax.set_xlim3d([1.1*self.x[idx,...,0].min(), 1.1*self.x[idx,...,0].max()])
            self.ax.set_ylim3d([1.1*self.y[idx,...,0].min(), 1.1*self.y[idx,...,0].max()])
            self.ax.set_zlim3d([1.1*self.z[idx,...,0].min(), 1.1*self.z[idx,...,0].max()])


        self.ax.view_init(self.init_elev, self.init_azim)
    
        return self.patches

    def update(self, i):
        """Update the scatter plot."""

        self.ax.view_init(self.init_elev + self.rot_e*(i/self.rot_speed)*360,
            self.init_azim + self.rot_a*(i/self.rot_speed)*360)

        if i>self.view_period:
            i -= self.view_period
            # Set x and y data...
            for j in range(len(self.lines)):
                # self.lines[j].set_data_3d(self.x[j,...,:i].T, self.y[j,...,:i].T, self.z[j,...,:i].T)
                self.lines[j].set_data(self.x[j,...,:i], self.y[j,...,:i])
                self.lines[j].set_3d_properties(self.z[j,...,:i])

            if self.do_scat:
                # self.scat.set_offsets([self.x[...,i], self.y[...,i]])
                # self.scat.set_3d_properties(self.z[...,i], 'z')
                self.scat._offsets3d = (self.x[...,i], self.y[...,i],self.z[...,i])

            if self.auto_zoom and i>0:
                # self.ax.set_xlim([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
                # self.ax.set_ylim([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])
                self.ax.set_xlim3d([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
                self.ax.set_ylim3d([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])
                self.ax.set_zlim3d([1.1*np.min(self.z[...,:i+1]), 1.1*self.z[...,:i+1].max()])

        # self.ax.view_init(30,(i/self.rot_speed)*360)

        return self.patches

class ScatterAnime3D(object):
    """ 
    Makes a (multiple) line plot(s), and animates 

    i hate matplotlib animation so much 
    """
    def __init__(self, x, y, z, auto_zoom=False, interval=10, blit=False,
        rotation_period=1000, after_period=0, view_period=0,
        init_elev=30, init_azim=30, rot_elev=0, rot_azim=1,
        fig=None, ax=None, equalize=False,
        **kwargs):
        ''' 
        x and y are shape (num_lines, num_data, num_time) 
        '''

        self.x = x
        self.y = y
        self.z = z
        self.kwargs = kwargs
        self.auto_zoom = auto_zoom
        self.view_period = view_period
        self.after_period = after_period
        self.equalize = equalize
        
        self.rot_speed = rotation_period
        self.init_elev = init_elev # initial elevation
        self.init_azim = init_azim # initial azimuth
        self.rot_e = rot_elev # how much to rotat3 elevation
        self.rot_a = rot_azim # how much to rotate azimuth

        self.tot_frames = x.shape[-1] + view_period + after_period

        # Setup the figure and axes...
        if fig is None and ax is None:
            self.fig = plt.figure()
            self.ax = plt.subplot(111, projection='3d')
        elif ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        elif fig is not None:
            self.fig = fig
            self.ax = fig.add_subplot(111, projection='3d')
            
        # Then setup FuncAnimation.
        self.ani = ani.FuncAnimation(self.fig, self.update, interval=interval, 
            init_func=self.setup_plot, blit=blit, frames=self.tot_frames)

    def save(self, save_dir, fps=30):
        self.ani.save(save_dir, writer=ani.FFMpegWriter(fps=fps))

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        self.patches = []

        idx = np.arange(len(self.x))

        # self.ax.plot(self.x,self.y)
        self.scat = self.ax.scatter(self.x[...,0].T, self.y[...,0].T, self.z[...,0].T, 
             **self.kwargs)

        self.patches += [self.scat,]

        if not self.auto_zoom:
            self.ax.set_xlim3d([1.1*self.x[idx].min(), 1.1*self.x[idx].max()])
            self.ax.set_ylim3d([1.1*self.y[idx].min(), 1.1*self.y[idx].max()])
            self.ax.set_zlim3d([1.1*self.z[idx].min(), 1.1*self.z[idx].max()])
        else:
            self.ax.set_xlim3d([1.1*self.x[idx,...,0].min(), 1.1*self.x[idx,...,0].max()])
            self.ax.set_ylim3d([1.1*self.y[idx,...,0].min(), 1.1*self.y[idx,...,0].max()])
            self.ax.set_zlim3d([1.1*self.z[idx,...,0].min(), 1.1*self.z[idx,...,0].max()])

        if self.equalize:
            set_axes_equal(self.ax)

        self.ax.view_init(self.init_elev, self.init_azim)
    
        return self.patches

    def update(self, i):
        """Update the scatter plot."""

        self.ax.view_init(self.init_elev + self.rot_e*(i/self.rot_speed)*360,
            self.init_azim + self.rot_a*(i/self.rot_speed)*360)

        if i>self.view_period and i<(self.tot_frames-self.after_period):
            i -= self.view_period
            # Set x and y data...
    
                # self.lines[j].set_data_3d(self.x[j,...,:i].T, self.y[j,...,:i].T, self.z[j,...,:i].T)
            self.scat._offsets3d = (self.x[...,i], self.y[...,i],self.z[...,i])
            # self.scat.set_color(self.color[i])

            if self.auto_zoom and i>0:
                # self.ax.set_xlim([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
                # self.ax.set_ylim([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])
                self.ax.set_xlim3d([1.1*self.x[...,:i+1].min(), 1.1*self.x[...,:i+1].max()])
                self.ax.set_ylim3d([1.1*self.y[...,:i+1].min(), 1.1*self.y[...,:i+1].max()])
                self.ax.set_zlim3d([1.1*np.min(self.z[...,:i+1]), 1.1*self.z[...,:i+1].max()])

        # self.ax.view_init(30,(i/self.rot_speed)*360)

        return self.patches
    