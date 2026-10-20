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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon
from matplotlib import animation as anime
from cycler import cycler
import scipy
import scipy.linalg as la
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linprog as lp
from itertools import permutations, combinations
import itertools as itt

#%%
class PCA3D(object):
    def __init__(self, X, special_vec=None, ax=None, **scat_args):
        ''' assume X is shape (dim_data, num_data) '''

        if special_vec is None:
            X_ = X
        else:
            X_ = X - (special_vec[:,None]@special_vec[None,:]/la.norm(special_vec))@X
            y_ = (special_vec[None,:]/la.norm(special_vec))@X

        U, S, _ = la.svd(X-X.mean(1)[:,None], full_matrices=False)
        pcs = X.T@U[:,:3]

        if special_vec is None:
            self.loadings = U[:,:3]
        else:
            self.loadings = np.concatenate([special_vec[:,None], U[:,:2]])

        if ax is None:
            self.ax = plt.subplot(111, projection='3d')
        else:
            self.ax = ax
        if special_vec is None:
            self.scat = self.ax.scatter(pcs[:,0],pcs[:,1],pcs[:,2], **scat_args)
        else:
            self.scat = self.ax.scatter(y_,pcs[:,0],pcs[:,1], **scat_args)

    def overlay(self, new_X, **scat_args):

        new_pcs = new_X.T@self.loadings
        self.ax.scatter(new_pcs[:,0],new_pcs[:,1],new_pcs[:,2], **scat_args)


def pca3d(X, ax=None, special_vec=None, **scat_args):
    '''
    Assume that "X" is N_feat x N_samp, "coloring" is a vector of dim N_samp
    '''

    if special_vec is None:
        X_ = X
    else:
        X_ = X - (special_vec[:,None]@special_vec[None,:]/la.norm(special_vec))@X
        y_ = (special_vec[None,:]/la.norm(special_vec))@X

    U, S, _ = la.svd(X-X.mean(1)[:,None], full_matrices=False)
    pcs = X.T@U[:,:3]

    if ax is None:
        ax = plt.subplot(111, projection='3d')
    if special_vec is None:
        scat = ax.scatter3D(pcs[:,0],pcs[:,1],pcs[:,2], **scat_args)
    else:
        scat = ax.scatter3D(y_,pcs[:,0],pcs[:,1], **scat_args)
    # ax.set_xlabel('pc1')
    # ax.set_ylabel('pc2')
    # ax.set_zlabel('pc3')

    set_axes_equal(ax)

    return scat

def pca(X, special_vec=None, **scat_args):
    '''
    Assume that "X" is N_feat x N_samp, "coloring" is a vector of dim N_samp
    '''

    if special_vec is None:
        X_ = X
    else:
        X_ = X - (special_vec[:,None]@special_vec[None,:]/la.norm(special_vec))@X
        y_ = (special_vec[None,:]/la.norm(special_vec))@X

    U, S, _ = la.svd(X-X.mean(1)[:,None], full_matrices=False)
    pcs = X.T@U[:,:2]

    if special_vec is None:
        scat = plt.scatter(pcs[:,0],pcs[:,1], **scat_args)
    else:
        scat = plt.scatter(y_,pcs[:,0], **scat_args)

    # set_axes_equal(ax)

    return scat

#################################################
############# Interactive slice plots ###########
#################################################

class SliceViewer(object):
    def __init__(self, ax=None):
        if ax is not None:
            self.ax = ax
        else:
            self.ax = plt.axes()

        if hasattr(self.ax,'claimed'):
            if self.ax.claimed:
                self.is_vassal = True
                self.leige = self.ax.leige
            else:
                self.ax.claimed = True
                self.is_vassal = False
                self.ax.leige = self
        else:
            self.ax.claimed = True
            self.is_vassal = False
            self.ax.leige = self

    def on_scroll(self,event):
        if self.is_vassal:
            self.ind = self.leige.ind
        else:
            if event.button == 'up':
                # self.ind = (self.ind + 1) % self.slices
                self.ind = np.min([self.ind + 1,self.slices-1])
            else:
                # self.ind = (self.ind - 1) % self.slices
                self.ind = np.max([self.ind - 1, 0])
        self.update() 

class ImageSlices(SliceViewer):
    def __init__(self, X, ax=None, **im_args):

        super(ImageSlices,self).__init__(ax)

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = self.ax.imshow(self.X[:, :, self.ind], 
            norm=mpc.Normalize(self.X.min(), self.X.max()), **im_args)
        self.update()

        self.ax.get_figure().canvas.mpl_connect('scroll_event', self.on_scroll)

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

class LineSlices(SliceViewer):
    def __init__(self, X, Y, ax=None, scroll_mag=0.02, slice_width=None, num_slice=None, 
        slice_idx=None, **ln_args):
        '''
        Plots X, only showing the parts within a certain range of Y -- scroll to change Y

        X is shape (num_line, num_time, 2)
        Y is shape (num_line, num_time)

        if slice_idx is supplied, it should be the center of each bin
        '''
        super(LineSlices,self).__init__(ax)

        self.X = X
        self.Y = Y 

        if slice_idx is None:
            if num_slice is None:
                self.scroll_mag = scroll_mag
            else:
                self.scroll_mag = (Y.max()-Y.min())/(num_slice-1)

            self.slice_idx = np.arange(Y.min(), Y.max()+self.scroll_mag, self.scroll_mag)
        else:
            self.slice_idx = slice_idx

        if slice_width is None:
            self.slice_width = np.diff(self.slice_idx)/2
            self.slice_width = np.append(self.slice_width, self.slice_width[0])
        else:
            self.slice_width = np.ones(len(slice_idx))*slice_width

        self.num_slices = len(self.slice_idx)

        self.ind = self.num_slices//2
        # self.idx = (self.Y < self.slice_idx[self.ind]+self.slice_width) \
        # & (self.Y >= self.slice_idx[self.ind]-self.slice_width)
        # self.mask_x1 = np.where(self.idx, self.X[...,0], np.nan)
        # self.mask_x2 = np.where(self.idx, self.X[...,1], np.nan)

        nan_pad = np.ones((self.num_slices, self.Y.shape[0], 1))*np.nan
        # print(nan_pad.shape)

        self.all_idx = [(self.Y < s+ds) & (self.Y > s-ds) for s, ds in zip(self.slice_idx,self.slice_width)]
        self.mask_X1 = np.where(self.all_idx, self.X[...,0], np.nan)
        self.mask_X1 = np.concatenate([nan_pad, self.mask_X1, nan_pad], axis=-1)
        # print(self.mask_X1.shape)
        self.mask_X2 = np.where(self.all_idx, self.X[...,1], np.nan)
        self.mask_X2 = np.concatenate([nan_pad, self.mask_X2, nan_pad], axis=-1)

        self.ln = self.ax.plot(self.mask_X1[self.ind].T, self.mask_X2[self.ind].T, **ln_args)
        # print(len(self.ln))
        self.update()

        self.ax.get_figure().canvas.mpl_connect('scroll_event', self.on_scroll)

    def update(self):
        # self.idx = (self.Y < self.slice_idx[self.ind]+self.slice_width) \
        # & (self.Y >= self.slice_idx[self.ind]-self.slice_width)
        # self.mask_x1 = np.where(self.idx,self.X[...,0], np.nan)
        # self.mask_x2 = np.where(self.idx, self.X[...,1], np.nan)
        # print(self.idx)
        # print(self.mask)

        for ln in self.ln:
            ln.set_data([self.mask_X1[self.ind], self.mask_X2[self.ind]])
        self.ax.set_ylabel('slice for y= %.1f' % self.slice_idx[self.ind])
        plt.draw()

class ScatterSlices(SliceViewer):
    def __init__(self, X, Y, ax=None, slice_width=0.1, scroll_mag=0.02,num_slice=None,slice_idx=None, 
        color=None, **scat_args):
        '''
        Plots X, only showing the parts within a certain range of Y -- scroll to change Y
        '''
        super(ScatterSlices,self).__init__(ax)

        if color is None:
            self.col = [(0.5,0.5,0.5),]*len(Y)
        else:
            self.col = color

        self.X = X
        self.Y = Y 
        self.slice_width = slice_width
        if slice_idx is None:
            if num_slice is None:
                self.scroll_mag = scroll_mag
            else:
                self.scroll_mag = (Y.max()-Y.min())/(num_slice-1)

            self.slice_idx = np.arange(Y.min(), Y.max()+self.scroll_mag, self.scroll_mag)
        else:
            self.slice_idx = slice_idx
        self.slices = len(self.slice_idx)

        self.ind = self.slices//2
        self.idx = (self.Y < self.slice_idx[self.ind]+self.slice_width) \
        & (self.Y >= self.slice_idx[self.ind]-self.slice_width)

        self.scat = self.ax.scatter(self.X[self.idx, 0], self.X[self.idx, 1], 
            facecolors=np.array(self.col)[self.idx], **scat_args)
        self.update()

        self.ax.get_figure().canvas.mpl_connect('scroll_event', self.on_scroll)

    def update(self):

        self.idx = (self.Y < self.slice_idx[self.ind]+self.slice_width) \
        & (self.Y >= self.slice_idx[self.ind]-self.slice_width)

        self.scat.set_offsets(self.X[self.idx, :])
        self.scat.set_color(np.array(self.col)[self.idx])
        self.ax.set_ylabel('slice for y= %.3f' % self.slice_idx[self.ind])
        plt.draw()

class QuiverSlices(SliceViewer):
    def __init__(self, X, V, ax=None, **quiv_args):
        """
        X is shape (num_arrow, 2)
        V is shape (num_slice, num_arrow, 2)
        """

        super(QuiverSlices,self).__init__(ax)

        if ax is not None:
            self.ax = ax
        else:
            self.ax = plt.axes()

        self.X = X
        self.V = V
        self.slices = V.shape[0]
        self.ind = self.slices//2

        self.quiv = self.ax.quiver(self.X[:,0], self.X[:,1], 
            self.V[self.ind,:,0],  self.V[self.ind,:,1], **quiv_args)
        self.update()

        self.ax.get_figure().canvas.mpl_connect('scroll_event', self.on_scroll)

    def update(self):
        self.quiv.set_UVC(self.V[self.ind, :, 0], self.V[self.ind, :, 1])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.quiv.axes.figure.canvas.draw()


##############################################
########### Grids of subplots ################
##############################################

def hierarchical_labels(row_labels, col_labels, row_names=None, col_names=None, hmarg=0.1,wmarg=0.1, **text_args):
    """
    
    """

    if row_names is None:
        label_rows = False
        row_names = np.arange(len(row_labels))
    else:
        label_rows = True
    if col_names is None:
        label_cols = False
        col_names = np.arange(len(col_labels))
    else:
        label_cols = True

    n_row = len(list(itt.product(*row_labels)))
    n_col = len(list(itt.product(*col_labels)))

    fig, axes = plt.subplots(n_row,n_col)
    axes = np.reshape(axes, (n_row, n_col))

    w_rat = [0.025,]*len(row_names) + [(1-0.025*len(row_names))/n_col,]*n_col
    h_rat = [0.025,]*len(col_names) + [(1-0.025*len(col_names))/n_row,]*n_row

    gs = gsp.GridSpec(n_row+len(col_names), n_col+len(row_names), 
                      width_ratios=w_rat, height_ratios=h_rat)
    gs.update(left=0.1, right=0.9, bottom=0.15, wspace=wmarg, hspace=hmarg)
    fig.subplots_adjust(left=.1)

    for i in range(n_row):
        for j in range(n_col):
            # if n_row == 1:
            #     axes[j].set_subplotspec(gs[i+len(col_names),j+len(row_names)])
            # elif n_col == 1:
            #     axes[i].set_subplotspec(gs[i+len(col_names),j+len(row_names)])
            # else:
            axes[i,j].set_subplotspec(gs[i+len(col_names),j+len(row_names)])

    row_lab_axs = []
    num_lab = n_row
    for i,lab in enumerate(row_names):
        num_lab = num_lab//len(row_labels[i])
        # gs_idx = np.arange(n_row)//num_lab
        
        tmp_axs = []
        for j in range(n_row//num_lab):
            idx0 = num_lab*j + len(col_names)
            idx1 = num_lab*(j+1) + len(col_names)
            tmp_axs.append(fig.add_subplot(gs[idx0:idx1,i]))
        
        lab_idx = np.mod(np.arange(n_row//num_lab),len(row_labels[i]))
        for j,ax in enumerate(tmp_axs):
            if label_rows:
                ax.set_ylabel('%s = %s'%(lab,row_labels[i][lab_idx[j]]), 
                              **text_args)
            else:
                ax.set_ylabel('%s'%(row_labels[i][lab_idx[j]]), 
                              **text_args)
        
        row_lab_axs += tmp_axs

    col_lab_axs = []
    num_lab = n_col
    for i,lab in enumerate(col_names):
        num_lab = num_lab//len(col_labels[i])
        
        tmp_axs = []
        for j in range(n_col//num_lab):
            idx0 = num_lab*j + len(row_names)
            idx1 = num_lab*(j+1) + len(row_names)
            tmp_axs.append(fig.add_subplot(gs[i,idx0:idx1]))
        
        lab_idx = np.mod(np.arange(n_col//num_lab),len(col_labels[i]))
        for j,ax in enumerate(tmp_axs):
            if label_cols:
                ax.set_title('%s = %s'%(lab,col_labels[i][lab_idx[j]]),
                              **text_args)
            else:
                ax.set_title('%s'%(col_labels[i][lab_idx[j]]),
                              **text_args)
        
        col_lab_axs += tmp_axs


    for ax in row_lab_axs:
        ax.tick_params(size=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_facecolor("none")
        for pos in ["right", "top", "bottom"]:
            ax.spines[pos].set_visible(False)
        ax.spines["left"].set_linewidth(3)
        ax.spines["left"].set_color("crimson")

    for ax in col_lab_axs:
        ax.tick_params(size=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_facecolor("none")
        for pos in ["right", "left", "bottom"]:
            ax.spines[pos].set_visible(False)
        ax.spines["top"].set_linewidth(3)
        ax.spines["top"].set_color("crimson")

    return axes

def dichotomy_plot(PS, CCGP, SD, PS_err=None, CCGP_err=None, SD_err=None, 
    cmap='copper', var_cmap='tab10', include_legend=True, include_cbar=True, 
    input_dics=None, output_dics=None, other_dics=None, s=31, c=[(0.5,0.5,0.5)],
     **scatter_args):
    '''
    Plot values for each abstraction metric as scatter plots with x-axis noise.

    Optionally highlight the first "num_special" dichotomies.
    '''
    ndic = len(PS)

    # be very picky about offsets
    offset = np.random.choice(np.linspace(-0.15,0.15,10), 3*ndic)
    if (output_dics or input_dics) is not None:
        spec_dics = []
        if output_dics is not None:
            spec_dics += output_dics
        if input_dics is not None:
            spec_dics += input_dics
        if other_dics is not None:
            spec_dics += other_dics

        special_offsets = np.linspace(-0.15,0.15,len(spec_dics))
        for i,n in enumerate(spec_dics):
            offset[n+np.arange(3)*ndic] = special_offsets[i]

    xfoo = np.repeat([0,1,2],ndic).astype(int) + offset # np.random.randn(ndic*3)*0.1
    yfoo = np.concatenate((PS, CCGP, SD))
    if PS_err is None:
        yfoo_err = np.zeros(xfoo.shape)
    else:
        yfoo_err = np.concatenate((PS_err, CCGP_err, SD_err))

    plt.errorbar(xfoo, yfoo, yerr=yfoo_err, linestyle='None', c=(0.5,0.5,0.5), zorder=0)
    if c is not None:
        scat = plt.scatter(xfoo, yfoo, s=s, c=np.tile(c,3), zorder=10, cmap=cmap, **scatter_args)
    else:
        scat = plt.scatter(xfoo, yfoo, s=s, c=[(0.5,0.5,0.5)], zorder=10, **scatter_args)
    # plt.errorbar(xfoo, yfoo, yerr=yfoo_err, linestyle='None', linecolor=cfoo)
    plt.xticks([0,1,2], labels=['PS', 'CCGP', 'Shattering'])
    plt.ylabel('PS or Cross-validated performance')
    if include_cbar:
        plt.colorbar(scat)

    # highlight special dichotomies
    anns = []
    leg_text = []
    if input_dics is not None:
        for i, n in enumerate(input_dics):
            anns.append(plt.scatter(xfoo[[n,n+ndic,n+2*ndic]], yfoo[[n,n+ndic,n+2*ndic]], marker='d',
                       s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))

        if include_legend:
            # plt.legend(anns, ['input %d'%(i+1) for i in range(len(anns))])
            leg_text += ['input %d'%(i+1) for i in range(len(anns))]
    if output_dics is not None:
        for i,n in enumerate(output_dics):
            anns.append(plt.scatter(xfoo[[n,n+ndic,n+2*ndic]], yfoo[[n,n+ndic,n+2*ndic]], marker='s',
                       s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))

        if include_legend:
            # plt.legend(anns, ['output %d'%(i+1) for i in range(len(anns))])
            leg_text += ['output %d'%(i+1) for i in range(len(output_dics))]
    if other_dics is not None:
        for i,n in enumerate(other_dics):
            anns.append(plt.scatter(xfoo[[n,n+ndic,n+2*ndic]], yfoo[[n,n+ndic,n+2*ndic]], marker='*',
                       s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))
    
        if include_legend:
            # plt.legend(anns, ['output %d'%(i+1) for i in range(len(anns))])
            leg_text += ['special %d'%(i+1) for i in range(len(other_dics))]

    if len(anns)>0 and include_legend:
        plt.legend(anns, leg_text)


################################################
################# Polytopes ####################
################################################

def plotcube(E, S, V=None, dim=2, hidden=None, show_hid=False,
    cmap='Dark2', linewidth=2, headwidth=5, **scat_args):
    """
    Plot a partial cube graph with edges E and labels S
    Plot the embedding with component vectors V. 
    V should be shape (k,2) or (k,3)
    """

    if V is not None:
        dim = V.shape[1] 
    else:
        K = (S-S.mean(0))@(S-S.mean(0)).T
        l,V = la.eigh(K)
        V = V[:,:dim]

    if hidden is None:
        hidden = (np.zeros(len(S))>0)

    F = S@V

    locs = F[E[0]].T
    e_lab = np.mod(S[E[0]]+S[E[1]], 2).argmax(1)
    dirs = V[e_lab].T

    if dim == 2:
        plt.scatter(F[~hidden,0], F[~hidden,1], 
                c='k', **scat_args, zorder=10)
        plt.quiver(locs[0], locs[1], dirs[0], dirs[1], e_lab, 
                angles='xy', scale_units='xy', scale=1, cmap=cmap,
                headwidth=headwidth)
    elif dim == 3:
        ax = plt.figure().add_subplot(projection='3d')

        ## deal with matplotlib's super jank 3d support
        cols = getattr(cm, cmap)(e_lab)
        cols = np.vstack([cols, np.repeat(cols,2, axis=0)])

        scatter3d(F[~hidden], ax=ax, **scat_args, zorder=10)
        ax.quiver(locs[0], locs[1], locs[2], dirs[0], dirs[1], dirs[2],
            colors=cols, linewidth=linewidth, zorder=0)

        if show_hid:
            scatter3d(F[hidden], ax=ax, 
                c='w', edgecolors='k', s=100, zorder=10)


def polygon(vertices, ax=None, facecolor=None, alpha=0.5, **plot_args):

    if vertices.shape[1] == 3:
        is_3d = True
    elif vertices.shape[1] == 2:
        is_3d = False

    n = len(vertices)

    order = [0]
    remaining = list(range(1,n))
    dvert = la.norm(vertices[:,None,:] - vertices[None,:,:], axis=-1)
    while len(order)<n:
        i = remaining[np.argmin(dvert[order[-1],remaining])]
        order.append(i)
        remaining.remove(i)

    vertices = vertices[order]

    verts = vertices.tolist()
    verts.append(verts[0]) # repeat the first point to create a 'closed loop'

    if is_3d:
        fill = Poly3DCollection([vertices], color=facecolor, alpha=alpha,
                                linewidth=0)
        xs, ys, zs = zip(*verts)

        if ax is None:
            ax = plt.subplot(111, projection='3d')

        ax.add_collection(fill)
        ax.plot(xs, ys, zs, **plot_args)

    else:
        poly = Polygon(vertices, color=facecolor, alpha=alpha)
        xs, ys = zip(*vertices) #create lists of x and y values

        if ax is None:
            ax = plt.axes()

        ax.plot(xs, ys, **plot_args)
        ax.add_patch(poly)

    return ax

def plane(a, b=0, length=1, width=1, ax=None, alpha=0.5, color=(0.5,0.5,0.5)):

    l,v = la.eigh(a[:,None]*a[None,:])
    null = v[:,:2]

    x, y = np.meshgrid([-length,length], [-width,width])

    coords = np.stack([x, y], axis=-1)@v[:,:2].T + a*b

    if ax is None:
        ax = plt.subplot(111, projection='3d')

    ax.plot_surface(coords[...,0], coords[...,1], coords[...,2],
                    linewidth=0, color=color, alpha=alpha)

def polyhedron(vertices=None, halfspace=None, 
    color=(0.5,0.5,0.5), edgecolor='k', alpha=0.5, 
    ax=None, plot_verts=True, **vert_args):
    """
    Plot a convex polytope in 3d

    code to do this is from stackexchange user JohanC
    """

    has_vert = (vertices is not None)
    has_half = (halfspace is not None)

    if not has_vert or has_half:
        raise ValueError('You have to supply something, man')

    elif has_vert and has_half:
        raise Warning('You gave two arguments, I will use the vertices')


    if ax is None:
        ax = plt.subplot(111, projection='3d')

    # if has_half:
    #     # to avoid using pypoman, that most people probably won't have,
    #     # I will use the much inferior scipy class, which requires
    #     # manually computing an interior point ...
    #     # (why can't they do that in the function??)

    #     hs = HalfspaceIntersection(halfspace, int_point)

    if has_vert:

        hull = ConvexHull(vertices)

        f = Faces([vertices[s] for s in hull.simplices])
        g = f.simplify()

        tri = Poly3DCollection(g, facecolor=color, edgecolor=edgecolor, alpha=alpha)

        ax.add_collection(tri)

        if plot_verts:
            ax.scatter(vertices[:,0],vertices[:,1], vertices[:,2], **vert_args)

    return ax

class Faces():
    """
    By stackexchange user ImportanceOfBeingErnest
    """
    def __init__(self,tri, sig_dig=12, method="convexhull"):
        self.method=method
        self.tri = np.around(np.array(tri), sig_dig)
        self.grpinx = list(range(len(tri)))
        norms = np.around([self.norm(s) for s in self.tri], sig_dig)
        _, self.inv = np.unique(norms,return_inverse=True, axis=0)

    def norm(self,sq):
        cr = np.cross(sq[2]-sq[0],sq[1]-sq[0])
        return np.abs(cr/np.linalg.norm(cr))

    def isneighbor(self, tr1,tr2):
        a = np.concatenate((tr1,tr2), axis=0)
        return len(a) == len(np.unique(a, axis=0))+2

    def order(self, v):
        if len(v) <= 3:
            return v
        v = np.unique(v, axis=0)
        n = self.norm(v[:3])
        y = np.cross(n,v[1]-v[0])
        y = y/np.linalg.norm(y)
        c = np.dot(v, np.c_[v[1]-v[0],y])
        if self.method == "convexhull":
            h = ConvexHull(c)
            return v[h.vertices]
        else:
            mean = np.mean(c,axis=0)
            d = c-mean
            s = np.arctan2(d[:,0], d[:,1])
            return v[np.argsort(s)]

    def simplify(self):
        for i, tri1 in enumerate(self.tri):
            for j,tri2 in enumerate(self.tri):
                if j > i: 
                    if self.isneighbor(tri1,tri2) and \
                       self.inv[i]==self.inv[j]:
                        self.grpinx[j] = self.grpinx[i]
        groups = []
        for i in np.unique(self.grpinx):
            u = self.tri[self.grpinx == i]
            u = np.concatenate([d for d in u])
            u = self.order(u)
            groups.append(u)
        return groups

##############################################
########### General utility ##################
##############################################

def raster(X, autocolor=True, **args):
    """
    make a raster plot for an array of timestamps, (n_channel, n_events)
    """

    n, k = X.shape

    y = np.repeat(np.expand_dims(np.arange(n), -1), k, axis=-1)
    if autocolor:
        c = np.repeat(np.expand_dims(np.arange(k), -1), n, axis=-1).T

    plt.scatter(X.flatten(), y.flatten(), c=c.flatten(), **args)

def vecs(x,y, **kwargs):
    """
    My version of quiver where vectors are always at the origin
    """

    n = len(x)

    plt.quiver(np.zeros(n), np.zeros(n), x, y, **kwargs)

def square_axis(ax=None):
    if ax is None:
        ax = plt.gca()
    newlims = [np.min([ax.get_ylim(), ax.get_xlim()]), np.max([ax.get_ylim(), ax.get_xlim()])]

    ax.axis('equal')
    ax.axis('square')
    ax.set_xlim(newlims)
    ax.set_ylim(newlims)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    From karlo on stack exchange

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def diverging_clim(ax):
    for im in ax.get_images():
        cmax = np.max(np.abs(im.get_clim()))
        im.set_clim(-cmax,cmax)


def scatter3d(X, ax=None, labels=None, **scat_args):
    if ax is None:
        ax = plt.subplot(111, projection='3d')
    scat = ax.scatter(X[...,0],X[...,1],X[...,2], **scat_args)

    if labels is not None:
        for x, lab in zip(X, labels):
            ax.text(x[0], x[1], x[2], lab)

    set_axes_equal(ax)

    return ax

def plot3d(X, ax=None, **line_args):
    """
    X is shape (N, M, 3), with N lines of length M. If there's only one, X can 
    just be shape (N, 3)
    """

    if ax is None:
        ax = plt.subplot(111, projection='3d')
    if np.ndim(X) > 2:
        for i in range(X.shape[-2]):
            lines = ax.plot(X[...,i,0],X[...,i,1],X[...,i,2], **line_args)
    else:
        lines = ax.plot(X[:,0],X[:,1],X[:,2], **line_args)

    set_axes_equal(ax)

    return ax

def fillrange(x, Y, center='mean', fill='std', **kwargs):
    """
    x is shape (n_value)
    Y is shape (n_sample, n_value)
    """
    if center == 'mean':
        Ey = Y.mean(0)
    elif center == 'median':
        Ey = np.median(Y,axis=0)
    if fill == 'std':
        yup = Y.std(0)
        ydown = -yup
    elif fill == 'err':
        yup = Y.std(0)/np.sqrt(len(Y))
        ydown = -yup
    elif fill == 'range':
        yup = Y.max(0) - Ey
        ydown = Y.min(0) - Ey

    plt.fill_between(x, Ey+ydown, Ey+yup, **kwargs)

def medianstd(x, Y, err=False, **kwargs):
    """
    x is shape (n_value)
    Y is shape (n_sample, n_value)
    """
    Ey = np.median(Y,axis=0)
    sigy = Y.std(0)
    if err:
        sigy = sigx/np.sqrt(len(Y))

    plt.fill_between(x, Ey-sigy, Ey+sigy, **kwargs)

def pairwise_lines(X, col=(0.5,0.5,0.5), ax=None):

    if ax is None:
        ax=plt.gca()
    num_dat = X.shape[0]
    for ix in combinations(range(num_dat),2):
        ax.plot(X[ix,0],X[ix,1],X[ix,2],color=col)

def color_cycle(cmap, num_col):
    # return getattr(cm,cmap)(np.linspace(0,1,num_col))
    these_cols = ['r','b','g','y','c','m']
    return [mpc.to_rgb(these_cols[i]) for i in range(num_col)]

def plot_matrix(X, **args):

    plt.pcolor(np.flipud(X), **args)
    square_axis()
    plt.axis('off')


def matshow(X, ax=None, 
            show_grid=True,
            color='k', 
            linewidth=1, 
            linestyle='-',
            grid_x=None,
            grid_y=None,
            **im_args):

    extent = (0, X.shape[1], X.shape[0], 0)

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(X, extent=extent, **im_args)
    # ax.set_xticks(range(X.shape[1]))
    # ax.set_xticklabels([])
    # ax.set_yticks(range(X.shape[0]))
    # ax.set_yticklabels([])
    # ax.grid(visible=True, 
    #         color=color, 
    #         linewidth=linewidth, 
    #         axis=axis, 
    #         linestyle=linestyle)
    if show_grid:
        if grid_x is None:
            grid_x = range(X.shape[0]+1)
        if grid_y is None:
            grid_y = range(X.shape[1]+1)
        plt.hlines(grid_x, 0, X.shape[1],
            color=color, linewidth=linewidth, linestyle=linestyle)
        plt.vlines(grid_y, 0, X.shape[0],
            color=color, linewidth=linewidth, linestyle=linestyle)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis(False)

    return im

def cluster_grid(clus, labels=None, **args):

    foo = np.where(np.diff(np.sort(clus), prepend=0))[0] - 0.5
    foo = np.concatenate([foo, [len(clus)-0.5]])

    plt.vlines(foo, len(clus), 0, 'k')
    plt.hlines(foo, len(clus), 0, 'k')

    plt.xticks(foo - np.diff(foo, prepend=0)/2, labels=labels)
    plt.yticks(foo - np.diff(foo, prepend=0)/2, labels=labels)

def hovertext(x, y, labels, offset=1, c=None, cmap=cm.tab10, 
    ax=None, max_ind=10, **scat_args):
    """
    Scatter plot of x and y, with labels when you hover over the points
    """    

    if c is None:
        c = np.zeros(len(x))

    if ax is None:
        ax = plt.gca()
        fig = ax.get_figure()

    scat = plt.scatter(x, y, c=c, **scat_args)

    annot = ax.annotate("", xy=(0,0), xytext=(5,5), textcoords="offset points", 
                        bbox={'facecolor':'white', 
                              'alpha': 0.8,
                              'boxstyle': 'round'})

    norm = plt.Normalize(1,len(np.unique(c)))
    def update_annot(ind):
        
        pos = scat.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        labs = [labels[n] for n in ind["ind"]]
        text = "{}, {}".format(" ".join(list(map(str,ind["ind"][:max_ind]))), 
                               " ".join([labels[n] for n in ind["ind"][:max_ind]]))
        if len(ind["ind"]) > max_ind:
            text += ' ...'
        annot.set_text(text)
        annot.get_bbox_patch().set_edgecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scat.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

def scatterlabel(x, y, labels, c=None, cmap=cm.tab10, ax=None, offset=1e-2, **scat_args):
    """
    Scatter plot of x and y, with labels when you hover over the points
    """    

    if c is None:
        c = 'k'

    if ax is None:
        ax = plt.gca()
        fig = ax.get_figure()

    scat = plt.scatter(x, y, c=c, **scat_args)

    xbar = [x.mean(), y.mean()]

    for n,l in enumerate(labels):
        if x[n] > xbar[0]:
            dx = -offset
            ha = 'right'
        else:
            dx = offset
            ha = 'left'
        if y[n] > xbar[1]:
            dy = -offset
            va = 'top'
            ha = 'right'
        else:
            dy = offset
            va = 'bottom'

        plt.text(x[n]+dx, y[n]+dy, l,
             horizontalalignment = ha,
             verticalalignment = va,
             bbox={'facecolor':'white', 
                   'edgecolor': 'black', 
                   'alpha': 0.8,
                   'boxstyle': 'round'})


def loglog(x, y=None, **kwargs):
    """
    A loglog function which doesn't have dumb defaults like pyplot does
    i.e. if only one vector is supplied, don't set the first x value to 
    negative infinity!!!!!
    """

    if y is None:
        yplot = x
        xplot = np.arange(1, len(yplot)+1)
    else:
        xplot = x
        yplot = y

    plt.loglog(xplot, yplot, **kwargs)


# def triangle(X, )