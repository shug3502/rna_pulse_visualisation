#!/usr/bin/python

#************ Prof. Ilan Davis, Dept of Biochemistry, the University of Oxford *************"""
"""
BASED ON:
Animation of Elastic collisions with Gravity

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
# To save the animation as an .mp4 requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

#*************************** EXPLANATION FROM ILAN *********************************************
# Black balls are at equiplibrium at start of animation (decay rate = synthesis rate).
# After a delay, 4sU is added and incorporation into newly synthesised RNA starts, so that all new balls are red
# and no new black balls appear and remaining black ball numbers decay exponentially
# New red balls start appearing in the nucleus and are exported to the cytoplasm.
# Over time, all black balls decay, and only red balls are present.
#*************************** EXPLANATION FROM ILAN *********************************************
import pandas as pd
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
#import scipy.integrate as integrate
import matplotlib.animation as animation
import imageio
import os

#********************************* SET FIXED PARAMETERS (CAN MODIFY HERE - but only some are simple) **************************
MAX_PARTICLES = 500  # total number of black particles and red particles
SLOW_ANIMATION_DOWN = 0.02  # time delay in seconds to slow down animation and the decay equation calculation that governs num_red and num_black.
PARTICLE_SIZE = 0.02 # size of the bouncing balls
NUCLEAR_SIZE = 0.1 # round light grey circle in the middle of the cell
CELL_SIZE = 1  # does not work if change this parameter alone.
DELTA_TIME = 0.01 # increment of time used for the decay equation independently of the dt increment for the animation
FIG_SIZE = 10 # size of figure and movie, but cannot be altered on its onw - affects other factors that need adjusting too (complex and confusing)
DELAY_BEFORE_RED = 2 # Time for first part of movie where only red particles are present, before red particles appear
SPEED_SYNTHESIS = 100 # Arbitrary constant in equation (not molecules per second for example)
SPEED_DECAY = 150     # Arbitrary constant in equation (not molecules per second for example)
NUCLEUS_LOCATION = (-0.4,-0.8)
#********************************* SET FIXED PARAMETERS (CAN MODIFY HERE) **************************

makemovie = input('Display or save movie [d/s]?: ') # user inputs whether to display animation or save to mp4 omvie
im_str = input('Path to image to plot: ') #user inputs path to image of cell
if type(im_str) is not str:
    print('oops need to input a string as a path to image')
    raise TypeError
elif not os.path.isfile(im_str): 
    print('oops the file path input does not exist')
    raise FileExistsError

totaltime = 0  # set the "time counter" for animation to zero
num_red = 0
num_black = 0
num_red_decayed = 0
num_black_decayed = 0
direction = 'forwards' #start by moving forward in a window of particles to display from the pre-created array.

#check if two line segments intersect, see https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def aux(a):
    if len(a)>0:
        return a[0]
    else:
        return []

def rotate(x,theta=np.pi/2):
    """https://en.wikipedia.org/wiki/Rotation_matrix"""
    rotation_matrix = np.array([[np.cos(theta), np.sin(-theta)],[np.sin(theta), np.cos(theta)]])
    if len(x)>0:
        return np.dot(rotation_matrix,x)
    else:
        return []

def get_normal_vector(prev_state,state,cell_path):
    """find which part of the cell path particles are interacting with and return normal vectors for the$
        prev_state - array (Nx4) where N is num of particles
        state - array (Nx4) 
        cell_path - matplotlib path object for boundary of cell """
    sx,sy = prev_state.shape
    if sy!=4 or (sx,sy)!=state.shape:
        raise ValueError("states should be the same size and consist of positions and velocities")
    which_boundaries_interacted_with = [np.array([intersect(prev_state[particle,:],state[particle,:],cell_path.vertices[i],cell_path.vertices[i+1]) \
for i in range(cell_path.vertices.shape[0]-1)]) for particle in range(state.shape[0])]
    boundary_segments = [aux(np.nonzero(b)[0]) for b in which_boundaries_interacted_with]
    normal_vectors = [rotate(cell_path.vertices[b+1]-cell_path.vertices[b]) if not isinstance(b,list) else [] for b in boundary_segments]
    return normal_vectors

def reflect(n,v):
    """ reflection in a line in the plane (or higher dimensions, should still work)
    n - normal vector
    v - direction of travel
    returns new direction of travel
    see https://en.wikipedia.org/wiki/Reflection_%28mathematics%29#Reflection_across_a_line_in_the_plane
    """
    if len(n) > 0:
        new_direction = v - 2/np.dot(n,n)*(np.dot(n,v)*n)
        return new_direction
    else:
        #no reflection needed
        return v

def reflect_all_particles(prev_state,state,cell_path):
    """ implement reflection for all the particles """
    if len(prev_state) + len(prev_state) <=0:
        print('nothing to reflect')
        return  None
    normal_vectors = get_normal_vector(prev_state,state,cell_path)
    new_velocities = [reflect(normal_vectors[i],state[i,2:]) for i in range(state.shape[0])]
    new_state = state.copy()
    new_state[:,2:4] = np.array(new_velocities)
    return new_state

class ParticleBox:

    def __init__(self,
                 init_state = [[1, 0, 0, -1],
                               [-0.5, 0.5, 0.5, 0.5],
                               [-0.5, -0.5, -0.5, 0.5]],
                 bounds = [-CELL_SIZE, CELL_SIZE, -CELL_SIZE, CELL_SIZE],  # was 2
                 bounds_nucleus = [-NUCLEAR_SIZE, NUCLEAR_SIZE, -NUCLEAR_SIZE, NUCLEAR_SIZE],
                 size = PARTICLE_SIZE,
                 M = 0.02,
                 G = 9.8):
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
	self.prev_state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.bounds_nucleus = bounds_nucleus
        self.G = G

    def step(self, dt, particle_color, path):
        """step once by dt seconds"""
        self.time_elapsed += dt
	self.prev_state = self.state.copy() #store previous state to help check for boundary interactions
        # update positions
        if particle_color == 'red':
            self.state[num_red_decayed:num_red, :2] += dt * self.state[num_red_decayed:num_red, 2:]
        else:
            self.state[num_black_decayed:num_black, :2] += dt * self.state[num_black_decayed:num_black, 2:]
        crossed = [not p for p in path.contains_points(self.state[:,:2])]
        prev_crossed = [not p for p in path.contains_points(self.prev_state[:,:2])]
	if sum(crossed)>0:
	    r = reflect_all_particles(self.prev_state[crossed,:],self.state[crossed,:],path)
            self.state[crossed, 2:] = r[:,2:]
        mistakes = np.array(crossed) & np.array(prev_crossed)
        self.state[mistakes,2:] = -self.prev_state[mistakes,2:]
#            self.state[crossed, 2] *= -1
#            self.state[crossed, 3] *= -1

#------------------------------------------------------------
im = imageio.imread(im_str)

def get_cell_vertices_from_csv(file_str, slice=56, total_slices=100):
    my_names = ['filename','t','z'] + ['region' + str(x) for x in range(total_slices)] #column names
    kk = pd.read_csv(file_str,header=0,names=my_names)
    even_rows = kk.iloc[::2] # x coords
    odd_rows = kk.iloc[1::2] # y coords
    even_rows.drop(even_rows.columns[[0,1,2]], axis=1, inplace=True)
    odd_rows.drop(odd_rows.columns[[0,1,2]], axis=1, inplace=True)
    cell_vertices = np.transpose(np.concatenate((np.array(even_rows.iloc[[slice]]),np.array(odd_rows.iloc[[slice]])),axis=0))
    return cell_vertices

def convert_from_pxls_to_fig_coords(pxl_coords, image_size):
    fig_coords = [p*CELL_SIZE/image_size[j]*2 - CELL_SIZE for j,p in enumerate(pxl_coords)]
    return fig_coords

#cell_vertices = get_cell_vertices_from_csv('/Users/jonathan/Dropbox/DTC_DPhil/ilan_visualisation/cell1.csv', slice=40)
cell_vertices = get_cell_vertices_from_csv('_outputROI.csv',slice=0)
cell_vertices = [convert_from_pxls_to_fig_coords([c[1],c[0]],im.shape) for c in cell_vertices]
if NUCLEUS_LOCATION is None:
    NUCLEUS_LOCATION = np.mean(cell_vertices,axis=0)

# set up initial state
# create a landscape of 1000,000 red and black particles in the nucleus. Only display and move a subset.
num_particles = 1000
np.random.seed(0)
init_state = -0.5 + np.random.random((num_particles, 4)) # was: -0.5
init_state[:, :2] = 0.1*init_state[:, :2] + NUCLEUS_LOCATION  # nwas 2*CELL_SIZE - 0.05 the -0.05 keeps particles from spilling over the cell membrane)

box1 = ParticleBox(init_state, size=PARTICLE_SIZE)

init_state = -0.5 + np.random.random((num_particles, 4))
init_state[:, :2] = 0.1*init_state[:, :2] + NUCLEUS_LOCATION

box2 = ParticleBox(init_state, size=PARTICLE_SIZE)
dt = 1. / 30 # 30fps

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure(figsize=(FIG_SIZE,FIG_SIZE))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-CELL_SIZE, CELL_SIZE), ylim=(-CELL_SIZE, CELL_SIZE))
# particles holds the locations of the particles
particles1, = ax.plot([], [], 'ko', ms=6)
particles2, = ax.plot([], [], 'ro', ms=6)

cell_path = Path(cell_vertices)
patch = patches.PathPatch(cell_path, facecolor=None, lw=2, alpha=0.5)
ax.add_patch(patch)

# circle is the nucleus in the middle
circ = plt.Circle(NUCLEUS_LOCATION, NUCLEAR_SIZE, color='lightgrey')
ax.add_patch(circ)

use_rect = False
# rect is the box edge
if use_rect:
    rect = plt.Rectangle(box1.bounds[::2], box1.bounds[1] - box1.bounds[0],box1.bounds[1] - box1.bounds[0],ec='none', lw=2, fc='none')
    rect.set_edgecolor('k')
    ax.add_patch(rect)
else:
#    ax.imshow(np.flipud(np.transpose(im,(1,0,2))),aspect='equal',extent=[-CELL_SIZE, CELL_SIZE, -CELL_SIZE, CELL_SIZE])
    rect = None


def init():
    """initialize animation"""
    global box1, box2, rect, dt
    particles1.set_data([], [])
    particles2.set_data([], [])
    if use_rect:
        rect.set_edgecolor('black')
    return particles1, particles2, rect

def animate(i):
    """perform animation step"""
    global box1, box2, rect, dt, ax, fig, totaltime, num_red, num_red_decayed, num_black, num_black_decayed, direction, cell_path
    totaltime = totaltime + DELTA_TIME # the counter that keeps time independently of the animation used in the decay equation
    if totaltime < DELAY_BEFORE_RED:  #plot just black particles for the first period until totaltime=5
        justblack()
    else:
        black_and_red()
    return particles1, particles2, rect

def justblack(): # black balls synthesised and decay equal at equilibrium in first part of the movie
    global box1, box2, rect, dt, ax, fig, totaltime, num_red, num_red_decayed, num_black, num_black_decayed, direction, cell_path
    if num_black < num_particles:
        num_black = int((SPEED_SYNTHESIS*(totaltime)) + (MAX_PARTICLES-MAX_PARTICLES*np.exp(-totaltime)))
               #move the window displayed along to give impression of synthesis.  black particles grow to equiplib.
        num_black_decayed = int(SPEED_DECAY*(totaltime)) #move the window displayed along to give impression of decay
    particle_color = 'black'
    box1.step(dt, particle_color, cell_path)
    particles1.set_data(box1.state[num_black_decayed:num_black, 0], box1.state[num_black_decayed:num_black, 1]) #update red particles
    return particles1, particles2, rect

def black_and_red(): # black balls at equilib and red balls begining to synthesise
    global box1, box2, rect, dt, ax, fig, totaltime, num_red, num_red_decayed, num_black, num_black_decayed, direction, cell_path
    if num_black_decayed != num_black:  # black particles remain the same number and just decay
            num_black_decayed = int(SPEED_DECAY*(totaltime))  # black particles decay at constant rate
    #4sU is incorporated to make red particles after a delay - black particles have reached equilibrium
    if num_red < num_particles:
        num_red = int(SPEED_SYNTHESIS*(totaltime-DELAY_BEFORE_RED)) + int(MAX_PARTICLES-MAX_PARTICLES*np.exp(DELAY_BEFORE_RED-totaltime)) # red particles grow to max
        num_red_decayed = int(SPEED_DECAY*(totaltime-DELAY_BEFORE_RED)) # red particles decay at constant rate
    box1.step(dt, 'black', cell_path)
    particles1.set_data(box1.state[num_black_decayed:num_black, 0], box1.state[num_black_decayed:num_black, 1]) #update red
    box2.step(dt, 'red', cell_path)
    particles2.set_data(box2.state[num_red_decayed:num_red, 0], box2.state[num_red_decayed:num_red, 1]) #update red particles
    return particles1, particles2, rect

ani = animation.FuncAnimation(fig, animate, frames=600, interval=10, blit=use_rect, init_func=init) #600 frame movie

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see http://matplotlib.sourceforge.net/api/animation_api.html

if makemovie == 's':
    ani.save('decay.mp4', fps=30, extra_args=['-vcodec', 'libx264']) #write movie instead of displaying
else:
    plt.show()
