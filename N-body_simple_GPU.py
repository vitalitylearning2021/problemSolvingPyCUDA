from __future__ import division

import numpy as np
import random
import pygame
from collections import defaultdict

# --- PyCUDA initialization
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray    as gpuarray

# --- ctypes for importing the DLL
import ctypes
from ctypes import * 

###################
# iDivUp FUNCTION #
###################
def iDivUp(a, b):
    # Round a / b to nearest higher integer value
    a = np.int32(a)
    b = np.int32(b)
    return (a / b + 1) if (a % b != 0) else (a / b)
    
# --- Display parameters
zoomFactor          = 1.0

WIN_N, WIN_M        = 900, 600
WIN_N2, WIN_M2      = WIN_N / 2., WIN_M / 2.

# --- Arbitrarily set gravity coefficient
G = 1.e4

# --- Planet D 
D = 0.001

# --- Inizial number of bodies
N                           = 100

# --- CUDA parameters
BLOCKSIZE = 32

# --- State vector = (posx, posy, velx, vely, ax, ay, r, m, active)
pos                         = np.zeros((N, 2), dtype = np.float32)
vel                         = np.zeros((N, 2), dtype = np.float32)
rad                         = np.zeros((N, 1), dtype = np.float32)
mass                        = np.zeros((N, 1), dtype = np.float32)
active                      = np.zeros((N, 1), dtype = np.int32)

# --- Time-step
t, dt                       = 0., 0.1  

#############################################
# SIMULATION WINDOW INITIALIZATION FUNCTION #
#############################################
def initSimulWindow(N):
    pygame.init()
    
    simulWindow = pygame.display.set_mode((WIN_N, WIN_M))

    keysBuffer = defaultdict(bool)

    pygame.display.set_caption('Simulation window. Press keypad +/- to zoom in/out - Number of current particles: {}'.format(N))

    return simulWindow, keysBuffer

##########################
# UPDATE SCREEN FUNCTION #
##########################
def updateScreen():
    # --- Update screen
    pygame.display.flip()
    # --- The window is filled with back color
    simulWindow.fill((0, 0, 0))
    # --- Lock the surface memory for pixel access, see https://kite.com/python/docs/pygame.Surface.lock
    simulWindow.lock()

##########################
# SCAN KEYBOARD FUNCTION #
##########################
def checkKeyPressed():
    while True:
        # --- Get a single event from the queue. The pygame.event queue gets pygame.KEYDOWN and pygame.KEYUP events when the keyboard buttons are pressed and released.
        evt = pygame.event.poll()
        if evt.type == pygame.NOEVENT:
            break
        elif evt.type in [pygame.KEYDOWN, pygame.KEYUP]:
            # --- key: an integer ID representing every key on the keyboard
            keysBuffer[evt.key] = evt.type == pygame.KEYDOWN

###########################
# DRAW PARTICLES FUNCTION #
###########################
def drawParticles():
    for p in range(N):
        if (active[p]):
            # circle based on their radius, but take zoomFactor factor into account
            pygame.draw.circle(simulWindow, (255, 255, 255),
                (int(WIN_N2 + zoomFactor * WIN_N2 * (pos[p, 0] - WIN_N2) / WIN_N2), int(WIN_M2 + zoomFactor * WIN_M2 * (pos[p, 1] - WIN_M2) / WIN_M2)), int(rad[p] * zoomFactor), 0)

##############################
# COMPUTE DISTANCES FUNCTION #
##############################
def computeDistances(pos1, pos2):
    dx  = pos1[0] - pos2[0]
    dy  = pos1[1] - pos2[1]
    dsq = dx * dx + dy * dy     # --- distance squared
    dr  = np.sqrt(dsq)          # --- distance
    return dx, dy, dr, dsq 

###########################
# MERGE WITH SUN FUNCTION #
###########################
def mergeWithSun(pos, rad, mass, active, N):
    for p in range(1, N):
        dx, dy, dr, dsq = computeDistances(pos[0, :], pos[p, :])
        if (dr <= (rad[p] + rad[0])):
            # --- Keep the total mass
            mass[0]   += mass[p]
            # --- Remove particle p from the active particles list  
            active[p] = False

###########################
# COMPUTE FORCES FUNCTION #
###########################

mod = SourceModule("""
    
    #define G 1.0e4f                          // --- Arbitrarily set gravity coefficient
    #define D         0.001f                          // --- Planet D 
    #define PI_f            3.1415927410125732421875f       // --- pi
    
    /************************/
    /* MERGE WITH SUN KERNEL*/
    /************************/
    __global__ void mergeWithSunKernel(const float2 * __restrict__ d_pos, float * __restrict__ d_mass, int * __restrict__ d_active, const float * __restrict__ d_rad, const int N) {
  
        int p = blockDim.x * blockIdx.x + threadIdx.x;
  
        if ((p >= N) || (p == 0) || (d_active[p] == 0)) return;

        float dx            = d_pos[0].x - d_pos[p].x;
        float dy            = d_pos[0].y - d_pos[p].y;
        float dr            = sqrtf(dx * dx + dy * dy);            // --- Distance 
        if (dr <= (d_rad[p] + d_rad[0])) {
            d_mass[0]       = d_mass[0] + d_mass[p];
            // --- Remove particle p from the active particles list
            d_active[p] = 0;
        }
    }

    /***************************/
    /* MERGE WITH PLANET KERNEL*/
    /***************************/
    __global__ void mergeWithPlanetKernel(const float2 * __restrict__ d_pos, const float * __restrict__ d_mass, int * __restrict__ d_active, float2 * __restrict__ d_vel, float * __restrict__ d_rad, const int N) {
  
        int p = blockDim.x * blockIdx.x + threadIdx.x;
  
        if ((p >= N) || (p == 0) || (d_active[p] == 0)) return;

        for (int q = 0; q < N; q++) {
            if (q != p) { 
                float dx            = d_pos[q].x - d_pos[p].x;
                float dy            = d_pos[q].y - d_pos[p].y;
                float dr            = sqrtf(dx * dx + dy * dy);            // --- Distance 
                if (dr <= (d_rad[p] + d_rad[q])) {
                    d_vel[p].x  = (d_vel[p].x * d_mass[p] + d_vel[q].x * d_mass[q]) / (d_mass[p] + d_mass[q]);
                    d_vel[p].y  = (d_vel[p].y * d_mass[p] + d_vel[q].y * d_mass[q]) / (d_mass[p] + d_mass[q]);
                    d_rad[p]    = pow((3.f * d_mass[p] / (D * 4. * PI_f)), 0.3333);  
                    // --- Remove particle q from the active particles list
                    d_active[q] = 0;
                }
            }
        }
    }

    /*******************************/
    /* SIMPLE COMPUTE FORCES KERNEL*/
    /*******************************/
    __global__ void computeForcesKernel(const float2 * __restrict__ d_pos, const float * __restrict__ d_mass, const int * __restrict__ d_active, float2 * __restrict__ d_force, const float dt, const int N) {
  
        int p = blockDim.x * blockIdx.x + threadIdx.x;
  
        if ((p >= N) || (p == 0) || (d_active[p] == 0)) return;

        float fx = 0.0f; float fy = 0.0f;

        for (int q = 0; q < N; q++) {
            float dx            = d_pos[q].x - d_pos[p].x;
            float dy            = d_pos[q].y - d_pos[p].y;
            float dsq           = dx * dx + dy * dy;            // --- Distance squared
            float invDistSqr    = 1.0f / dsq;                   // --- Inverse distance squared    
            float invDist       = rsqrtf(dsq);                  // --- Inverse distance
            if (dsq > 1.e-10f) {
                fx                  += G * d_mass[q] * invDistSqr * dx * invDist; 
                fy                  += G * d_mass[q] * invDistSqr * dy * invDist; 
            } 
        }
    
        d_force[p].x = dt * fx;
        d_force[p].y = dt * fy;
    }

 
""")

computeForcesGPU        = mod.get_function("computeForcesKernel")
mergeWithPlanetGPU      = mod.get_function("mergeWithPlanetKernel")
mergeWithSunGPU         = mod.get_function("mergeWithSunKernel")

#######################
# RK4 UPDATE FUNCTION #
#######################
def rk4GPU(d_pos, d_vel, d_rad, d_mass, d_active, d_k1v, d_k2v, d_k3v, d_k4v, N):

    blockDim  = (BLOCKSIZE, 1, 1)
    gridDim   = (int(iDivUp(N, BLOCKSIZE)), 1, 1)

    # --- If too close to sun, merge planets with sun
    mergeWithSunGPU(d_pos, d_mass, d_active, d_rad, np.int32(N), block = blockDim, grid = gridDim)

    # --- If two planets are too close, merge them keeping angular momentum and total mass
    mergeWithPlanetGPU(d_pos, d_mass, d_active, d_vel, d_rad, np.int32(N), block = blockDim, grid = gridDim)
  
    # --- Compute k1v, k1x
    computeForcesGPU(d_pos, d_mass, d_active, d_k1v, np.float32(dt), np.int32(N), block = blockDim, grid = gridDim)
    d_k1x   = dt * d_vel

    # --- Compute k2v, k2x
    computeForcesGPU(d_pos + 0.5 * d_k1x, d_mass, d_active, d_k2v, np.float32(dt), np.int32(N), block = blockDim, grid = gridDim)
    d_k2x = dt * (d_vel + 0.5 * d_k1v)

    # --- Compute k3v, k3x
    computeForcesGPU(d_pos + 0.5 * d_k2x, d_mass, d_active, d_k3v, np.float32(dt), np.int32(N), block = blockDim, grid = gridDim)
    d_k3x = dt * (d_vel + 0.5 * d_k2v)

    # --- Compute k4v, k4x
    computeForcesGPU(d_pos + 0.5 * d_k3x, d_mass, d_active, d_k4v, np.float32(dt), np.int32(N), block = blockDim, grid = gridDim)
    d_k4x = dt * (d_vel + d_k3v)

    d_vel   = d_vel + (1. / 6.) * (d_k1v + 2. * d_k2v + 2. * d_k3v + d_k4v)
    d_pos   = d_pos + (1. / 6.) * (d_k1x + 2. * d_k2x + 2. * d_k3x + d_k4x)

    return d_pos, d_vel, d_rad, d_mass, d_active

#############################
# ARRAY COMPACTION FUNCTION #
#############################
def arrayCompaction(pos, vel, rad, mass, active, N):
    indices                             = np.argsort(active)
    np.take_along_axis(pos,     indices, axis = 0)
    np.take_along_axis(vel,     indices, axis = 0)
    np.take_along_axis(mass,    indices, axis = 0)
    np.take_along_axis(rad,     indices, axis = 0)
    np.take_along_axis(active,  indices, axis = 0)
    N                                   = np.sum(active, axis = 0)
    N                                   = int(N[0])

class FLOAT2(Structure):
    _fields_ = ("x", c_float), ("y", c_float)

def get_array_compaction_GPU():
    dll = ctypes.windll.LoadLibrary("arrayCompactionGPUDLL.dll") 
    func = dll.arrayCompactionGPU
    dll.arrayCompactionGPU.argtypes = [POINTER(c_int), POINTER(FLOAT2), POINTER(FLOAT2), POINTER(c_float), POINTER(c_float), c_size_t] 
    dll.arrayCompactionGPU.restype  = c_int
    return func

arrayCompactionGPU = get_array_compaction_GPU()

# --- State inizialization
pos[:, 0]                   = WIN_N * np.random.rand(N,)
pos[:, 1]                   = WIN_M * np.random.rand(N,)
vel[:, 0]                   = 200. * (np.random.rand(N,) - 0.5)
vel[:, 1]                   = 200. * (np.random.rand(N,) - 0.5)
rad[:, 0]                   = 1.5 * np.random.rand(N,)
mass[:, 0]                  = D * 4. * np.pi * (rad[:, 0]**3.) / 3.
active[:, 0]                = 1

# --- Sun
pos[0, 0]                   = WIN_N2
pos[0, 1]                   = WIN_M2
vel[0, 0]                   = 0.
vel[0, 1]                   = 0.
mass[0, 0]                  = 100.
rad[0, 0]                   = (3. * mass[0] / (D * 4. * np.pi))**(0.3333)
active[0, 0]                = 1

# --- Initialize simulation window
simulWindow, keysBuffer = initSimulWindow(N)

d_k1v                       = gpuarray.zeros((N, 2), dtype = np.float32)
d_k2v                       = gpuarray.zeros((N, 2), dtype = np.float32)
d_k3v                       = gpuarray.zeros((N, 2), dtype = np.float32)
d_k4v                       = gpuarray.zeros((N, 2), dtype = np.float32)

#############
# MAIN LOOP #
#############
while True:
    t += dt
 
    updateScreen()

    d_pos                       = gpuarray.to_gpu(pos)
    d_vel                       = gpuarray.to_gpu(vel)
    d_mass                      = gpuarray.to_gpu(mass)
    d_rad                       = gpuarray.to_gpu(rad)
    d_active                    = gpuarray.to_gpu(active)

    d_k1v.fill(np.float32(0))
    d_k2v.fill(np.float32(0))
    d_k3v.fill(np.float32(0))
    d_k4v.fill(np.float32(0))

    d_pos, d_vel, d_rad, d_mass, d_active = rk4GPU(d_pos, d_vel, d_rad, d_mass, d_active, d_k1v, d_k2v, d_k3v, d_k4v, N)   

    d_active_p                  = ctypes.cast(d_active.ptr, ctypes.POINTER(c_int))
    d_pos_p                     = ctypes.cast(d_pos.ptr,    ctypes.POINTER(FLOAT2))
    d_vel_p                     = ctypes.cast(d_vel.ptr,    ctypes.POINTER(FLOAT2))
    d_rad_p                     = ctypes.cast(d_rad.ptr,    ctypes.POINTER(c_float))
    d_mass_p                    = ctypes.cast(d_mass.ptr,   ctypes.POINTER(c_float))
    N = arrayCompactionGPU(d_active_p, d_pos_p, d_vel_p, d_rad_p, d_mass_p, N)

    active  = d_active.get()
    pos     = d_pos.get()    
    vel     = d_vel.get()
    mass    = d_mass.get()
    rad     = d_rad.get()

    N = gpuarray.sum(d_active, dtype = np.int32)
    N = N.get()
    
    drawParticles()
    simulWindow.unlock()
    checkKeyPressed()

    pygame.display.set_caption('Simulation window. Press keypad +/- to zoom in/out - Number of current particles: {}'.format(N))
    
    # --- React to keyboard + and -
    if keysBuffer[pygame.K_KP_PLUS]:
        zoomFactor /= 0.99
    if keysBuffer[pygame.K_KP_MINUS]:
        zoomFactor /= 1.01
    # --- React to escape
    if keysBuffer[pygame.K_ESCAPE]:
        break
