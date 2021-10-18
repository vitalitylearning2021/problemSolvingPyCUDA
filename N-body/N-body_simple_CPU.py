import numpy as np
import random
import pygame
from collections import defaultdict

# --- Display parameters
zoomFactor                = 1.0

WIN_N, WIN_M        = 900, 600
WIN_N2, WIN_M2      = WIN_N / 2., WIN_M / 2.

# --- Arbitrarily set gravity coefficient
G = 1.e4

# --- Planet D 
D = 0.001

# --- Inizial number of bodies
N                           = 100

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

###############################
# MERGE WITH PLANETS FUNCTION #
###############################
def mergeWithPlanet(pos, vel, rad, mass, active, N):
    for p in range(1, N):
        if (active[p]):
            for q in range(1, N):
                if (p != q):
                    dx, dy, dr, dsq = computeDistances(pos[q, :], pos[p, :])
                    if (dr <= (rad[p] + rad[q])):
                        # --- Keep the angular momentum and the mass
                        vel[p, 0] = (vel[p, 0] * mass[p] + vel[q, 0] * mass[q]) / (mass[p] + mass[q])
                        vel[p, 1] = (vel[p, 1] * mass[p] + vel[q, 1] * mass[q]) / (mass[p] + mass[q])
                        rad[p] = (3. * mass[p] / (D * 4. * np.pi))**(0.3333)  
                        # --- Remove particle q from the active particles list
                        active[q] = False

###########################
# COMPUTE FORCES FUNCTION #
###########################
def computeForces(pos, mass, active, forcesx, forcesy, N):
    for p in range(1, N):
        if (active[p]):
            for q in range(N):
                if (p != q):
                    dx, dy, dr, dsq = computeDistances(pos[q, :], pos[p, :])
                    forcesx[p, q] = (G * mass[p] * mass[q] / dsq) * dx / dr if dsq > 1e-10 else 0.
                    forcesy[p, q] = (G * mass[p] * mass[q] / dsq) * dy / dr if dsq > 1e-10 else 0.

#####################
# UPDATEKV FUNCTION #
#####################
def updatekv(kv, forcesx, forcesy, mass, active, dt, N):
    for p in range(1, N):
        if (active[p]):
            for q in range(N):
                if (p != q):
                    kv[p, 0]   += dt * forcesx[p, q] / mass[p]
                    kv[p, 1]   += dt * forcesy[p, q] / mass[p]

#######################
# RK4 UPDATE FUNCTION #
#######################
def rk4(pos, vel, rad, mass, active, N):
    # --- If too close to sun, merge planets with sun
    mergeWithSun(pos, rad, mass, active, N)
    # --- If two planets are too close, merge them keeping angular momentum and total mass
    mergeWithPlanet(pos, vel, rad, mass, active, N)

    forcesx                     = np.zeros((N, N), dtype = np.float32)
    forcesy                     = np.zeros((N, N), dtype = np.float32)
    k1v                         = np.zeros((N, 2), dtype = np.float32)
    k1x                         = np.zeros((N, 2), dtype = np.float32)
    k2v                         = np.zeros((N, 2), dtype = np.float32)
    k2x                         = np.zeros((N, 2), dtype = np.float32)
    k3v                         = np.zeros((N, 2), dtype = np.float32)
    k3x                         = np.zeros((N, 2), dtype = np.float32)
    k4v                         = np.zeros((N, 2), dtype = np.float32)
    k4x                         = np.zeros((N, 2), dtype = np.float32)

    # --- Compute k1v, k1x
    computeForces(pos, mass, active, forcesx, forcesy, N)
    updatekv(k1v, forcesx, forcesy, mass, active, dt, N)
    k1x = dt * vel[0 : N]
    
    # --- Compute k2v, k2x
    forcesx.fill(0.)
    forcesy.fill(0.)
    computeForces(pos[0 : N] + 0.5 * k1x, mass, active, forcesx, forcesy, N)
    updatekv(k2v, forcesx, forcesy, mass, active, dt, N)
    k2x = (vel[0 : N] + 0.5 * k1v) * dt

    # --- Compute k3v, k3x
    forcesx.fill(0.)
    forcesy.fill(0.)
    computeForces(pos[0 : N] + 0.5 * k2x, mass, active, forcesx, forcesy, N)
    updatekv(k3v, forcesx, forcesy, mass, active, dt, N)
    k3x = (vel[0 : N] + 0.5 * k2v) * dt

    # --- Compute k4v, k4x
    forcesx.fill(0.)
    forcesy.fill(0.)
    computeForces(pos[0 : N] + k3x, mass, active, forcesx, forcesy, N)
    updatekv(k4v, forcesx, forcesy, mass, active, dt, N)
    k4x = (vel[0 : N] + k3v) * dt

    vel[0 : N] = vel[0 : N] + (1. / 6.) * (k1v + 2. * k2v + 2. * k3v + k4v)
    pos[0 : N] = pos[0 : N] + (1. / 6.) * (k1x + 2. * k2x + 2. * k3x + k4x)

    return pos, vel, rad, mass, active

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
    return pos, vel, rad, mass, active, N

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

#############
# MAIN LOOP #
#############
while True:
    t += dt
 
    updateScreen()

    pos, vel, rad, mass, active         = rk4(pos, vel, rad, mass, active, N)                      

    pos, vel, rad, mass, active, N = arrayCompaction(pos, vel, rad, mass, active, N)

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
