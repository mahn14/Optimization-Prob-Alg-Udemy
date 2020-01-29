# Particle Swarm Algorithma
import numpy as np
import matplotlib.pyplot as plt
import ObjectiveFunction as obj

# Define Objective Function params
nVar = 10
ub = np.ones(10)*10   # upper bound on position
lb = np.ones(10)*-10  # lower bound on position

# Define PSO params
nParticles = 30
maxIter = 500
wMax = 0.9
wMin = 0.2
c1 = 2  # self direction weight
c2 = 2  # group direction weight
vMax = (ub - lb)*0.2
vMin = -vMax

# PSO Algorithm: X (Position) and V (Velocity)
SwarmParticlesX = [0]*nParticles
SwarmParticlesV = [0]*nParticles
SwarmParticlesPBEST_X = [0]*nParticles
SwarmParticlesPBEST_O = [0]*nParticles

# Initial Position and Velocity
for k in range(nParticles):
    SwarmParticlesX[k] = (ub-lb) * np.random.rand(10) + lb #random starting position
    SwarmParticlesV[k] = np.zeros(nVar)                    #no velocity

    SwarmParticlesPBEST_X[k] = np.zeros(nVar)              #no best position yet
    SwarmParticlesPBEST_O[k] = np.inf                      #best is infinite

# Initial Group Solution
SwarmParticlesGBEST_X = np.zeros(nVar)              #same for group
SwarmParticlesGBEST_O = np.inf
BEST = [0]*maxIter


# Iter
# Apply PSO Algorithm
for t in range(maxIter):

    # UPDATES
    SwarmParticles_O = [0]*nParticles
    for k in range(nParticles):

        # Update ObjectiveFunction
        currentX = SwarmParticlesX[k]
        currentO = obj.ObjectiveFunction(currentX)

        # Update PBEST
        if currentO < SwarmParticlesPBEST_O[k]: # Symbol < since we are minimizing
            SwarmParticlesPBEST_X[k] = currentX
            SwarmParticlesPBEST_O[k] = currentO

        # Update GBEST
        if currentO < SwarmParticlesGBEST_O:
            SwarmParticlesGBEST_X = currentX
            SwarmParticlesGBEST_O = currentO

    # Update weight
    w= wMax - t*((wMax - wMin) / maxIter)

    # Update Position and Velocity
    for k in range(nParticles):

        # Update the Velocity
        SwarmParticlesV[k] = w*SwarmParticlesV[k] + \
                             c1*np.random.rand(10)*(SwarmParticlesPBEST_X[k] - SwarmParticlesX[k]) + \
                             c2*np.random.rand(10)*(SwarmParticlesGBEST_X - SwarmParticlesX[k])

        # Check Velocity is within bounds
        SwarmParticlesV[k] = np.array([max(x) for x in zip(SwarmParticlesV[k], vMin)])
        SwarmParticlesV[k] = np.array([min(x) for x in zip(SwarmParticlesV[k], vMax)])

        # Update current position
        SwarmParticlesX[k] = SwarmParticlesX[k] + SwarmParticlesV[k]

        # Check Position is within bounds
        SwarmParticlesX[k] = np.array([max(x) for x in zip(SwarmParticlesX[k], lb)])
        SwarmParticlesX[k] = np.array([min(x) for x in zip(SwarmParticlesX[k], ub)])

    # Solution
    BEST[t] = SwarmParticlesGBEST_O
    results = ['Iteration: ', t, 'GBEST: ', SwarmParticlesGBEST_O]
    print(results)

# Visualize Convergence
plt.plot(BEST)
plt.show()
