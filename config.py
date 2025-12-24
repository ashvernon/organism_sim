"""
Simulation tuning knobs.
"""

# Population controls
START_POP = 25
MAX_POP = 120

# Energy + life
REPRO_ENERGY_THRESHOLD = 7.0
REPRO_COST = 3.0
CHILD_SPAWN_JITTER = 32.0
DEATH_ENERGY_FLOOR = 0.1
MAX_AGE_SECONDS = 120.0
MAX_ENERGY = 16.0

# Runtime pacing
SIM_SPEED = 2  # simulation sub-steps per rendered frame

# Environment
SCREEN_W, SCREEN_H = 980, 720

# Physics/brain tuning
ENERGY_DRAIN_PER_SEC = 0.012
EAT_REACH = 22.0

# Movement/actuation
ACTUATOR_COST_SCALE = 0.08

# Food field
FOOD_TARGET_PELLETS = 520
FOOD_SPAWN_RATE = 2.1
FOOD_RADIUS_RANGE = (2.4, 7.2)
FOOD_LIFESPAN_RANGE = (18.0, 260.0)

# Mutation
MUT_P_WEIGHT = 0.12
MUT_P_BIAS = 0.10
MUT_SIGMA = 0.30
