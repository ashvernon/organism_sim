"""
Simulation tuning knobs.
"""

# Population controls
START_POP = 25
MAX_POP = 120

# Energy + life
REPRO_ENERGY_THRESHOLD = 8.0
REPRO_COST = 4.0
CHILD_SPAWN_JITTER = 32.0
DEATH_ENERGY_FLOOR = 0.1
MAX_AGE_SECONDS = 120.0

# Runtime pacing
SIM_SPEED = 2  # simulation sub-steps per rendered frame

# Environment
SCREEN_W, SCREEN_H = 980, 720

# Physics/brain tuning
ENERGY_DRAIN_PER_SEC = 0.06
EAT_REACH = 14.0

# Mutation
MUT_P_WEIGHT = 0.12
MUT_P_BIAS = 0.10
MUT_SIGMA = 0.30
