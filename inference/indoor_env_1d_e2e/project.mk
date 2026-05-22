# This file can be used to set build configuration
# variables.  These variables are defined in a file called
# "Makefile" that is located next to this one.

# For instructions on how to use this system, see
# https://analogdevicesinc.github.io/msdk/USERGUIDE/#build-system

# **********************************************************

# Add your config here!
PROJECT = indoor_env_1d_e2e

# Eclipse does not always inherit shell environment variables.
# Keep this set for IDE builds; command-line builds can still override it.
MAXIM_PATH ?= /Users/hamza/MaximSDK
ARM_PREFIX ?= $(MAXIM_PATH)/Tools/GNUTools/10.3/bin/arm-none-eabi
export PATH := $(MAXIM_PATH)/Tools/GNUTools/10.3/bin:$(PATH)

# Benchmark modes (see README):
# PROJ_CFLAGS += -DE2E_PMON_ADI_BLOCKS=0      # skip ADI idle/weight/input/fused PMON (e2e loop only)
# PROJ_CFLAGS += -DE2E_PMON_ADI_ONLY=1        # stop after ADI 4-window PMON sequence
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=1        # System Power Mode: prep only
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=2        # System Power Mode: activation only
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=3        # System Power Mode: full e2e
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=4        # System Power Mode: input-copy only
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=5        # System Power Mode: fixed 1 s trigger calibration
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=6        # System Power Mode: CNN-mode timer-clear + fixed 1 s calibration
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=7        # Sustained System mW: prep loop for P_prep_mW
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=8        # Optional diagnostic: fixed prep-idle System mW
# PROJ_CFLAGS += -DE2E_SYS_PMON_REPS=2000     # override System Power repetitions
# PROJ_CFLAGS += -DE2E_SYS_PMON_ARM_DELAY_MS=2000  # optional: short System PMON arming delay for fresh-PMON test
# PROJ_CFLAGS += -DE2E_SYS_PMON_FIXED_MS=10000 # mode 5: make P1.6 high long enough to probe with meter/scope
# PROJ_CFLAGS += -DE2E_SYS_PMON_IDLE_MS=60000 # mode 8: optional prep-idle diagnostic duration
# PROJ_CFLAGS += -DE2E_PMON_PREP_BLOCK=1     # add optional e2e preprocess PMON block after ADI sequence
# PROJ_CFLAGS += -DE2E_PMON_ADI_REPS=100      # repetitions per ADI PMON block
# PROJ_CFLAGS += -DE2E_PMON_SETTLE_TICKS=500000
# PROJ_CFLAGS += -DE2E_CNN_WAIT_NOP=1        # force __NOP (default auto: 1 if alpha<=71)
# PROJ_CFLAGS += -DE2E_CNN_WAIT_NOP=0        # force sleep (91+ style)
# PROJ_CFLAGS += -DE2E_CNN_WAIT_ALPHA_THRESHOLD=71
# PROJ_CFLAGS += -DE2E_RUN_KAT=1          # optional ADI fused sampledata-loop KAT
# PROJ_CFLAGS += -DE2E_RUN_E2E_KAT=1      # only if ctf_recordings.h matches the synthesis sample
# PROJ_CFLAGS += -DE2E_NUM_RUNS=100
