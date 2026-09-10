# Frozen protocol. Change only with one ADR line in DESIGN.md, then re-measure.

N_CLASSES = 3
N_SOURCE = 400
N_TARGET = 400
MAX_SAMPLES = 5000
SEED = 0
EPS = 0.08
PCA_DIM = 2
FLOOR_PCT = 85.0
SHIFT = (7.5, 2.5)

METRICS_PATH = "metrics.json"
METRICS_GEN = "web/src/metrics.gen.ts"
