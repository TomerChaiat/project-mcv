"""Project-wide numerical constants used by both tuning and evaluation."""

# Cached probabilities are float16.  Values below this can underflow to zero,
# so every NLL/CE path uses the same explicit floor.
PROB_EPS = 1e-7
