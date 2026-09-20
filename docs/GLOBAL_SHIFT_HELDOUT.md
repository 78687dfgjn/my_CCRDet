# Sequence-held-out global shift validation

The validation annotation metadata contains numeric filenames (`04900.jpg`
through `06124.jpg`) and image IDs, but no sequence/video identifier or
frame-range metadata. The released dataset loader also exposes only image IDs
and filenames. A reliable sequence grouping cannot therefore be reconstructed
from the released files.

Because a random image split would leak adjacent frames, no DEV/HOLDOUT split
was fabricated and no held-out claim is made. The cross-checkpoint shift test
is reported separately; it is not a substitute for sequence-held-out
validation.

Required status: **NEED MORE EVIDENCE** for held-out generalization.
