set -euo pipefail
torchrun --standalone --nproc-per-node=4 -m lbep.entrypoint train --config configuration/main.yaml
