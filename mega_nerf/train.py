from argparse import Namespace

import torch
from torch.distributed.elastic.multiprocessing.errors import record
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from mega_nerf.opts import get_opts_base
from mega_nerf.runner import Runner


def _get_train_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--dataset_path', type=str, required=True)

    return parser.parse_args()

@record
def main(hparams: Namespace) -> None:
    if hparams.detect_anomalies:
        with torch.autograd.detect_anomaly():
            Runner(hparams).train()
    else:
        Runner(hparams).train()


if __name__ == '__main__':
    main(_get_train_opts())
