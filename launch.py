import subprocess
from concurrent import futures

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=int, nargs="*", default=[0])
parser.add_argument("--cfg", default="./configs/fit_img_dir_no_split.cfg")
args = parser.parse_args()


seqs = [
    ("023390_mpii_test", "02"),
    ("024159_mpii_test", "03"),
]

with futures.ProcessPoolExecutor(max_workers=len(args.gpus)) as ex:
    for i, (video_seq, track_id) in enumerate(seqs):
        gpu = args.gpus[i % len(args.gpus)]
        cmd_args = [
            f"CUDA_VISIBLE_DEVICES={gpu}",
            "python humor/fitting/run_fitting.py",
            "--video-seq",
            video_seq,
            "--track-id",
            track_id,
            f"@{args.cfg}",
        ]
        cmd = " ".join(cmd_args)
        print(cmd)
        ex.submit(subprocess.call, cmd, shell=True)
