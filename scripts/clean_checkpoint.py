import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
args = parser.parse_args()
ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
state = ckpt["model"] if "model" in ckpt else ckpt
torch.save({"model": state}, args.dst)
torch.load(args.dst, map_location="cpu", weights_only=True)
print(f"{args.src} -> {args.dst}: {len(state)} tensors, dropped keys {sorted(set(ckpt) - {'model'})}")
