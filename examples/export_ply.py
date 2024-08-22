import os
import torch
import numpy as np
from plyfile import PlyElement, PlyData
import tyro
from dataclasses import dataclass

@dataclass
class ExportConfig:
    # Path to the checkpoint file
    ckpt: str
    # Output PLY file path (optional)
    output: str = None

def construct_list_of_attributes(splats):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    if "sh0" in splats:
        for i in range(splats["sh0"].shape[1] * splats["sh0"].shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(splats["shN"].shape[1] * splats["shN"].shape[2]):
            l.append(f'f_rest_{i}')
    else:
        for i in range(splats["colors"].shape[1]):
            l.append(f'f_dc_{i}')
        for i in range(splats["features"].shape[1]):
            l.append(f'f_rest_{i}')
    l.append('opacity')
    for i in range(splats["scales"].shape[1]):
        l.append(f'scale_{i}')
    for i in range(splats["quats"].shape[1]):
        l.append(f'rot_{i}')
    return l

def save_ply(splats, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = splats["means"].cpu().numpy()
    normals = np.zeros_like(xyz)
    opacities = splats["opacities"].unsqueeze(-1).cpu().numpy()
    scale = splats["scales"].cpu().numpy()
    rotation = splats["quats"].cpu().numpy()
    
    if "sh0" in splats:
        f_dc = splats["sh0"].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = splats["shN"].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    else:
        f_dc = splats["colors"].cpu().numpy()
        f_rest = splats["features"].cpu().numpy()
    
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(splats)]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def main(cfg: ExportConfig):
    # Load the checkpoint
    ckpt = torch.load(cfg.ckpt, map_location='cpu')
    splats = ckpt["splats"]

    # Export to PLY
    output_path = cfg.output if cfg.output else f"splat_{ckpt['step']}.ply"
    save_ply(splats, output_path)
    print(f"PLY file exported to: {output_path}")

if __name__ == "__main__":
    cfg = tyro.cli(ExportConfig)
    main(cfg)