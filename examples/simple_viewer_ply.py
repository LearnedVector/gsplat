import argparse
import time
from typing import Tuple

import numpy as np
import torch
import viser
import nerfview
from plyfile import PlyData

from gsplat.rendering import rasterization

def load_ply(path):
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    
    xyz = np.stack((vertex['x'], vertex['y'], vertex['z']), axis=-1)
    opacities = vertex['opacity']
    
    f_dc = np.stack([vertex[f'f_dc_{i}'] for i in range(3)], axis=-1)
    scales = np.stack([vertex[f'scale_{i}'] for i in range(3)], axis=-1)
    rotations = np.stack([vertex[f'rot_{i}'] for i in range(4)], axis=-1)
    
    return {
        "means": torch.tensor(xyz, dtype=torch.float32),
        "opacities": torch.tensor(opacities, dtype=torch.float32),
        "colors": torch.tensor(f_dc, dtype=torch.float32),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "quats": torch.tensor(rotations, dtype=torch.float32),
    }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load PLY file
    splats = load_ply(args.ply_path)
    for k, v in splats.items():
        splats[k] = v.to(device)
    
    print("Number of Gaussians:", len(splats["means"]))

    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        render_colors, _, _ = rasterization(
            splats["means"],
            splats["quats"],
            splats["scales"],
            splats["opacities"],
            splats["colors"],
            viewmat[None],
            K[None],
            width,
            height,
            render_mode="RGB",
        )
        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print(f"Viewer running on port {args.port}... Ctrl+C to exit.")
    time.sleep(100000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the .ply file")
    parser.add_argument("--port", type=int, default=8080, help="Port for the viewer server")
    args = parser.parse_args()

    main(args)
