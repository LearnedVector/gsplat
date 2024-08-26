import os
import json
import numpy as np
from datasets.colmap import Parser

def save_camera_params_as_json(parser: Parser, save_path: str) -> None:
    """Save camera intrinsics and extrinsics as a single JSON file with detailed labels."""
    data = {
        "intrinsics": {},
        "extrinsics": {}
    }

    # Save intrinsics (K matrix) for each camera
    for camera_id, K in parser.Ks_dict.items():
        data["intrinsics"][camera_id] = {
            "fx": K[0, 0],  # Focal length in x
            "fy": K[1, 1],  # Focal length in y
            "cx": K[0, 2],  # Principal point x-coordinate
            "cy": K[1, 2],  # Principal point y-coordinate
            "K_matrix": K.tolist()  # Full K matrix
        }

    # Save extrinsics (camera-to-world matrices) for each image
    for idx, camtoworld in enumerate(parser.camtoworlds):
        data["extrinsics"][f"image_{idx}"] = {
            "camtoworld_matrix": camtoworld.tolist()  # Full camera-to-world matrix
        }

    # Write the data to a JSON file
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Saved camera parameters to {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory containing COLMAP output")
    parser.add_argument("--factor", type=int, default=1, help="Downsampling factor for images")
    parser.add_argument("--save_path", type=str, default="camera_params.json", help="Path to save the JSON file")
    args = parser.parse_args()

    # Initialize the COLMAP parser
    colmap_parser = Parser(data_dir=args.data_dir, factor=args.factor, normalize=True)

    # Save camera parameters as a single JSON file with detailed labels
    save_camera_params_as_json(colmap_parser, args.save_path)