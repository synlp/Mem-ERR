import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from memerr.corpus import read_annotations, record_images


COLORS = np.array(
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 165, 0],
        [0, 255, 255],
        [0, 0, 139],
        [75, 0, 130],
        [0, 128, 128],
        [128, 128, 128],
    ],
    dtype=np.float32,
)
GROUPS = (0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9)


def overlay(image, masks, threshold, alpha):
    if masks.shape[0] != len(GROUPS):
        raise ValueError("PSPNet must return fourteen anatomical masks")
    width, height = image.size
    side = min(width, height)
    left, top = width // 2 - side // 2, height // 2 - side // 2
    cropped = image.crop((left, top, left + side, top + side))
    base = np.asarray(cropped.convert("RGB").resize((masks.shape[-1], masks.shape[-2]))).astype(np.float32)
    result = base.copy()
    for index, group in enumerate(GROUPS):
        active = masks[index] >= threshold
        result[active] = result[active] * (1.0 - alpha) + COLORS[group] * alpha
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or not 0 <= args.alpha <= 1:
        raise ValueError("threshold and alpha must be in [0,1]")

    import torchxrayvision as xrv
    import torchvision

    annotations = read_annotations(args.annotations)
    image_paths = sorted(
        {
            path
            for split in ("train", "val", "test")
            for record in annotations[split]
            for path in record_images(record)
        }
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    transform = torchvision.transforms.Compose((xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)))
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    for relative_path in image_paths:
        with Image.open(image_root / relative_path) as source:
            rgb = source.convert("RGB")
            image = np.asarray(rgb)
        normalized = xrv.datasets.normalize(image, 255).mean(2)[None]
        tensor = torch.from_numpy(transform(normalized)).unsqueeze(0).to(device)
        with torch.no_grad():
            masks = torch.sigmoid(model(tensor))[0].cpu().numpy()
        output = output_root / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        overlay(rgb, masks, args.threshold, args.alpha).save(output)


if __name__ == "__main__":
    main()
