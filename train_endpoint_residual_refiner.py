"""Train a small local endpoint residual refiner from reviewed training labels only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PATCH_SIZE = 128
ANCHOR = np.asarray([32.0, 64.0], np.float32)
GUIDE = np.asarray([96.0, 64.0], np.float32)


def read_image(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tofile(str(path))


def load_points(path: Path, width: int, height: int) -> list[np.ndarray]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = np.asarray([float(value) for value in line.split()])
        result.append(values[5:].reshape(15, 3)[:, :2] * np.asarray([width, height]))
    return result


def endpoint_affine(guess: np.ndarray, inward: np.ndarray) -> np.ndarray | None:
    vector = inward - guess
    distance = float(np.linalg.norm(vector))
    if distance < 2:
        return None
    unit = vector / distance
    normal = np.asarray([-unit[1], unit[0]])
    scale = float(np.linalg.norm(GUIDE - ANCHOR)) / distance
    return np.asarray([[scale * unit[0], scale * unit[1], ANCHOR[0] - scale * np.dot(unit, guess)],
                       [scale * normal[0], scale * normal[1], ANCHOR[1] - scale * np.dot(normal, guess)]],
                      np.float32)


def endpoint_patch(image: np.ndarray, guess: np.ndarray, inward: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    affine = endpoint_affine(guess, inward)
    if affine is None:
        return None
    patch = cv2.warpAffine(image, affine, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    return patch, affine


def build_samples(dataset: Path, output: Path, variants: int, seed: int) -> Path:
    manifest = output / "manifest.csv"
    if manifest.exists():
        return manifest
    rng = np.random.default_rng(seed)
    rows = []
    for image_path in sorted((dataset / "images/train").iterdir()):
        image = read_image(image_path)
        height, width = image.shape[:2]
        points_all = load_points(dataset / "labels/train" / f"{image_path.stem}.txt", width, height)
        fold = int(hashlib.sha1(image_path.name.encode("utf-8")).hexdigest()[:8], 16) % 10
        subset = "holdout" if fold == 0 else "train"
        for instance, points in enumerate(points_all):
            for endpoint_type, endpoint_index, inward_index in ((0, 0, 2), (1, 14, 12)):
                endpoint, inward = points[endpoint_index], points[inward_index]
                base = inward - endpoint
                distance = float(np.linalg.norm(base))
                if distance < 4:
                    continue
                unit = base / distance; normal = np.asarray([-unit[1], unit[0]])
                for variant in range(variants):
                    # Most real errors stop inward; retain outward and lateral cases as safeguards.
                    tangent = rng.uniform(-0.25, 1.05) * distance
                    lateral = rng.normal(0, 0.20) * distance
                    guess = endpoint + tangent * unit + lateral * normal
                    noisy_inward = inward + rng.normal(0, 0.06) * distance * unit + rng.normal(0, 0.10) * distance * normal
                    extracted = endpoint_patch(image, guess, noisy_inward)
                    if extracted is None:
                        continue
                    patch, affine = extracted
                    target = cv2.transform(endpoint.reshape(1, 1, 2).astype(np.float32), affine)[0, 0]
                    if not (-16 <= target[0] < PATCH_SIZE + 16 and -16 <= target[1] < PATCH_SIZE + 16):
                        continue
                    relative = (target - ANCHOR) / 64.0
                    name = f"{image_path.stem}__i{instance:02d}_e{endpoint_type}_v{variant:02d}.jpg"
                    relative_path = Path("images") / subset / name
                    write_image(output / relative_path, patch)
                    rows.append({"image": relative_path.as_posix(), "subset": subset,
                                 "endpoint_type": endpoint_type, "dx": float(relative[0]),
                                 "dy": float(relative[1])})
    output.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (output / "build_summary.json").write_text(json.dumps({"samples": len(rows),
        "train": sum(r["subset"] == "train" for r in rows),
        "holdout": sum(r["subset"] == "holdout" for r in rows), "variants": variants,
        "source": str(dataset)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class EndpointDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], augment: bool):
        self.root, self.rows, self.augment = root, rows, augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = read_image(self.root / row["image"]).astype(np.float32) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augment:
            image = np.clip(image * random.uniform(0.82, 1.18) + random.uniform(-0.06, 0.06), 0, 1)
            if random.random() < 0.25:
                image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.1, 0.8))
        kind = np.full((PATCH_SIZE, PATCH_SIZE, 1), float(row["endpoint_type"]), np.float32)
        yy, xx = np.mgrid[-1:1:complex(PATCH_SIZE), -1:1:complex(PATCH_SIZE)]
        features = np.concatenate([image, kind, xx[..., None].astype(np.float32),
                                   yy[..., None].astype(np.float32)], axis=2)
        target = np.asarray([float(row["dx"]), float(row["dy"])], np.float32)
        return torch.from_numpy(features.transpose(2, 0, 1).copy()), torch.from_numpy(target)


class EndpointRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = [6, 32, 48, 72, 112, 160]
        for source, target in zip(channels, channels[1:]):
            layers += [nn.Conv2d(source, target, 3, 2, 1, bias=False), nn.BatchNorm2d(target), nn.SiLU(),
                       nn.Conv2d(target, target, 3, 1, 1, groups=target, bias=False),
                       nn.BatchNorm2d(target), nn.SiLU()]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(160 * 4 * 4, 256), nn.SiLU(),
                                  nn.Dropout(0.15), nn.Linear(256, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.features(value))) * 1.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    manifest = build_samples(args.dataset, args.output / "samples", args.variants, args.seed)
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train_rows = [r for r in rows if r["subset"] == "train"]
    holdout_rows = [r for r in rows if r["subset"] == "holdout"]
    train_loader = DataLoader(EndpointDataset(manifest.parent, train_rows, True), batch_size=args.batch,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    holdout_loader = DataLoader(EndpointDataset(manifest.parent, holdout_rows, False), batch_size=args.batch * 2,
                                shuffle=False, num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndpointRefiner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=3e-5)
    loss_fn = nn.SmoothL1Loss(beta=0.08)
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "weights").mkdir(exist_ok=True)
    history, best = [], math.inf
    for epoch in range(1, args.epochs + 1):
        model.train(); train_sum = train_count = 0
        for features, target in train_loader:
            features, target = features.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); prediction = model(features)
            loss = loss_fn(prediction, target); loss.backward(); optimizer.step()
            train_sum += float(loss) * len(features); train_count += len(features)
        model.eval(); val_sum = val_count = 0
        with torch.no_grad():
            for features, target in holdout_loader:
                features, target = features.to(device), target.to(device)
                loss = loss_fn(model(features), target)
                val_sum += float(loss) * len(features); val_count += len(features)
        scheduler.step(); val_loss = val_sum / max(val_count, 1)
        record = {"epoch": epoch, "train_loss": train_sum / train_count, "holdout_loss": val_loss,
                  "lr": optimizer.param_groups[0]["lr"]}; history.append(record); print(json.dumps(record), flush=True)
        payload = {"model": model.state_dict(), "epoch": epoch, "holdout_loss": val_loss}
        torch.save(payload, args.output / "weights/last.pt")
        if epoch % 5 == 0:
            torch.save(payload, args.output / f"weights/epoch{epoch}.pt")
        if val_loss < best:
            best = val_loss; torch.save(payload, args.output / "weights/best.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
