# Data

The dataset for this project is a 100-class slice of two things:

- **ImageNet-C**, the standard benchmark for image-corruption robustness
  (15 corruption types at 5 severity levels), and
- the **clean ImageNet validation images** for the same 100 classes.

Using 100 of ImageNet's 1000 classes keeps the dataset small and quick to work
with while still giving stable, trustworthy accuracy numbers.

The images are not stored in the repository. Four short steps download and
assemble everything. Run them once, from the repository root.

## Before you start

```bash
pip install -r requirements.txt
```

The scripts also need to see one folder that contains all 1000 ImageNet class
folders (named `n01440764`, `n01443537`, ...) so they can look up each class's
official index. Any ImageNet `train/` or `val/` directory works. The repository
already ships one such folder (`data/imagenet-c/gaussian_noise/1`) and the
scripts use it by default, so normally there is nothing to do here.

## Step 1 - choose the 100 classes

```bash
python scripts/build_subset.py
```

Picks 100 of the 1000 ImageNet classes uniformly at random (seed 0, no
stratification by category or difficulty), so everyone who runs this gets the
same 100, and writes them to `data/subset_classes.json`. Every other step reads
that file. It is committed to the repository, so you only need this step if you
want to change the selection.

## Step 2 - clean images

```bash
python scripts/fetch_clean_images.py
```

Downloads the ImageNet validation images for the 100 classes (50 per class,
5000 total) from a public copy of ImageNet on Hugging Face. No account needed.
Saved as:

```
data/imagenet/val/<class>/<image>.JPEG
```

## Step 3 - corrupted images

```bash
python scripts/fetch_imagenet_c.py
```

Downloads ImageNet-C for the same 100 classes: 15 corruption types, 5 severity
levels each, 50 images per class. Saved as:

```
data/imagenet-c/<corruption>/<severity>/<class>/<image>.JPEG
```

The 15 corruptions, by group:

| group   | corruptions |
|---------|-------------|
| noise   | gaussian_noise, shot_noise, impulse_noise |
| blur    | defocus_blur, glass_blur, motion_blur, zoom_blur |
| weather | snow, frost, fog, brightness |
| digital | contrast, elastic_transform, pixelate, jpeg_compression |

This step takes a while, because the source server is slow. It is safe to
interrupt and re-run: anything already downloaded is skipped. To speed it up,
download the four groups at the same time:

```bash
for g in digital blur weather noise; do
  python scripts/fetch_imagenet_c.py --tarballs $g &
done
```

Useful options:

| option | effect |
|--------|--------|
| `--tarballs noise blur` | only these groups |
| `--corruptions fog frost` | only these corruptions |
| `--severities 1 3 5` | only these severity levels |

## Step 4 - check it worked

```bash
python - <<'EOF'
import json
from pathlib import Path

subset = set(json.load(open("data/subset_classes.json"))["wnid_to_index"])
root = Path("data/imagenet-c")

problems = []
for corruption in sorted(p.name for p in root.iterdir() if p.is_dir()):
    for severity in "12345":
        d = root / corruption / severity
        classes = {p.name for p in d.iterdir() if p.is_dir()} if d.is_dir() else set()
        n_images = sum(1 for _ in d.rglob("*.JPEG"))
        if classes != subset or n_images != 5000:
            problems.append(f"{corruption}/{severity}: {len(classes)} classes, {n_images} images")

print("\n".join(problems) if problems
      else "imagenet-c OK: 15 corruptions x 5 severities, 100 classes, 5000 images each")

val = Path("data/imagenet/val")
print("clean val:", sum(1 for _ in val.rglob("*.JPEG")), "images,",
      sum(1 for p in val.iterdir() if p.is_dir()), "classes")
EOF
```

Expected:

```
imagenet-c OK: 15 corruptions x 5 severities, 100 classes, 5000 images each
clean val: 5000 images, 100 classes
```

For an end-to-end check, run the first experiment step on a small slice:

```bash
python scripts/cache_profiles.py --models resnet18 \
    --corruptions gaussian_noise defocus_blur fog contrast \
    --severities 3 --n-per-class 5
```

It prints `orig acc` per corruption. The plain ResNet-18 top-1 accuracy at
severity 3 should land near gaussian_noise 25%, defocus_blur 46%, fog 46%,
contrast 37%, matching the numbers in the ImageNet-C paper. (Delete the tiny
`cache/` it writes before the real run, or pass `--overwrite` later.)

## What you end up with

```
data/
  subset_classes.json                            the 100 chosen classes
  imagenet/val/<class>/                           5000 clean images
  imagenet-c/<corruption>/<severity>/<class>/     375,000 corrupted images
```

Only this file and `subset_classes.json` are kept in git. The images are meant
to be rebuilt with the steps above.

## Source

ImageNet-C comes from Hendrycks and Dietterich, "Benchmarking Neural Network
Robustness to Common Corruptions and Perturbations" (2019),
https://zenodo.org/records/2235448. The clean images are the standard
ImageNet-1k validation set.
