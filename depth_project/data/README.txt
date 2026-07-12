NYU Depth v2 Dataset
====================

Directory structure (required):
  data/
    train/
      rgb/     0001.png, 0002.png, ...   (RGB images, any common format)
      depth/   0001.png, 0002.png, ...   (16-bit PNG in millimetres, or metres)
    val/
      rgb/     0001.png, ...
      depth/   0001.png, ...

How to get data:
  Option A (recommended): HuggingFace
    python scripts/download_nyu_hf.py --out data

  Option B: Official .mat file
    Download nyu_depth_v2_labeled.mat from:
      https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
    Then convert:
      python scripts/convert_nyu_mat.py --mat nyu_depth_v2_labeled.mat --out data

  Option C: Lab's own indoor depth data
    Place RGB+depth pairs in the directory structure above.
    Depth format: 16-bit PNG (mm or m), .npy, .tiff, or .exr.
    The loader auto-detects units (values > 100 treated as millimetres).

Standard NYU Depth v2 labeled subset: 1449 images (1200 train + 249 val).
Depth range: 0.5 to 10.0 metres (loader clips to this range).
