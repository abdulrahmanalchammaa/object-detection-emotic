# Object & Emotion Detection (EMOTIC)

Graduation project: object detection and per-image emotion/context categorization using the [EMOTIC dataset](http://sunai.uoc.edu/emotic/) and RetinaNet.

- **Built:** December 2022
- **Stack:** Python, ImageAI (RetinaNet), pandas, numpy, tqdm

> This repo contains the source code only. The EMOTIC dataset, the trained RetinaNet weights (`model.h5`), and generated output images are not included here due to their size — see below for how to obtain them.

## What it does
- `objectdetection.py` — loads the EMOTIC image set, runs RetinaNet object detection on each image, maps detected objects to emotional/context super-categories (`catgores.txt`), and writes a per-image summary to `images_csv.csv`.
- `TAKEPHHOTOS.py` — copies a selected subset of dataset images (listed in `data.csv`) into a working folder.
- `firstdetection.py` / `test.py` — early experiments/scratch scripts used while building the detection pipeline.
- `dataset_path.txt` / `all_dataset_path.json` — configurable paths to the dataset splits and output directory.

## Setup
1. Download the [EMOTIC dataset](http://sunai.uoc.edu/emotic/) and a RetinaNet model file (`model.h5`, e.g. from [ImageAI](https://github.com/OlafenwaMoses/ImageAI)).
2. Update the paths in `dataset_path.txt` and `all_dataset_path.json` to point to your local dataset location.
3. Install dependencies:
   ```bash
   pip install imageai pandas numpy tqdm
   ```
4. Run detection:
   ```bash
   python objectdetection.py
   ```
