# Mahjong Dataset
Computer Vision Dataset for Chinese Mahjong Tiles
<p float="left">
  <img src="https://i.imgur.com/vyO4M5C.jpg" width="50" />
  <img src="https://i.imgur.com/ges1OYy.jpg" width="50" />
  <img src="https://i.imgur.com/ZUdhHhM.png" width="50" />
  <img src="https://i.imgur.com/VXZyUFn.png" width="50" />
  <img src="https://i.imgur.com/ey89Qkp.jpg" width="50" />
  <img src="https://i.imgur.com/8xnwIOW.jpg" width="50" /> 
  <img src="https://i.imgur.com/X6z6CHN.jpg" width="50" />
</p>

## Structures
- Raw, unsliced images are located under `./raw-images`
- Unscaled tile images are located under `./raw-tiles`
- Scaled, ready-to-use images are located under `./tiles-resized`
- Csv file containing tagged labels of the images, and the csv template file are located under `./tiles-data`
- Raw, unsliced images for untagged tiles are located under `./untagged-images-raw`
- Untagged tile images are located under `./untagged-tiles`

## Data
- Images are mostly scraped from Google image search, Ebay and Alibaba
- Scaled images have the dimension `240(W) x 320(H)`. Format used are `.jpg`

**Ready to use data bundle can be located at the root of the repo, namely `train.zip`. It contains**
- `images` folder for mahjong tile images
- `data.csv` for image labels
- Tagged labels are stored in csv with the following format:

| image-name | label | label-name |
| ---------- | ----- | ---------- |
|   1.jpg    |  38   | bonus-winter |
| ... | ... | ... |

`label` is the index of the respective class of the image, see the classes below

## Classes
| table-name | table-index |
| --- | --- |
| dots-1 |	1 |
| dots-2 |	2 |
| dots-3 |	3 |
| dots-4 |	4 |
| dots-5 |	5 |
| dots-6 |	6 |
| dots-7 |	7 |
| dots-8 |	8 |
| dots-9 |	9 |
| bamboo-1 |	10 |
| bamboo-2 |	11 |
| bamboo-3 |	12 |
| bamboo-4 |	13 |
| bamboo-5 |	14 |
| bamboo-6 |	15 |
| bamboo-7 |	16 |
| bamboo-8 |	17 |
| bamboo-9 |	18 |
| characters-1 |	19 |
| characters-2 |	20 |
| characters-3 |	21 |
| characters-4 |	22 |
| characters-5 |	23 |
| characters-6 |	24 |
| characters-7 |	25 |
| characters-8 |	26 |
| characters-9 |	27 |
| honors-east |	28 |
| honors-south |	29 |
| honors-west |	30 |
| honors-north |	31 |
| honors-red |	32 |
| honors-green |	33 |
| honors-white |	34 |
| bonus-spring |	35 |
| bonus-summer |	36 |
| bonus-autumn |	37 |
| bonus-winter |	38 |
| bonus-plum |	39 |
| bonus-orchid |	40 |
| bonus-chrysanthemum |	41 |
| bonus-bamboo |	42 |

Refer the classes of mahjong tiles [here](https://en.wikipedia.org/wiki/Mahjong_tiles)

Feel free contribute to this repo with your own data.

## Training with YOLO

This repository includes a script to convert the dataset into [YOLO](https://docs.ultralytics.com/) format for object-detection training.

### 1. Prepare the dataset

```bash
python prepare_yolo_dataset.py
```

This creates a `yolo_dataset/` directory with the following layout:

```
yolo_dataset/
    images/
        train/   *.jpg   (80 % of images)
        val/     *.jpg   (20 % of images)
    labels/
        train/   *.txt
        val/     *.txt
    dataset.yaml
```

Each label file contains one line in YOLO format:

```
<class_id> <cx> <cy> <width> <height>
```

Because every source image contains exactly one tile that fills the frame, the bounding box is set to the full image (`cx=0.5 cy=0.5 w=1.0 h=1.0`).  Class IDs are 0-indexed (label index from `data.csv` minus 1).

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--val-split` | `0.2` | Fraction of images used for validation |
| `--seed` | `42` | Random seed for reproducible splits |
| `--images-dir` | `tiles-resized` | Source image directory |
| `--csv` | `tiles-data/data.csv` | Label CSV file |
| `--output` | `yolo_dataset` | Output directory |

### 2. Train

Install [Ultralytics](https://docs.ultralytics.com/) and start training:

```bash
pip install ultralytics
yolo detect train data=yolo_dataset/dataset.yaml model=yolov8n.pt epochs=100 imgsz=320
```

A pre-built `dataset.yaml` template is also available at the root of this repository.

## License
Open sourced under MIT License
