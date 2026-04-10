# 2D Object Detection for Road Perception

1 - Dataset
The dataset used for this project is the [KITTI Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d). It contains images captured from a car driving around a city, with annotations for various objects.

Download: "left color images of object data set" from the link above, along with "training labels of object data set". Data is organized in folders: `Data/training/image_2` for images and `Data/training/label_2` for labels.

The dataset preparation script `prepare_kitti_to_yolo.py` converts the KITTI dataset into a format suitable for training YOLO models. It also allows filtering by difficulty level (easy, moderate, hard) based on the KITTI annotations. You can specify the difficulty level using the `--difficulty` argument.
- Easy: Objects that are fully visible, with a bounding box height greater than 40 pixels, occlusion less than 0, and a truncation level less than 0.15.
- Moderate: Objects that are partially visible, with a bounding box height greater than 25 pixels, occlusion less than 1, and a truncation level less than 0.3.
- Hard: Objects that are heavily occluded or truncated, with a bounding box height greater than 25 pixels, occlusion less than 2, and a truncation level less than 0.5.

Execute the following command to prepare the dataset:

```bash
python prepare_kitti_to_yolo.py \
  --images KIITIData/data_object_image_2/training/image_2 \
  --labels KIITIData/data_object_label_2/training/label_2 \
  --output KIITIData/yolo_dataset_easy \
  --classes Car Pedestrian Cyclist \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1
  --difficulty easy

```

2 - Training (with Data Augmentation)

Install dependency:

```bash
pip install ultralytics
```

Train with default augmentation settings from `train_yolo.py`. Note that the default settings already include some augmentations, such as mosaic and mixup:

```bash
python train_yolo.py \
  --data KIITIData/yolo_dataset_easy/dataset.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640
```

Example with stronger augmentation:

```bash
python train_yolo.py \
  --data KIITIData/yolo_dataset_easy/dataset.yaml \
  --model yolov26s.pt \
  --epochs 120 \
  --batch 16 \
  --imgsz 640 \
  --degrees 8 \
  --translate 0.12 \
  --scale 0.6 \
  --shear 2.5 \
  --fliplr 0.5 \
  --mosaic 1.0 \
  --mixup 0.15
```
