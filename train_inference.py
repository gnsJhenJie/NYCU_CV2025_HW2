#!/usr/bin/env python3
"""
使用 Faster R-CNN (fpn_v2) 完成數字偵測及整數識別的訓練與 inference 程式

訓練模式:
    python train_inference.py --mode train --data_path data --num_epochs 10 --log_dir logs

推論模式:
    python train_inference.py --mode inference --data_path data --model_path logs/fasterrcnn_epoch9.pth

注意：
1. 訓練/驗證 JSON 檔應符合 COCO 格式，包含 "images" 與 "annotations" 兩個欄位。
2. 測試時會依 test/ 資料夾中的影像檔進行推論，影像檔名稱需符合 "image_id.png" 格式（例如 "1.png", "2.png", ...）。
3. 推論結果會產生 pred.json (偵測結果) 與 pred.csv (Task2 整數組合結果)。
"""

import os
import json
import argparse
from PIL import Image

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import ResNet50_Weights

import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter

# 啟用 cuDNN benchmark（若影像尺寸固定，可加速運算）
torch.backends.cudnn.benchmark = True

# -------------------- 資料集定義 --------------------


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, json_file, transforms=None):
        """
        root: 影像所在資料夾 (例如 data/train)
        json_file: 標註檔（COCO 格式，包含 "images" 與 "annotations"）
        transforms: 影像轉換函式
        """
        self.root = root
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.imgs = data["images"]
        self.annotations = data.get("annotations", [])
        self.imgid_to_annotations = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.imgid_to_annotations.setdefault(img_id, []).append(ann)
        self.transforms = transforms

    def __getitem__(self, idx):
        img_info = self.imgs[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        # 讀取標註並將 bbox 從 [x, y, width, height] 轉為 [x, y, x+w, y+h]
        anns = self.imgid_to_annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels,
                  "image_id": torch.tensor([img_id])}
        return image, target

    def __len__(self):
        return len(self.imgs)

# -------------------- 影像轉換 --------------------
# 先在 PIL 階段做資料增強，再轉成 Tensor


def get_transform(train):
    transforms = []
    if train:
        # 增加 ColorJitter
        transforms.append(
            T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    # 最後轉為 tensor
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

# -------------------- 模型定義 --------------------


def get_model(num_classes):
    # 使用 fasterrcnn_resnet50_fpn_v2 並以 ResNet50_Weights.IMAGENET1K_V2 為 backbone 預訓練權重
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# -------------------- 評估 Metric --------------------


def evaluate_map(model, data_loader, device, threshold=0.5, gt_json_path=None):
    """
    使用 COCOeval 計算 mAP (Task 1)，不做 category_id 轉換
    """
    model.eval()
    all_predictions = []
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        with torch.no_grad():
            predictions = model(images)
        for target, pred in zip(targets, predictions):
            image_id = target["image_id"].item()
            boxes = pred["boxes"].cpu().numpy().tolist()
            scores = pred["scores"].cpu().numpy().tolist()
            labels = pred["labels"].cpu().numpy().tolist()
            valid_indices = [i for i, s in enumerate(scores) if s >= threshold]
            for i in valid_indices:
                box = boxes[i]
                x_min, y_min, x_max, y_max = box
                w = x_max - x_min
                h = y_max - y_min
                all_predictions.append({
                    "image_id": image_id,
                    "bbox": [x_min, y_min, w, h],
                    "score": scores[i],
                    "category_id": labels[i]
                })
    if gt_json_path is None:
        raise ValueError("請提供 ground truth JSON 檔路徑以進行 COCO 評估")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cocoGt = COCO(gt_json_path)
    if len(all_predictions) == 0:
        return 0.0
    cocoDt = cocoGt.loadRes(all_predictions)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    mAP = cocoEval.stats[0]
    return mAP


def evaluate_accuracy(model, dataset, device, threshold=0.5):
    """
    計算 Task 2 整體全數字辨識 Accuracy，
    ground truth 與模型預測皆將 category_id 減 1 後拼成數字字串
    """
    model.eval()
    correct = 0
    total = 0
    for i in range(len(dataset)):
        image, target = dataset[i]
        boxes = target["boxes"].tolist()
        labels = target["labels"].tolist()
        if len(boxes) == 0:
            gt_label = "-1"
        else:
            sorted_gt = sorted(zip(boxes, labels), key=lambda x: x[0][0])
            # 轉換：category_id - 1
            gt_label = "".join([str(l - 1) for _, l in sorted_gt])
        image = image.to(device)
        with torch.no_grad():
            prediction = model([image])[0]
        pred_boxes = prediction["boxes"].cpu().numpy().tolist()
        pred_scores = prediction["scores"].cpu().numpy().tolist()
        pred_labels = prediction["labels"].cpu().numpy().tolist()
        valid_indices = [i for i, s in enumerate(
            pred_scores) if s >= threshold]
        if len(valid_indices) == 0:
            pred_full = "-1"
        else:
            sorted_pred = sorted([(pred_boxes[i], pred_labels[i])
                                 for i in valid_indices], key=lambda x: x[0][0])
            pred_full = "".join([str(l - 1) for _, l in sorted_pred])
        if pred_full == gt_label:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

# -------------------- 訓練一個 epoch --------------------


def train_one_epoch(
        model, optimizer, data_loader, device, epoch, writer, global_step,
        warmup_scheduler=None, warmup_iters=500):
    model.train()
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 若尚處於 warmup 迭代內，則更新 warmup scheduler
        if warmup_scheduler is not None and global_step < warmup_iters:
            warmup_scheduler.step()

        writer.add_scalar("Loss/train", losses.item(), global_step)

        if i % 50 == 0:
            print(f"Epoch {epoch} | Iteration {i} | Loss: {losses.item():.4f}")
        global_step += 1
    return global_step

# -------------------- 主程式 --------------------


def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN 數字識別訓練與推論")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'inference'],
                        help="模式：train 或 inference")
    parser.add_argument('--data_path', type=str, default='data', help="資料根目錄")
    parser.add_argument(
        '--model_path', type=str, default='logs/fasterrcnn.pth',
        help="模型權重儲存/讀取路徑")
    parser.add_argument('--num_epochs', type=int,
                        default=10, help="訓練 epoch 數")
    parser.add_argument('--log_dir', type=str, default='logs',
                        help="TensorBoard 與模型權重儲存目錄")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("使用裝置：", device)
    num_classes = 11  # 10 個數字 + 背景

    if args.mode == 'train':
        writer = SummaryWriter(log_dir=args.log_dir)
        global_step = 0

        # 建立資料集
        train_dataset = COCODataset(
            root=os.path.join(args.data_path, 'train'),
            json_file=os.path.join(args.data_path, 'train.json'),
            transforms=get_transform(train=True)
        )
        valid_dataset = COCODataset(
            root=os.path.join(args.data_path, 'valid'),
            json_file=os.path.join(args.data_path, 'valid.json'),
            transforms=get_transform(train=False)
        )

        def collate_fn(batch):
            return tuple(zip(*batch))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, shuffle=True, num_workers=8,
            pin_memory=True, collate_fn=collate_fn)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=8, shuffle=False, num_workers=8,
            pin_memory=True, collate_fn=collate_fn)

        model = get_model(num_classes)
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=0.006, momentum=0.88, weight_decay=0.0005)
        # 定義 warmup scheduler (迭代數小於 500 時線性增加學習率)
        from torch.optim.lr_scheduler import LambdaLR, StepLR
        warmup_iters = 500
        warmup_scheduler = LambdaLR(
            optimizer, lambda it: min(1, float(it + 1) / warmup_iters))
        # 定義 epoch scheduler，每 3 epoch 衰減學習率
        epoch_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

        print("開始訓練...")
        for epoch in range(args.num_epochs):
            global_step = train_one_epoch(
                model, optimizer, train_loader, device, epoch, writer,
                global_step, warmup_scheduler, warmup_iters)
            epoch_scheduler.step()  # 每個 epoch 後更新學習率
            # 評估驗證指標
            gt_json_path = os.path.join(args.data_path, 'valid.json')
            mAP_val = evaluate_map(
                model, valid_loader, device, threshold=0.5,
                gt_json_path=gt_json_path)
            accuracy_val = evaluate_accuracy(
                model, valid_dataset, device, threshold=0.5)
            # 注意：對於 Task2，這裡的預測與 ground truth 都將 category_id 減 1 後再組合字串
            writer.add_scalar("Metric/mAP", mAP_val, epoch)
            writer.add_scalar("Metric/Accuracy", accuracy_val, epoch)
            print(
                f"Epoch {epoch} 評估結果: mAP={mAP_val:.4f}，Task2 Accuracy={accuracy_val:.4f}")

            # 每個 epoch 存檔模型權重
            weight_path = os.path.join(
                args.log_dir, f"fasterrcnn_epoch{epoch}.pth")
            torch.save(model.state_dict(), weight_path)
            print(f"Epoch {epoch} 完成，模型權重已儲存至 {weight_path}")
        writer.close()
        print("訓練完成。")

    elif args.mode == 'inference':
        model = get_model(num_classes)
        model.to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()

        test_dir = os.path.join(args.data_path, 'test')
        image_files = sorted(
            [f for f in os.listdir(test_dir) if f.endswith('.png')],
            key=lambda x: int(os.path.splitext(x)[0]))
        pred_results = []
        pred_labels = []
        transform = get_transform(train=False)
        print("開始推論...")
        for img_file in image_files:
            image_id = int(os.path.splitext(img_file)[0])
            img_path = os.path.join(test_dir, img_file)
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).to(device)
            with torch.no_grad():
                prediction = model([image_tensor])[0]
            boxes = prediction['boxes'].cpu().numpy().tolist()
            scores = prediction['scores'].cpu().numpy().tolist()
            labels = prediction['labels'].cpu().numpy().tolist()
            threshold = 0.5
            valid_indices = [i for i, score in enumerate(
                scores) if score >= threshold]
            if len(valid_indices) == 0:
                pred_label = -1
            else:
                valid_dets = [(boxes[i], labels[i]) for i in valid_indices]
                valid_dets.sort(key=lambda x: x[0][0])
                # Task2 輸出時，將 category_id 減 1 對應正確數字
                pred_label = "".join([str(l - 1) for _, l in valid_dets])
            for i in valid_indices:
                box = boxes[i]
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                pred_results.append({
                    "image_id": image_id,
                    "bbox": [x_min, y_min, width, height],
                    "score": scores[i],
                    "category_id": labels[i]
                })
            pred_labels.append(
                {"image_id": image_id, "pred_label": pred_label})
        with open("pred.json", "w") as f:
            json.dump(pred_results, f)
        try:
            import pandas as pd
            pd.DataFrame(pred_labels).to_csv("pred.csv", index=False)
        except ImportError:
            import csv
            with open("pred.csv", "w", newline="") as csvfile:
                fieldnames = ["image_id", "pred_label"]
                writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer_csv.writeheader()
                for row in pred_labels:
                    writer_csv.writerow(row)
        print("推論完成，預測結果已儲存為 pred.json 與 pred.csv")


if __name__ == "__main__":
    main()
