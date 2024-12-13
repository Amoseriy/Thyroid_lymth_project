import json
import sys
import os
# 获取当前脚本所在的目录
current_dir = os.getcwd()
# 添加 detection 目录到 sys.path，以便解释器能够找到detection文件夹下的模块
sys.path.append(os.path.join(current_dir, 'detection'))

from torchvision.ops import masks_to_boxes
from torchvision.io import decode_image

import torch
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.transforms import v2 as T

from DeepLearning.detection import utils
from detection.engine import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter


from loguru import logger
if not os.path.exists("./logs"):
    os.makedirs("./logs")

logger.remove()  # 清除默认的日志记录器，避免重复记录
# 配置日志记录器，将日志输出到文件和控制台。
# sys.stderr 用于输出到控制台，即时显示，可交互，可彩色，临时性
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO", colorize=True)
logger.add("./logs/training.log", format="{time} {level} {message}", level="INFO", colorize=False)  # 文件不能彩色输出

writer = SummaryWriter('runs/ex_model')


class SegmentationToDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, transform=None):
        super().__init__()
        self.data_dict = data_dict
        self.transforms = transform
        self.image_path_list = []
        self.mask_path_list = []
        # 遍历字典，整理数据
        for img_path, mask_path in data_dict.items():
            self.image_path_list.append(img_path)
            self.mask_path_list.append(tuple(mask_path))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        # load images and masks

        img_path = self.image_path_list[idx]
        mask_path_tuple = self.mask_path_list[idx]  # 同一张图片可能有良性和恶性的mask，所以tuple的大小可能为1或者2
        # print(img_path)

        img = decode_image(img_path)  # 直接读取图片为tensor

        labels_list = []

        if len(mask_path_tuple) > 1:  # 如果图片的mask有两个
            masks_list = []
            for mask_path in mask_path_tuple:
                temp_mask = decode_image(mask_path)

                obj_ids = torch.unique(temp_mask)
                # first id is the background, so remove it
                obj_ids = obj_ids[1:]
                # print(f"obj_ids: {obj_ids}")
                num_objs = len(obj_ids)

                if "malignant" in mask_path:
                    # 创建一个新的 tensor 来存储更新后的 mask
                    updated_mask = temp_mask.clone()  # 克隆原始 mask 以避免修改原数据
                    # 对于每个 obj_id，找到其在 mask 中的位置，并将其值设置为 100 + obj_id
                    for obj_id in obj_ids:
                        updated_mask[temp_mask == obj_id] = 100 + obj_id  # 避免恶性的mask值和良性的冲突，直接加100
                    masks_list.append(updated_mask)
                else:
                    masks_list.append(temp_mask)

                labels = torch.ones((num_objs,), dtype=torch.int64) if "benign" in mask_path else torch.ones((num_objs,), dtype=torch.int64) + 1  # 良性为1，恶性为2，背景为0
                labels_list.append(labels)
            # print(masks_list[1])
            mask = torch.max(masks_list[0], masks_list[1])  # 合并两个mask，如果两个mask有重叠的部分，则取最大值
        else:   # 如果图片的mask只有一个
            mask_path = mask_path_tuple[0]
            mask = decode_image(mask_path_tuple[0])
            obj_ids = torch.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]
            num_objs = len(obj_ids)

            labels = torch.ones((num_objs,), dtype=torch.int64) if "benign" in mask_path else torch.ones((num_objs,), dtype=torch.int64) + 1  # 良性为1，恶性为2，背景为0
            labels_list.append(labels)

        # 如果只有一个mask，则直接返回，如果有两个mask，则变量mask为合并后的mask，labels_list为两个mask对应的标签
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # print(f"obj_ids_after: {obj_ids}")

        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)
        # print(labels_list)
        labels = torch.cat(labels_list)

        image_id = idx
        # 计算每个实例的面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)
        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "labels": labels,
            "masks": tv_tensors.Mask(masks),
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # 过滤掉面积为0的实例
        non_zero_index = target["area"] != 0  # 获取非0面积的索引
        keys_to_filter = ["boxes", "labels", "masks", "area", "iscrowd"]
        filtered_target = {key: (value[non_zero_index] if key in keys_to_filter else value) for key, value in target.items()}

        if self.transforms is not None:
            img, filtered_target = self.transforms(img, filtered_target)

        return img, filtered_target


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes):
    """
    修改了原始模型的box_predictor和mask_predictor的输出类别数，从默认的91修改为了num_classes。
    """
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    # 默认的分类器就是FasterRCNNPredictor，只不过输出类别数是91，这里修改为num_classes。
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    # 默认的分类器就是MaskRCNNPredictor，只不过输出类别数是91，这里修改为num_classes。
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


with open("img_mask_path.json", "r") as f:
    data_dict = json.load(f)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 3
# use our dataset and defined transformations
dataset = SegmentationToDetectionDataset(data_dict, get_transform(train=True))
dataset_test = SegmentationToDetectionDataset(data_dict, get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 1

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

print("That's it!")
