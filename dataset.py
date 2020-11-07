"""
__author__: bishwarup
created: Saturday, 7th November 2020 3:27:36 pm
"""

from PIL import Image
import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as trsf
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, image_dir, json_path, transforms=None, return_ids=False, nsr=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transform = transforms

        self.coco = COCO(json_path)
        self.image_ids = self._filter_empty_annotations()
        print(f"valid images: {len(self.image_ids)}")

        self.return_ids = return_ids
        self.nsr = nsr if nsr is not None else 1.0
        self.img_transforms = trsf.Compose([trsf.ToPILImage(), trsf.ToTensor()])
        self.load_classes()
        # self._obtain_weights()
        print(f"number of classes: {self.num_classes}")

    def _filter_empty_annotations(self):
        valid_image_ids = []
        for imgid in self.coco.getImgIds():
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[imgid]))
            if len(anns):
                valid_image_ids.append(imgid)
        return valid_image_ids

    def _obtain_weights(self):
        weights = []
        for imid in self.image_ids:
            anns = self.coco.getAnnIds([imid])
            if anns:
                weights.append(1)
            else:
                weights.append(self.nsr)
        self.weights = weights

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes) + 1] = c["id"]
            self.coco_labels_inverse[c["id"]] = len(self.classes) + 1
            self.classes[c["name"]] = len(self.classes) + 1

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
        # print(len(self.labels))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot, iscrowd = self.load_annotations(idx)
        sample = {"img": img, "annot": annot, "iscrowd": iscrowd}
        if self.transform:
            sample = self.transform(sample)
        #         return sample
        #         if self.return_ids:
        #             return sample, self.image_ids[idx]

        target = self._format_target(idx, sample)

        return self.img_transforms(sample["img"].astype(np.uint8)), target

    def _format_target(self, idx, sample):

        annots = sample["annot"]
        iscrowd = sample["iscrowd"]

        assert len(annots) == len(
            iscrowd
        ), f"annots and iscrowd should have same dimenstion, got {len(annots)} and {len(iscrowd)}"

        target = dict()
        bboxes = np.ascontiguousarray(annots[:, :4].copy())
        class_ids = annots[:, 4].copy()
        areas = (bboxes[:, 2] - bboxes[:, 0]) * ((bboxes[:, 3] - bboxes[:, 1]))

        target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(class_ids, dtype=torch.int64)
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        target["scale"] = sample.get("scale", None)
        target["offset_x"] = sample.get("offset_x", None)
        target["offset_y"] = sample.get("offset_y", None)

        return target

    def load_image(self, image_index, normalize=True):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.image_dir, image_info["file_name"])
        img = np.array(Image.open(path))
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))
        iscrowd = []

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations, []

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a["bbox"][2] < 1 or a["bbox"][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a["bbox"]
            annotation[0, 4] = self.coco_label_to_label(a["category_id"])
            annotations = np.append(annotations, annotation, axis=0)
            iscrowd.append(a["iscrowd"])

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations, iscrowd

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image["width"]) / float(image["height"])

    @property
    def num_classes(self):
        return len(self.labels)


def letterbox(image, expected_size, fill_value=0):
    ih, iw, _ = image.shape
    eh, ew = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # print(image)
    new_img = np.full((eh, ew, 3), fill_value, dtype=np.float32)
    # fill new image with the resized image and centered it

    offset_x, offset_y = (ew - nw) // 2, (eh - nh) // 2

    new_img[offset_y : offset_y + nh, offset_x : offset_x + nw, :] = image.copy()
    return new_img, scale, offset_x, offset_y


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, annots, iscrowd = sample["img"], sample["annot"], sample["iscrowd"]
        rsz_img, scale, offset_x, offset_y = letterbox(image, self.size)

        annots[:, :4] *= scale
        annots[:, 0] += offset_x
        annots[:, 1] += offset_y
        annots[:, 2] += offset_x
        annots[:, 3] += offset_y

        return {
            "img": rsz_img,
            "annot": annots,
            "iscrowd": iscrowd,
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
        }
