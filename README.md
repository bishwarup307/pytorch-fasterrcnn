# FasterRCNN (Pytorch)

Mostly taken from [Official PyTorch repo](https://github.com/pytorch/vision/tree/master/references/detection)

## Train DDP:

```sh
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22
```
