### Config 설정하고 Pretrained 모델 다운로드 후 Training
'''
mmdetections/'model' 에서 하위 4개의 config(of _base_)를 결합한 것이 하나의 config로 생성됨.
mmdetection/configs/_base_/models/ 에 dict형태로 모델에 대한 config가 들어있음.
mmdetection/configs/_base_/datasets 
mmdetection/configs/_base_/schedules : optimizer 등
mmdetection/configs/_base_/default_runtime.py : hook(callback)


★★★★ Config 대분류 ★★★★

1. Dataset
--> dataset의 type(CustomDataset[load_annotations method에서 CD를 파싱해서 MIddle format으로 변환], CocoDataset 등), train/val/test Dataset의 유형, data_root, train_val_test Dataset의 주요 파라미터 설정(type, ann_file, img_prefix, pipeline 등)

2. model
--> Customizing이 필수는 아님 (신중함 필요)
--> Object Detection Model의 backbone, neck, dense head, roi extractor, roi head(num classes 변경) 주요 영역별로 세부 설정

3. schedule
--> optimizer 유형 설정 (SGD, Adam, RMSprop 등), 최초 learning 설정
--> 학습 중 동적 learning rate 적용 정책 설정 (step, cyclic, CosineAnnealing 등)
--> train 시 epochs 횟수

4. run time
--> 주로 hook(callback)관련 설정
--> 학습 중 checkpoint 파일, log 파일 생성을 위한 interval epochs 수

'''

config_file = '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

!cd mmdetection; mkdir checkpoints
!wget -O /content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

  
from mmcv import Config

# fromfile : config_file을 cfg 객체로 로딩
# dictionary 형태
cfg = Config.fromfile(config_file) # '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
print(cfg.pretty_text)

### Config 설정-------------------------------------------------------------------------------------------------
###----------------------------------------------------------------------------------------------------------------

from mmdet.apis import set_random_seed

### Tip ###
# Data_root는 절대경로로 해주는 것이 좋음

# config는 data dict이다
# dataset에 대한 환경 파라미터 수정. 
cfg.dataset_type = 'KittyTinyDataset'
cfg.data_root = '/content/kitti_tiny/'

# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
cfg.data.train.type = 'KittyTinyDataset'
cfg.data.train.data_root = '/content/kitti_tiny/' # 맨 끝 /
cfg.data.train.ann_file = 'train.txt' # ann_file은 반드시 하나. 나중에 data_root하고 concat이 됨. -> / X
cfg.data.train.img_prefix = 'training/image_2' # 이것도 나중에 data_root하고 concat이 됨 -> 앞에 / 없어야됨  // data_root + img_prefix --> /content/kitti_tiny/training/image_2

# validation은 내부 자체적으로 찾음. framework에서 알아서 설정해버림.
cfg.data.val.type = 'KittyTinyDataset'
cfg.data.val.data_root = '/content/kitti_tiny/'
cfg.data.val.ann_file = 'val.txt'
cfg.data.val.img_prefix = 'training/image_2'

cfg.data.test.type = 'KittyTinyDataset'
cfg.data.test.data_root = '/content/kitti_tiny/'
cfg.data.test.ann_file = 'val.txt'
cfg.data.test.img_prefix = 'training/image_2'



# class의 갯수 수정. 
cfg.model.roi_head.bbox_head.num_classes = 4

# pretrained 모델 로딩
cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth' # 상대경로로 했을 때 유의해야 할 점 : 
'''
여기서 pretrained가 필요한 이유
-> 대부분 현대 딥러닝 모델은 pretrained 모델을 사전에 적용합니다. 

딥러닝은 weight를 최적화 하는 것이 핵심입니다.  weight를 최적화 하려면 대량의 데이터로 좋은 모델을 가지고 학습을 시키는게 중요합니다.
그런데 딥러닝은 대량의 데이터로 학습시 너무 시간이 오래 걸립니다. 
그래서 사전에 학습된 모델을 이용합니다. 사전에 학습된 모델은 어느정도 최적화된 weight를 가지고 있기 때문입니다.  
물론 사전 학습된 모델이 실제 학습된 모델과 완전히 다른 이미지로 학습 되어 있다면 사전 학습된 모델을 이용하는 효과가 줄겠지만, 그렇지 않다면 효과가 매우 좋습니다
'''

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 
# 현재 . : content
cfg.work_dir = './tutorial_exps'


##################### 
# 나중에 익숙해지면 Tuning을 위해 많이 수정하는 부분

# schedule_1x.py에서 확인
# 학습율 변경 환경 파라미터 설정. 

# The Original lr is set for 8-GPU Training
# We divide it by 8 since we only use one GPU
cfg.optimizer.lr = 0.02 / 8 # task가 8개의 gpu 했을 때 설정
cfg.lr_config.warmup = None
cfg.log_config.interval = 10
# cfg.runner.max_epochs = 5  # epoch를 5번만 돌릴 것임


# config 수행 시마다 policy값이 없어지는 bug로 인하여 설정. 
cfg.lr_config.policy = 'step'

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'

# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12

# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# 학습 시 Batch size 설정(단일 GPU 별 Batch size로 설정됨)
cfg.data.samples_per_gpu = 4 # 여기서 gpu가 2개면 4 x 2 = 8로 됨.

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')




'''
Config:
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
dataset_type = 'KittyTinyDataset'
data_root = '/content/kitti_tiny/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1500, 900), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1500, 900),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='KittyTinyDataset',
        ann_file='train.txt',
        img_prefix='training/image_2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1500, 900), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        data_root='/content/kitti_tiny/'),
    val=dict(
        type='KittyTinyDataset',
        ann_file='val.txt',
        img_prefix='training/image_2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1500, 900),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        data_root='/content/kitti_tiny/'),
    test=dict(
        type='KittyTinyDataset',
        ann_file='val.txt',
        img_prefix='training/image_2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        data_root='/content/kitti_tiny/'))
evaluation = dict(interval=12, metric='mAP')
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=12)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = './tutorial_exps'
seed = 0
gpu_ids = range(0, 1)


'''
