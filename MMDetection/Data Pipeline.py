'''
★★★ Data Pipeline 구성 ★★★
---------------------------------

◈◈ ↓↓↓↓ Source Images ↓↓↓↓ ◈◈

◈◈ < Data Loading > ◈◈

▶ LoadImageFromFile
{ 
  'img': (+)
  'img_shape': (+)
  'ori_shape' (+)
 }
 
 ▶ LoadAnnotaitons
 {
   'img':
   'img_shape':
   'ori_shape':
   'gt_bboxes': (+)
   'gt_labels' : (+)
   'bbox_fields': (+)
 }
 

◈◈ < Preprocessing > ◈◈ - 원본 이미지 pixel 값 변경
 

▶ Resize
{
   'img': (U)
   'img_shape': (U)
   'ori_shape':
   'pad_shape' : (+)
   'gt_bboxes' : (U)
   'gt_labels' :
   'bbox_fields' :
   'scale' : (+)
   'scale_idx' : (+)
   'scale_factor' : (+)
   'keep_ratio' : (+)
 }

▶ RandomFlip
{
   'img': (U)
   'img_shape': 
   'ori_shape':
   'pad_shape' : 
   'gt_bboxes' : (U)
   'gt_labels' :
   'bbox_fields' :
   'scale' : 
   'scale_idx' : 
   'scale_factor' : 
   'keep_ratio' : 
   'flip' : (+)
 }
 
▶ Normalize
 {
   'img': (U)
   'img_shape': 
   'ori_shape':
   'pad_shape' : 
   'gt_bboxes' : 
   'gt_labels' :
   'bbox_fields' :
   'scale' : 
   'scale_idx' : 
   'scale_factor' : 
   'keep_ratio' : 
   'flip' : 
   'img_norm_cfg' : (+)
 }
 
 
▶ Pad
{
   'img': (U)
   'img_shape': 
   'ori_shape':
   'pad_shape' : (U)
   'gt_bboxes' : 
   'gt_labels' :
   'bbox_fields' :
   'scale' : 
   'scale_idx' : 
   'scale_factor' : 
   'keep_ratio' : 
   'flip' : 
   'img_norm_cfg' : 
   'pad_fixed_size: (+)
   'pad_size_divisor' : (+)
  }
  
 
◈◈ < Formatting > ◈◈ -  Model에 넣기 전에 최종적으로 Filtering

▶ DefaultFormatBundle
{
   'img': (U)
   'img_shape': 
   'ori_shape':
   'pad_shape' : 
   'gt_bboxes' : (U)
   'gt_labels' : (U)
   'bbox_fields' :
   'scale' : 
   'scale_idx' : 
   'scale_factor' : 
   'keep_ratio' : 
   'flip' : 
   'img_norm_cfg' : 
   'pad_fixed_size: 
   'pad_size_divisor' : 
 }

▶ Collect
{
   'img': 
   'img_meta': { (+)
     'ori_shape':
     'img_shape':
     'pad_shape':
     'scale_factor':
     'flip':
     'img_norm_cfg':
   }
   'gt_bboxes':
   'gt_labels':
}


◈◈ ↓↓↓↓↓↓↓↓↓↓ ◈◈

◈◈    Model   ◈◈
'''

## Train Pipeline ##

img_norm_cfg = dict( # Normalization --> (C,H,W) - mean / std
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) # BGR을 RGB로 변환해주고 Network 진행, False로도 설정 가능
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1500, 900), keep_ratio=True), # img_scale = Model에 들어갈 크기/ keep_ratio=True : 비율 그대로 유지
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True), 
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']) # Model에 입력할 형식
]
 

## Test Pipeline ##
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
  
  
  
