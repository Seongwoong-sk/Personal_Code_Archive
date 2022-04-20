'''
PASCAL VOC형태의 BCCD Dataset를 Download 후 MS-COCO 형태로 변경
BCCD Dataset은 백혈구(WBC), 적혈구(RBC), 혈소판(Platelets) 세가지 유형의 Object Class를 가짐.
다운로드 받은 Dataset은 Pascal VOC 형태이므로 이를 별도의 유틸리티를 이용하여 MS-COCO 형태로 변환

BCCD_Dataset
 - BCCD
   - Annotations
     -- BloodImage_*.xml
   - ImageSets
     -- Main
       -- test.txt
       -- train.txt
       -- trainval.txt
       -- val.txt
   - JPEGImages
     -- BloodImage_*.jpg
'''

# BCCD DAataset
!git clone https://github.com/Shenggan/BCCD_Dataset.git

# VOC를 COCO로 변환하는 package적용하기
!git clone https://github.com/yukkyo/voc2coco.git
    
'''
1. Make labels.txt
Label1
Label2
...

2. Run script

$ python voc2coco.py \
    --ann_dir /path/to/annotation/dir \  --> anno dir
    --ann_ids /path/to/annotations/ids/list.txt \ --> train/val/test id를 가지고 있는 것
    --labels /path/to/labels.txt \  --> 방금 위에서 만든 것
    --output /path/to/output.json \ --> 출력은 json 파일로 됨
    <option> --ext xml
'''


## ★★ 1. Make labels.txt ★★
import os

# colab 버전은 아래 명령어로 ballnfish_classes.txt 를 수정합니다. 
with open('/content/BCCD_Dataset/BCCD/labels.txt', "w") as f:
    f.write("WBC\n")
    f.write("RBC\n")
    f.write("Platelets\n")

!cat /content/BCCD_Dataset/BCCD/labels.txt
'''
WBC
RBC
Platelets
'''

## ★★ 2. Run Script ★★

# VOC를 COCO로 변환 수행. 학습/검증/테스트 용 json annotation을 생성. 
%cd voc2coco
!python voc2coco.py --ann_dir /content/BCCD_Dataset/BCCD/Annotations \
--ann_ids /content/BCCD_Dataset/BCCD/ImageSets/Main/train.txt \
--labels /content/BCCD_Dataset/BCCD/labels.txt \
--output /content/BCCD_Dataset/BCCD/train.json \
--ext xml

!python voc2coco.py --ann_dir /content/BCCD_Dataset/BCCD/Annotations \
--ann_ids /content/BCCD_Dataset/BCCD/ImageSets/Main/val.txt \
--labels /content/BCCD_Dataset/BCCD/labels.txt \
--output /content/BCCD_Dataset/BCCD/val.json \
--ext xml

!python voc2coco.py --ann_dir /content/BCCD_Dataset/BCCD/Annotations \
--ann_ids /content/BCCD_Dataset/BCCD/ImageSets/Main/test.txt \
--labels /content/BCCD_Dataset/BCCD/labels.txt \
--output /content/BCCD_Dataset/BCCD/test.json \
--ext xml


# !cat /content/BCCD_Dataset/BCCD/train.json 으로 하면 한 줄로 나와서 보기 좋게 

# annotation json 파일을 잘 볼수 있는 <jq 유틸리티 셋업>. 
!sudo apt-get install jq

!jq . /content/BCCD_Dataset/BCCD/train.json > output.json  # indent를 이쁘게 파싱을 잘해서 output.json에 넣어라.
!head -100 output.json
'''
{
  "images": [
    {
      "file_name": "BloodImage_00001.jpg",  --> 여기서 파일네임은 절대경로 XX. 여기선 진짜 파일 이름만 나옴.
      "height": 480,
      "width": 640,
      "id": "BloodImage_00001"
    },
    {
      "file_name": "BloodImage_00003.jpg",
      "height": 480,
      "width": 640,
      "id": "BloodImage_00003"
    },
    {
      "file_name": "BloodImage_00004.jpg",
      "height": 480,
      "width": 640,
      "id": "BloodImage_00004"
    },
    .....
}
'''

# 뒷부분
!jq . /content/BCCD_Dataset/BCCD/train.json > output.json
!tail -100 output.json
'''
 "categories": [
    {
      "supercategory": "none",
      "id": 1,
      "name": "WBC"
    },
    {
      "supercategory": "none",
      "id": 2,
      "name": "RBC"
    },
    {
      "supercategory": "none",
      "id": 3,
      "name": "Platelets"
    }
  ]
}

여기서 보면 class에 id가 할당되어 있지만 MMDetection에선 이 class id를 안쓰고,
Class BCCDDataset(CocoDataset):
  CLASSES = ('WBC','RBC','Platelets') 이렇게 정한 클래스에서 자동적으로 class id를 정함.
'''



##########################################################################################################################
################################################################################################################
# Training with Voc2coco 

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

%cd /content

@DATASETS.register_module(force=True)
class BCCDDataset(CocoDataset):
  CLASSES = ('WBC', 'RBC', 'Platelets')  # 단일 클래스일 때는 CLASSES = ('WBC',) 2차원이라고 표시를 해줘야함.
  
config_file = '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

!cd /content/mmdetection; mkdir checkpoints
!wget -O /content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

from mmcv import Config

cfg = Config.fromfile(config_file)
print(cfg.pretty_text)


# Config 수정
from mmdet.apis import set_random_seed

# dataset에 대한 환경 파라미터 수정. 
cfg.dataset_type = 'BCCDDataset'
cfg.data_root = '/content/BCCD_Dataset/BCCD/'

# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
cfg.data.train.type = 'BCCDDataset'
cfg.data.train.data_root = '/content/BCCD_Dataset/BCCD/'
cfg.data.train.ann_file = 'train.json'  # 앞에서는 meta file이였고 여기선 진짜 Annotation 파일
cfg.data.train.img_prefix = 'JPEGImages'

cfg.data.val.type = 'BCCDDataset'
cfg.data.val.data_root = '/content/BCCD_Dataset/BCCD/'
cfg.data.val.ann_file = 'val.json'
cfg.data.val.img_prefix = 'JPEGImages'

cfg.data.test.type = 'BCCDDataset'
cfg.data.test.data_root = '/content/BCCD_Dataset/BCCD/'
cfg.data.test.ann_file = 'test.json'
cfg.data.test.img_prefix = 'JPEGImages'

# class의 갯수 수정. 
cfg.model.roi_head.bbox_head.num_classes = 3

# pretrained 모델
cfg.load_from = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 
cfg.work_dir = './tutorial_exps'

# 학습율 변경 환경 파라미터 설정. 
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# ★★ CocoDataset의 경우 metric을 bbox로 설정해야 함.(mAP아님. bbox로 설정하면 mAP를 iou threshold를 0.5 ~ 0.95까지 변경하면서 측정) ★★
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 12
cfg.checkpoint_config.interval = 12

# 두번 config를 로드하면 lr_config의 policy가 사라지는 오류로 인하여 설정. 
cfg.lr_config.policy='step'

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)




### Dataset을 만들고, 모델 학습 및 Inference 적용 ###
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

# train용 Dataset 생성. 
datasets = [build_dataset(cfg.data.train)]

print(datasets[0])
# datasets[0].__dict__ 로 모든 self variables의 key와 value값을 볼 수 있음. 
datasets[0].__dict__.keys()

datasets[0].data_infos

datasets[0].pipeline

# 빈 모델을 만드는 느낌 build_detector이지만 Pretrained 된 모델
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES
print(model.CLASSES)

## 학습
import os.path as osp
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

# epochs는 config의 runner 파라미터로 지정됨. 기본 12회
# 결과창에서 Evaluation은 val data로 진행됨.
train_detector(model, datasets, cfg, distributed=False, validate=True)




##########################################################################################################################
################################################################################################################
# Inference with Voc2coco

import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

img = cv2.imread('/content/BCCD_Dataset/BCCD/JPEGImages/BloodImage_00007.jpg')

model.cfg = cfg

result = inference_detector(model, img)
show_result_pyplot(model, img, result)

##########################################################################################################################
################################################################################################################
# Test data Inference

# config 파일 생성
cfg.dump('/content/tutorial_exps/bccd_faster_rcnn_conf.py')

!mkdir -p /content/show_test_output

# tools/test.py 는 colab에서 제대로 동작하지 않음. 
%cd /content/mmdetection
!python tools/test.py /content/tutorial_exps/bccd_faster_rcnn_conf.py /content/tutorial_exps/epoch_12.pth \
--eval 'bbox' \ # coco dataset이라서 mAP가 bbox별로 나옴.
--show-dir /content/show_test_output

########################### 코랩 메모리상 버그로 위 test.py 코드가 오류 발생 ######################################################


################### 대안 ############################

# --> 테스트용 dataset와 dataloader를 별도로 설정하고 trained된 checkpoint 모델을 로딩하여 test 수행.
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)

# test용 Dataset과 DataLoader 생성. 
# build_dataset()호출 시 list로 감싸지 않는 것이 train용 dataset 생성시와 차이. 
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        # 반드시 아래 samples_per_gpu 인자값은 1로 설정
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

# 반드시 아래 코드에서 'img' 키값이 tensor로 출력되어야 함. 
# 1건에 대한 tensor, sample per gpu (batch)를 1로 설정했기 때문
next(iter(data_loader))


# 모델 불러오기
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

checkpoint_file = '/content/tutorial_exps/epoch_12.pth'

# checkpoint 저장된 model 파일을 이용하여 모델을 생성, 이때 Config는 위에서 update된 config 사용. 
model_ckpt = init_detector(cfg, checkpoint_file, device='cuda:0')

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# 병렬처리하기 위해 적용 / test.py에 있음.
model_ckpt = MMDataParallel(model_ckpt, device_ids=[0])

# single_gpu_test() 를 호출하여 test데이터 세트의 interence 수행. 반드시 batch size는 1이 되어야 함. 
# 위에서 만든 /content/show_test_output 디렉토리에 interence 결과가 시각화된 이미지가 저장됨. 
outputs = single_gpu_test(model_ckpt, data_loader, True, '/content/show_test_output', 0.3)


### 반환된 test용 데이터세트의 inference 적용 결과 확인 및 성능 evaluation 수행 ###
print('결과 outputs type:', type(outputs))
print('evalution 된 파일의 갯수:', len(outputs))
print('첫번째 evalutation 결과의 type:', type(outputs[0]))
print('첫번째 evaluation 결과의 CLASS 갯수:', len(outputs[0]))
print('첫번째 evaluation 결과의 CLASS ID 0의 type과 shape', type(outputs[0][0]), outputs[0][0].shape)
'''
결과 outputs type: <class 'list'>
evalution 된 파일의 갯수: 72
첫번째 evalutation 결과의 type: <class 'list'>
첫번째 evaluation 결과의 CLASS 갯수: 3
첫번째 evaluation 결과의 CLASS ID 0의 type과 shape <class 'numpy.ndarray'> (1, 5) :: 1은 detect된 obj개수 / 5는 좌표 4개 + confidence score
'''

metric = dataset.evaluate(outputs, metric='bbox')
print(metric)
'''
Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.58s).
Accumulating evaluation results...
DONE (t=0.07s).

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.633
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.925
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.747
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.615
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.512
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.476
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.524

OrderedDict([('bbox_mAP', 0.633), ('bbox_mAP_50', 0.925), ('bbox_mAP_75', 0.747), ('bbox_mAP_s', 0.615), ('bbox_mAP_m', 0.512), ('bbox_mAP_l', 0.476), ('bbox_mAP_copypaste', '0.633 0.925 0.747 0.615 0.512 0.476')])
'''


