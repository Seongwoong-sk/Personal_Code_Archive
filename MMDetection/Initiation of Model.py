# Pretrained Model (Weight) Download
# https://github.com/open-mmlab/mmdetection/tree/master/configs
!wget -O /content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth


# config 파일을 설정하고, 다운로드 받은 pretrained 모델을 checkpoint로 설정. 
# config는 절대경로보다 상대경로로 표시하는 경우가 많음 
config_file = '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'


# config 파일과 pretrained 모델을 기반으로 Detector 모델을 생성. 
# init_detector가 pt모델을 가져오는 것(만드는 것)
# coco 80개 데이터셋으로 기반으로 학습이 된 모델이 생성됨
from mmdet.apis import init_detector, inference_detector
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# mmdetection은 상대 경로를 인자로 주면 무조건 mmdetection 디렉토리를 기준으로 함. 
# 상대경로로 쓸려면 %cd로 바꿔줌
%cd mmdetection
# 이 경로에 mmdetection이 생략되어 있음.
# 1x는 epoch를 12번했다는 뜻 / 2x : epoch 24번
from mmdet.apis import init_detector, inference_detector
model = init_detector(config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', checkpoint='checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')


'''
mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
--> base의 이것을 참고하세요.
--> mmdetection의 여러 가지 모델들이 각각의 config를 가지고 있는 게 아니라 공통적으로 가지고 있는 config들이 있음. 
--> 그럴 때 이렇게 base 지정
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
'''
