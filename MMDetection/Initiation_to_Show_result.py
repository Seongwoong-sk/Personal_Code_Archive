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


# Inference

img = '/content/mmdetection/demo/demo.jpg'
# inference_detector의 인자로 string(file 절대경로), ndarray가 단일 또는 list형태로 입력 될 수 있음. 
results = inference_detector(model, img)

# type(results) : list
# len(results) : 80 --> coco datasets이라서 80개
type(results), len(results)

'''
result 해석
results는 list형으로 coco class의  0부터 79까지 class_id별로 80개의 array를 가짐. 
-> 개별 array들은 각 클래스별로 5개의 값(좌표값과 class별로 confidence)을 가짐. 개별 class별로 여러개의 좌표를 가지면 여러개의 array가 생성됨. 
-> 좌표는 좌상단(xmin, ymin), 우하단(xmax, ymax) 기준. 
-> 개별 array의 shape는 (Detection된 object들의 수, 5(좌표와 confidence)) 임 --> xmin, ymin, xmax, ymax, confidence (obj 1개에 대한)
-> class id는 자동으로 할당이 됨 (리스트안에 있는  array 인덱스에 따라)
'''


# 모델과 이미지,결과를 입력하면 자동으로 그림을 그려주는 유틸리티
from mmdet.apis import show_result_pyplot

# inference 된 결과를 원본 이미지에 적용하여 새로운 image로 생성(bbox 처리된 image)
# Default로 score threshold가  0.3 이상인 Object들만 시각화 적용. show_result_pyplot은 model.show_result()를 호출. 
show_result_pyplot(model, img, results)
