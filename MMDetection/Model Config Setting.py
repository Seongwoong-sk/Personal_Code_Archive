### Config 설정하고 Pretrained 모델 다운로드 후 Training
'''
mmdetection/configs/_base_/models/ 에 dict형태로 모델에 대한 config가 들어있음.
mmdetection/configs/_base_/datasets 
mmdetection/configs/_base_/schedules : optimizer 등
mmdetection/configs/_base_/default_runtime.py : hook(callback)
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
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# config 수행 시마다 policy값이 없어지는 bug로 인하여 설정. 
cfg.lr_config.policy = 'step'

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'

# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12

# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

