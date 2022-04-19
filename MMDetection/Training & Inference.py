### TRINING ###
################################################################################################################
from mmdet.datasets import build_dataset # Dataset을 비로소 만듦
'''
Build DAtaset이 수행하는 것
0. DAtaset을 위한 Config 설정 (data_root, ann_file, img_prefix)
1. CustomDataset 객체를 MMDetection Framework에 등록
2. Config에 설정된 주요 값으로 CustomDataset 객체 생성
'''

from mmdet.models import build_detector
from mmdet.apis import train_detector

# train용 Dataset 생성. build_dataset -> config 파일을 읽어서 만듦
# build_dataset : Cumstom Dataset 생성 주요 로직 수행
datasets = [build_dataset(cfg.data.train)]


model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES


# 주의, config에 pretrained 모델 지정이 상대 경로로 설정됨 cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# 아래와 같이 %cd mmdetection 지정 필요. 
%cd mmdetection 

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir)) # tutorials/

# epochs는 config의 runner 파라미터로 지정됨. 기본 12회 
# validate=True : val data를 알아서 찾아서 작업을 해줌
train_detector(model, datasets, cfg, distributed=False, validate=True)


### INFERENCE ###
################################################################################################################
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# BGR Image 사용 
img = cv2.imread('/content/kitti_tiny/training/image_2/000068.jpeg') # BGR

# 모델에 config 집어 넣기
model.cfg = cfg

result = inference_detector(model, img) # Inference하면서 BGR을 RGB로 만듬
show_result_pyplot(model, img, result)

