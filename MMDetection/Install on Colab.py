import torch
print(torch.__version__) # 1.10.0+cu111

# https://mmcv.readthedocs.io/en/latest/get_started/installation.html 설치 과정 참조.  
!pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
  
!git clone https://github.com/open-mmlab/mmdetection.git
!cd mmdetection; python setup.py install


# 아래를 수행하기 전에 kernel을 restart(런타임 다시 시작) 해야 함.
from mmdet.apis import init_detector, inference_detector
import mmcv
