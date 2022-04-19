'''
★ MMDetcection Custom Dataset 생성 주요 로직 ★

0. DAtaset을 위한 Config 설정 (data_root, ann_file, img_prefix) 
  -> Src data에서 Middle Format으로 변경하기 위한 중요한 파라미너 셋
1. CustomDataset 객체를 MMDetection Framework에 등록
  -> @Dataset.register_module
2. Config에 설정된 주요 값으로 CustomDataset 객체 생성 
  -> 우리가 생성하지 않고 MMDetection Framework이 호출함
  -> 요청하는 작업(build-dataset)을 호출하면 1,2번 작업

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
★ data_root, ann_file, img_prefix의 활용 ★

MM Dataset은 하습용, 검증용, 테스트용으로 각각 만들어질 수 있어야 한다.
Src Data가 MMDataset으로 쓰일려면 변환이 필요하다.
Src Dataset의 디렉토리구조(Formatting)에 기반을 해서 어떻게 Tr와 Val와 Test용 Dataset으로 변환을 할 수 있을까를 주요하게 결정하는 3개의 파라미터.
img_prefix는 여러 개의 image들을 포함할 수 있는 디렉토리 형태로 지정되지만, ann_file은 단 하나만(일반적으로 file) 지정할 수 있음
--> pascal voc같은 경우 tr용 ann, img를 가지는 list를 가지는 meta file을 만들어서 그걸 ann_file로 지정. 그리고 파싱

train.data_root = /content/kitti_tiny
train.ann_file = 'train.txt'
train.img_prefix = 'training/image_2

train_dataset = KIttiTinyDataset(ann_file = train.ann_file, 
                                 data_root = train.data_root,
                                 img_prefix = train.img_prefix)


< 소스 데이터들의 학습용, 검증용, 테스트용 분리 유형 >
- image들과 annotation 파일이 학습용, 검증용, 테스트용 디렉토리에 별도로 분리
- 별도의 메타 파일에서 학습용, 검증용, 테스트용 image들과 annotation 파일을 지정.
- image들은 학습용, 검증용, 테스트용 디렉토리 별로 분리, annotation은 학습용, 검증용, 테스트용으로 하나만 가짐
>> 이 3가지 src 데이터 유형을 다 감안해서 MMDataset으로 집어넣다보니깐 data_root, ann_file, img_prefix 3개 파라미터가 필요 -> 이걸 가지고 src data의 다양한 유형을 커버
'''

@DATASETS.register_module() # ★★ Dataset 객체를 Config에 등록 ★★
class CustomDataset(Dataset):
  CLASSES = None
  
  # ★★ Config로부터 객체 생성 시 인자가 입력됨. (MMDetection Framework이 자체적으로 호출)  ★★
  def __init__(self, ann_file, pipeline, classes=None, data_root=None, img_prefix=",,):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.CLASSES = self.get_classes(classes)
               
        '''
        osp.join : os.path.join
        self.data_root : '/content/kittitiny'
        self.ann_file : 'train.txt'
        '''
        self.ann_file = osp.join(self.data_root, self.ann_file) # 절대 경로 만듬.       
        self.img_prefix = osp.join(self.data_root, self.img_prefix) # self.data_root : annfile 파일의 상위 디렉토리        
        ......
        self.pipeline = Compose(pipeline)
               
   
   # ★★ Middle Format으로 변환하는 부분. 직접 Customize 코드를 작성해야 함. ★★
   def load_annotations(self, ann_file): #  여기서 ann_file은 바로 위에서 정의한 ann_file
               
       return mmmcv.load(ann_file)
       

'''
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CustomDataset을 "상속"받아 Dataset 생성
Class를 생성하고 def __init__()이 없는 이유 
--> CustomDatset의 인자를 ( __init__() 초기화)를 그대로 사용하겠다는 의미
-->  https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/custom.py의 def__init__이 그대로 들어옴. 
'''

import copy
import os.path as osp
import cv2

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/custom.py 여기에 있는 CustomDataset 클래스
from mmdet.datasets.custom import CustomDataset
 
# 반드시 아래 Decorator 설정 할것.
# @DATASETS.register_module() 설정 시 force=True를 입력하지 않으면 Dataset 재등록 불가. (나중에 수정이 불가) 
@DATASETS.register_module(force=True) # Decorator를 써서 MMDetection Framework에 등록

class KittyTinyDataset(CustomDataset): # CustomDataset 상속받음
  CLASSES = ('Car', 'Truck', 'Pedestrian', 'Cyclist') # CLASSES(대문자) 이대로 적고 클래스 명을 적으면 자동적으로 id 적용함  / 리스트로 해도 됨
  
  '''
  def __init__()이 없는 이유
  : CustomDatset의 인자를 ( __init__() 초기화)를 그대로 사용하겠다는 의미

  https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/custom.py
  의 def__init__이 그대로 들어옴. 
  '''

  ##### self.data_root: /content/kitti_tiny/ self.ann_file: /content/kitti_tiny/train.txt self.img_prefix: /content/kitti_tiny/training/image_2
  # Concat된 경로
  #### ann_file: /content/kitti_tiny/train.txt
  # annotation에 대한 모든 파일명을 가지고 있는 텍스트 파일을 __init__(self, ann_file)로 입력 받고, 이 self.ann_file이 load_annotations()의 인자로 입력
  # ann_file을 받아서 Middle Format Dataset 형태로 만듦

  def load_annotations(self, ann_file): # 여기서 ann_file은 concat된 경로가 들어감
    print('##### self.data_root:', self.data_root, 'self.ann_file:', self.ann_file, 'self.img_prefix:', self.img_prefix)
    print('#### ann_file:', ann_file)

    cat2label = {k:i for i, k in enumerate(self.CLASSES)}
    image_list = mmcv.list_from_file(self.ann_file) # ann_file을 다 받아서 리스트를 만듬

    # 포맷 중립 데이터를 담을 list 객체
    data_infos = []
    
    ###  위에 있는 이미지 작업 과정
    for image_id in image_list: # 개별적으로 000000~ 부터 받아서 끝까지 루프

      # self.img_prefix: /content/kitti_tiny/training/image_2
      # 절대경로가 필요한 이유 : opencv imread를 통해서 이미지의 height, width 구함
      filename = '{0:}/{1:}.jpeg'.format(self.img_prefix, image_id) # 파일의 절대경로 / 중간의 width와 height를 구하기 위해


      # 원본 이미지의 너비, 높이를 image를 직접 로드하여 구함. 
      image = cv2.imread(filename)
      height, width = image.shape[:2]

      # 개별 image의 annotation 정보 저장용 Dict 생성. key값 filename 에는 image의 파일명만 들어감(디렉토리는 제외)
      # 이미지 하나는 하나의 data_info를 가지게 됨
      data_info = {'filename': str(image_id) + '.jpeg',
                   'width': width, 'height': height}

      # 개별 annotation이 있는 서브 디렉토리의 prefix 변환. 
      # annotation 정보는 training/label_2에서 가지고 있음
      label_prefix = self.img_prefix.replace('image_2', 'label_2')

      # 개별 annotation 파일을 1개 line 씩 읽어서 list 로드 
      lines = mmcv.list_from_file(osp.join(label_prefix, str(image_id)+'.txt')) # label_2/000000.txt의 절대 경로 etc

      # 전체 lines를 개별 line별 공백 레벨로 parsing 하여 다시 list로 저장. content는 list의 list형태임.
      # ann 정보는 numpy array로 저장되나 텍스트 처리나 데이터 가공이 list 가 편하므로 일차적으로 list로 변환 수행.   
      # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01 이렇게 되어 있는 것을 하나씩 추려내는 작업 training/label_2/
      content = [line.strip().split(' ') for line in lines]

      # 오브젝트의 클래스명은 bbox_names로 저장. 
      bbox_names = [x[0] for x in content] # 인덱스 첫번째 : class name

      # bbox 좌표를 저장
      bboxes = [ [float(info) for info in x[4:8]] for x in content]

      # 클래스명이 해당 사항이 없는 대상 Filtering out, 'DontCare'sms ignore로 별도 저장.
      gt_bboxes = []
      gt_labels = []
      gt_bboxes_ignore = []
      gt_labels_ignore = []

      #### 위 이미지에서 'ann' key를 만드는 작업 --> bbox를 만드는 작업
      ## training/label_2/000000xx.txt에 있는 정보들 loop
      # loop를 한번에 담기 위해서 생성

      for bbox_name, bbox in zip(bbox_names, bboxes):

        # 만약 bbox_name이 클래스명에 해당 되면, gt_bboxes와 gt_labels에 추가, 그렇지 않으면 gt_bboxes_ignore, gt_labels_ignore에 추가
        if bbox_name in cat2label: # filtering : 위에서 지정한 class name이 있는 지 확인
          gt_bboxes.append(bbox) # 리스트의 리스트

          # gt_labels에는 class id를 입력
          gt_labels.append(cat2label[bbox_name])
        else: # don't care (class에 포함되지 않는 것)은 여기에 집어넣기
          gt_bboxes_ignore.append(bbox)
          gt_labels_ignore.append(-1)
     
      # 개별 image별 annotation 정보를 가지는 Dict 생성. 해당 Dict의 value값은 모두 np.array임. 
      # 위의 것들을 한꺼번에 담는 anno를 만듬 ->   위에서 작업한 것들을 한 middle format의 ann을 만드는 중
      data_anno = {
          'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4), # 리스트의 개별 리스트인 gt_boxes를 np.array로 만들어버림
          'labels': np.array(gt_labels, dtype=np.long), # 1차원
          'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
          'labels_ignore': np.array(gt_labels_ignore, dtype=np.long)
      }
      # image에 대한 메타 정보를 가지는 data_info Dict에 'ann' key값으로 data_anno를 value로 저장. 
      data_info.update(ann=data_anno) # 위에서 만든 data_info dict에 ann이라는 키와 data_anno를 value로 추가함

      # 전체 annotation 파일들에 대한 정보를 가지는 data_infos에 data_info Dict를 추가
      data_infos.append(data_info)

    return data_infos # 리스트 객체

               
# train_ds = KittyTinyDataset(data_root='/content/data', ann_file='train.txt', img_prefix='images')
# print(train_ds.data_infos[:10])
