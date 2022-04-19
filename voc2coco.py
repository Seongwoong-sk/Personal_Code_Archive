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
