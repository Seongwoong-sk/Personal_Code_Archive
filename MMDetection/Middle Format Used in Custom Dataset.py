#Support a new dataset
'''
There are three ways to support a new dataset in MMDetection:

1. Reorganize the dataset into a COCO format.
2. Reorganize the dataset into a middle format.
3. Implement a new dataset.
'''

# Custom Dataset에 사용되는 Middle Format #
'''
- 모든 이미지들에 대한 annotation 정보들을 list 객체로 가짐.
- list 내의 개별 원소는 dict로 구성되면 개별 dict는 1개 이미지에 대한 annotation 정보를 가짐
- 1개 이미지는 여러 개의 Object bbox와 labels annotation 정보들을 개별 dict으로 가짐
- 1개 이미지의 Object bbox는 2차원 array로, object label은 1차원 array로 구성

filename, width, height, ann을 Key로 가지는 Dictionary를 이미지 개수대로 가지는 list 생성.
- filename: 이미지 파일명(디렉토리는 포함하지 않음)
- width: 이미지 너비
- height: 이미지 높이
- ann: bbounding box와 label에 대한 정보를 가지는 Dictionary
 -- bboxes: 하나의 이미지에 있는 여러 Object 들의 numpy array. 4개의 좌표값(좌상단, 우하단)을 가지고, 해당 이미지에 n개의 Object들이 있을 경우 array의 shape는 (n, 4)
 -- labels: 하나의 이미지에 있는 여러 Object들의 numpy array. shape는 (n, )
 -- bboxes_ignore: 학습에 사용되지 않고 무시하는 bboxes. 무시하는 bboxes의 개수가 k개이면 shape는 (k, 4)
 -- labels_ignore: 학습에 사용되지 않고 무시하는 labels. 무시하는 bboxes의 개수가 k개이면 shape는 (k,)
 '''

[
  
 { # 1개 img에 대한 annotaiton
   'filename': 'a.jpg',
   'width': 1280,
   'height' : 720,
   'ann' : { # obj에 대한 annotations
     'bboxes' : <np.ndarray, float32> (n,4), # 여기서 n은 이미지 안에 있는 obj 개수 / 2차원 array
     'labels' : <np.ndarray, int64> (n, ),
     'bboxes_ignore' : <np.ndarray, float32> (k,4),
     'labels_ignore' : <np.ndarray, int64> (k, ) (optional field)
     }
 },
  
  
 { .....
  ......
 },
  
]
