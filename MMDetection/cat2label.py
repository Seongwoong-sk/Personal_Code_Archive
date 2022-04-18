# Data anno에서 클래스 이름으로 받아서 딕셔너리 만들 때 키하고 밸류 순서 바꿈

CLASSES =('Car', 'Truck', 'Pedestrian', 'Cyclist')
cat2label = {k:i for i, k in enumerate(CLASSES)} # name이 키가 됨. value가 이름
print(cat2label) 

'''
{'Car': 0, 'Truck': 1, 'Pedestrian': 2, 'Cyclist': 3}
0
'''
