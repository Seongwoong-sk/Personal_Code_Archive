import xml.etree.ElementTree as ET # XML 파싱하는 편한 유틸리티

# 1개의 annotation 파일에서 bbox 정보 추출. 여러개의 object가 있을 경우 이들 object의 name과 bbox 좌표들을 list로 반환.
def get_bboxes_from_xml(anno_dir, xml_file):
  anno_xml_file = osp.join(anno_dir, xml_file)
  tree = ET.parse(anno_xml_file)
  root = tree.getroot()
  bbox_names = []
  bboxes = []

  # 파일내에 있는 모든 object Element를 찾음. 
  for obj in root.findall('object'): # 하나의 이미지에 여러개의 obj가 있을 수 있으니 findall
 
    #obj.find('name').text는 cat 이나 dog을 반환     
    #bbox_name = obj.find('name').text
    # object의 클래스명은 파일명에서 추출. 
    bbox_name = xml_file[:xml_file.rfind('_')] # rfind : index 추출 / xml 파일명 : Abyssinian_1-1.xml

    xmlbox = obj.find('bndbox')
    x1 = int(xmlbox.find('xmin').text) # int 안하면 str형태로 넘어옴.
    y1 = int(xmlbox.find('ymin').text)
    x2 = int(xmlbox.find('xmax').text)
    y2 = int(xmlbox.find('ymax').text)

    bboxes.append([x1, y1, x2, y2]) # 리스트 안의 단일 리스트 -> 2차원 
    bbox_names.append(bbox_name)

  return bbox_names, bboxes
