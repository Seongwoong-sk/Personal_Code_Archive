# 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')

# 경로 이동
%cd /content/drive/MyDrive/------
!unzip -qq "zip 파일 경로"

# 업로드가 잘 됐는지 파일 개수로 확인
from glob import glob
filepaths = list(glob('content/image/*.jpg'))
len(filepaths)
