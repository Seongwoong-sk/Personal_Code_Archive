# 코랩에서 캐글 데이터 다운로드

# kaggle.json 다운로드

# kaggle.json 위치 변경 및 실행
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c 대회명