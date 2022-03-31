# 코랩에서 캐글 데이터 다운로드

# kaggle.json 다운로드
# 403 에러 발생 시 캐글 해당 대회 Rule Acception 확인 

# kaggle.json 위치 변경 및 실행
# kaggle.json을 .kaggle 폴더 안에 바로 넣으면 안되고 .kaggle 폴더와 같은 경로에서 명령어로 넣어야 다운로드 실행됨.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c 대회명

