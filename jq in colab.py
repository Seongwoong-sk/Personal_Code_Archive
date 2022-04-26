# annotation json 파일을 잘 볼수 있는 jq 유틸리티 셋업. 
!sudo apt-get install jq

!jq . 경로/train.json > output.json
!head -100 output.json
