'''
모듈

한 파일 내에 많은 기능들을 구현한다면 코드가 복잡해지며 지저분하게 보일 수 있다. 또한 같은 기능이나 별도로 관리해야 할 내용에 대해서 가독성이 떨어진다. 따라서 코드를 구현할 때에는 기능에 따라 파일을 별도로 만들어 전체적인 코드를 완성한다. 또한 별도의 파일을 생성하기 때문에 다른 업무에서 동일 작업이 필요 할 때 파일만 불러와서 사용할 수 있어 중복 된 코드를 작성할 필요가 없다. 이 때 전역 변수, 함수 등을 포함한 별도의 파일을 모듈이라고 한다.

'''

# models.py 안에 linear_regression, gaussian_mixture, random_forest 함수가 있다고 가정
# models.py 경로랑 연결해서

# 모델 불러오기  : 라이브러리와 동일
import models

models.linear_regression(data)
models.gaussian_mixture(data)
models.random_forest(data)


# 모듈의 특정 함수만 가져오기
from models import linear_regression

linear_regression(data)