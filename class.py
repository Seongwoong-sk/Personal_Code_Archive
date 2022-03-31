# Class

'''
어떤 하나의 기능을 구현 하는데 여러 개의 함수가 필요할 때가 있다. 이 때, 데이터와 세부적인 기능을 수행하는 함수들을 묶어서 구현할 수 있는데 이 때 기본적으로 사용되는 것이 클래스이다. 클래스는 상속 등의 다양한 기능을 통해 프로그램의 복잡도를 감소시켜 주며 확장에 매우 유리하게 작용한다. 또한 중요 변수를 클래스에 넣어 외부의 변수들과 혼동 될 위험을 줄여준다.

'''

import numpy as np
from matplotlib import pyplot as plt

data = np.random.normal(0,10, (100,1)) # 평균이 0이고 표준편차가 10인 정규분포에서 (100 x 1 배열 : 2차원)
target = np.random.normal(0,1, 100) # 평균이 0, 표준편차가 1인 샘플 데이터 100개



# 클래스는 쉽게 함수들의 모임
# 보통 클래스명은 대문자
class LinearReg:

    # 클래스 안에 있는 함수들은 self라는 것을 반드시 적어줘야 함. 그 다음 우리가 불러올 값들 적어주면 됨.
    # --> Self가 있냐 없냐의 차이는 ? 이 클래스 안에서 self라고 지정을 해주면, 다른 함수에서도 별도의 선언 없이 그대로 사용할 수 있음
    # --> self : self로 지정된 인스턴스 기준으로 계산하겠다. 클래스 안에 있는 함수는 self를 적어준다.

    # 클래스 외부에서 내부로 값을 받고 싶다면 __init__이라는 함수를 사용해서 외부의 값을 받을 수 있음.
    def __init__(self, data, target, scaling=False):
        self.data = data
        self.target = target
        self.num_instances = self.data.shape[0]  # 행
        self.num_features = self.data.shape[1]  # 열
        self.scaling = scaling
        print(f"num_instances={self.num_instances}, num_features={self.num_features}, scaling={scaling}")


    # 여기서 data를 self라고 지정안하면 data 쓸 때 또 받아야됨.--> self.data라고 이미 선언을 해서 이 data는 클래스 안에서 자유롭게 쓸 수 있따.
    def minmax(self):
        for i in range(self.num_features):
            col = self.data[:,i]
            # self.data가 업데이트됨.
            self.data[:,i] = (self.data[:,i]-np.min(col))/(np.max(col)-np.min(col))
        
        return self # 메서드에서 self를 반환한다는 것은 단순히 메서드가 호출 된 인스턴스 개체에 대한 참조를 반환한다는 것을 의미  
                    # 이 함수가 호출되서 다시 인스턴스 개체 (__init__)에 반환했따.

    # data processing
    # data split
    # 등 ...
    
    def solver(self):   
        
        if self.scaling == True:
            self.minmax()
        
        # LR.......
        # ........
        plt.plot(self.data, self.target,'.')
        plt.show()
          



## 호출

m1 = LinearReg(data, target) # scaling은 이미 지정된 값이 있어서 굳이 안 적어도됨. 지정된 값이 업을 경우엔 class 소환 시 지정 필수
m1.solver()
  
m2 = LinearReg(data, target, True)
m2.solver()