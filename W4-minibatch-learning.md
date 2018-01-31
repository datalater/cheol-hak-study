---
layout: post
title: W4 미니배치와 손실함수
---

ⓒ JMC 2018

---

## W4 :: 미니배치와 손실함수

**학습**:

+ 데이터로부터 가중치의 값을 정하는 방법을 학습이라고 한다.

**end-to-end**:

+ 데이터(입력)에서 목표한 결과(출력)를 사람의 개입 없이 얻는 알고리즘
+ 신경망은 모든 문제를 주어진 데이터 그대로를 입력 데이터로 활용해 'end-to-end'로 학습한다.

**오버피팅**:

+ 한쪽 이야기(특정 데이터셋)만 너무 많이 들어서 편견이 생겨버린 상태

**손실 함수**:

+ 모델에서 최적의 가중치 값을 탐색하기 위한 지표
+ 모델의 나쁨을 나타내는 지표

**...평균제곱오차 (MSE)**:

+ 추정값과 참값의 차이를 제곱해서 평균 낸 값
+ $E = \frac{1}{2}\Sigma_{k}(y_k - t_k)^{2}$

**...교차엔트로피오차 (CEE)**:

+ 추정값의 확률분포와 참값의 확률분포의 차이
+ $E = -\Sigma_{k}t_{k}\log{y_k}$
+ $t_k$가 원-핫 인코딩이기 때문에 실질적으로는 정답일때의 출력만 계산된다.

```python
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
```

**미니배치**:

+ 훈련 데이터 일부 중에서 표본 추출한 것

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```

+ `np.random.choice()` : 지정한 범위의 수 중에서 무작위로 원하는 개수만 꺼내는 함수
+ `np.random.choice(60000, 10)` : 0 이상 60000 미만의 수 중에서 무작위로 10개를 골라냅니다.

**미니배치 vs. 배치**

+ 배치: 중복 불가
+ 미니배치: 중복 가능 (무작위 랜덤 표본 추출이기 때문에)

`@@resume : 4.2.4부터 해야 함`

---

## Further study

+ **(not yet)** 피클로 저장하는 법 해보기

### solved

+ **(solved)** 부모 디렉토리의 부모 디렉토리에 있는 파이썬 파일 import하는 법 ==
```python
import os

os.listdir('.')
```
+ **(solved)** 네트워크를 짤 때 뉴런의 개수가 어떤 의도를 가진 것인지 의미 파악하기 == input feature를 뉴런 개수만큼 압축하거나 더 세밀하게 나눈다.
+ **(solved)** X의 row=1, column=2 인데 (2,)으로 표시되는 이유? == 1차원 칼럼벡터나 1차원 로우벡터는 (k,)으로 표시한다.
+ **(solved)** 출력층의 활성화 함수에서 binary classification에서는 sigmoid를 쓰고, multiclass classification에서는 softmax를 사용하는 이유? == 생각해보니 softmax는 sigmoid 보다 exp 연산을 더 많이 하므로 연산이 더 비싸서 binary에서는 sigmoid를 쓰는 것 같다.
+ **(solved)** 학습 단계에서 softmax 함수를 굳이 적용하는 이유가 무엇일까? == 학습시 loss를 계산해서 optimize를 할 때 카테고리 데이터는 score로 loss를 최적화하기보다는 각 카테고리에 해당할 확률을 계산해서 loss를 최적화하는 것이 논리적으로 더 맞다.
+ **(solved)** 추론 단계에서 softmax 함수를 생략해도 되는 이유는? == 학습 단계와 달리 loss를 구해서 optimize할 필요가 없고 단지 score 벡터의 argmax만 구하면 되기 때문이다.

---

**END (~4.2.4)**
