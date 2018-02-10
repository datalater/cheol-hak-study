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

**손실 함수 - 평균제곱오차 (MSE)**:

+ 추정값과 참값의 차이를 제곱해서 평균 낸 값
+ $E = \frac{1}{2}\Sigma_{k}(y_k - t_k)^{2}$

**손실 함수 - 교차엔트로피오차 (CEE)**:

$$E = -\Sigma_{k}t_{k}\log{y_k}$$

+ (1) 공식 $E$를 보면 $k$는 데이터 번호이고, $y_k$는 prediction, $t_k$는 원-핫 인코딩된 정답 레이블입니다.
+ (2) $t_k$의 shape는 (10, 1)이고 $y_k$는 shape가 (1,10)입니다.
+ (3) $t_k$와 $log{y_k}$가 곱해지면 $t_k$에서 값이 1인 row-index에 해당하는 $log{y_k}$인 column-index만 살아남습니다.
+ (4) 가령, $t_k$ = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]이면 $log{y_k}$[2]만 살아남습니다.
+ (5) 즉, $t_k$의 정답 레이블 인덱스에 해당하는 prediction 값만 loss에 계산됩니다

그러므로 교차엔트로피 오차는 정답이 아닌 것은 신경쓰지 않고 정답인 것은 무조건 잘 맞추도록 유도하는 손실 함수라고 볼 수 있습니다.

**교차엔트로피오차 (CEE) 코드 구현 (1) t가 원-핫 인코딩일 때**:

```python
# e.g. t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
def cross_entropy_error(y, t):
  delta = 1e-7
  return -np.sum(t * np.log(y + delta))
```
np.log() 함수에 0이 입력되면 마이너스 무한대를 뜻하는 -inf가 되어 더 이상 계산을 진행할 수 없게 된다.
그래서 아주 작은 값인 delta를 더해줘서 절대 0이 되지 않도록 만드는 것입니다.

**교차엔트로피오차 (CEE) 코드 구현 (2) t가 숫자 레이블일 때**:

```python
# e.g. t = [2]
def cross_entropy_error(y, t):
  delta = 1e-7
  return -np.sum(np.log(y + delta)[t])
```

$t_k$ = [2]이면 $log{y_k}$[2]만 살아남아야 하기 때문에 np.log(y + delta)에 [t]를 붙여줍니다.

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

**미니배치 vs. 배치**:

전체 데이터가 60,000 장이라고 했을 때,

+ 배치:
    + 데이터 전체를 여러 개 묶음으로 나눈 것
    + 배치 사이즈가 100 장이라면, 배치 하나당 100장이고 총 배치 개수는 600개가 됩니다.
+ 미니배치:
    + 데이터 전체를 일부로 추린 것
    + 미니배치 사이즈가 100 장이라면, 60,000 장에서 100장만 표본추출한 것입니다.

**배치용 교차엔트로피 오차 구현하기**:

```python
# t: 원핫 인코딩일 때
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshpae(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size
```

```python
# t: 숫자 레이블일 때
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / bath_size
```

**최적화의 기준으로 정확도가 아니라 손실 함수를 사용하는 이유**:

최적의 가중치를 찾을 때 정확도보다 손실 함수를 써야 정밀하게 최적화할 수 있습니다.
왜냐하면 정확도는 손실 함수 값보다 불연속적이기 때문입니다.
불연속적이면 미분값이 0이 되는 곳이 많다는 뜻이고,
미분값이 0이 되면 그만큼 정밀하게 최적화할 수 없다는 말이기 때문입니다.


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
+ **(solved)** X의 row=1, column=2 인데 (2,)으로 표시되는 이유? == numpy는 1차원 칼럼벡터나 1차원 로우벡터는 (k,)으로 표시한다.
+ **(solved)** 출력층의 활성화 함수에서 binary classification에서는 sigmoid를 쓰고, multiclass classification에서는 softmax를 사용하는 이유? == 생각해보니 softmax는 sigmoid 보다 exp 연산을 더 많이 하므로 연산이 더 비싸서 binary에서는 sigmoid를 쓰는 것 같다.
+ **(solved)** 학습 단계에서 softmax 함수를 굳이 적용하는 이유가 무엇일까? == 학습시 loss를 계산해서 optimize를 할 때 카테고리 데이터는 score로 loss를 최적화하기보다는 각 카테고리에 해당할 확률을 계산해서 loss를 최적화하는 것이 논리적으로 더 맞다.
+ **(solved)** 추론 단계에서 softmax 함수를 생략해도 되는 이유는? == 학습 단계와 달리 loss를 구해서 optimize할 필요가 없고 단지 score 벡터의 argmax만 구하면 되기 때문이다.

---

**END (~4.2)**
