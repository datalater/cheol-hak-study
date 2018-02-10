---
layout: post
title: W5 그래디언트
---

ⓒ JMC 2018

---

## W5 :: 그래디언트

**수치 미분 vs. 해석적 미분**

+ 수치 미분 (numerical gradient) : 아주 작은 차분(h)으로 미분하는 것 == 오차가 아주 작지만 존재할 수밖에 없는 미분
+ 해석적 미분 (analytic gradient) : 수식을 전개해서 미분하는 것 == 오차가 없는 진정한 미분

**수치 미분 코드 구현하기**:

```python
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)
```

$\frac{f(x+h)-f(x)}{h}$ 보다 $\frac{f(x+h)-f(x-h)}{2h}$를 쓰는 게 진정한 미분과의 오차가 적으므로 코드 구현할 때는 항상 후자를 쓰도록 합니다.

수치 미분 예시 코드:

$$y = 0.01x^{2} + 0.1x$$

```python
def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

>>> numerical_diff(function_1, 5)
0.1999999999990898
>>> numerical_diff(function_1, 10)
0.2999999999986347
```

**편미분**:

변수가 여럿인 함수에 대한 미분을 편미분이라고 합니다.

```python
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)
```

예시 코드:

$$(f(x_0, x_1) = x_0^{2}+x_1^{2}$$

```python
def function_2(x):
    return x[0]**2 + x[1]**2
```

문제1: x0=3, x1=4일 때 x0에 대한 편미분을 구하라. (x1은 상수 취급한다)

```python
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

>>> numerical_diff(function_tmp1, 3.0)
6.00000000000378
```

문제2: x0=3, x1=4일 때 x1에 대한 편미분을 구하라. (x0은 상수 취급한다)

```python
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

>>> numerical_diff(function_tmp2, 4.0)
7.999999999999119
```

**기울기(gradient)**:

모든 변수의 편미분을 벡터로 정리한 것을 기울기(gradient)라고 합니다.

$$(f(x_0, x_1) = x_0^{2}+x_1^{2}$$

```python
def function_2(x):
    return x[0]**2 + x[1]**2
```

기울기 예시 코드:

```python
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad
```

numerical_gradient(f, x) 함수의 인수인 f는 함수이고 x는 넘파이 배열이므로 넘파이 배열 x의 각 원소에 대해서 수치 미분을 구합니다.
그러면 이 함수를 사용해서 실제로 기울기를 계산해봅시다.
세 점 (3, 4), (2, 0), (3, 0)의 기울기를 구해보겠습니다.

```python
>>> numerical_gradient(function_2, np.array([3.0, 4.0]))
array([ 6.,  8.])
>>> numerical_gradient(function_2, np.array([0.0, 2.0]))
array([ 0.,  4.])
>>> numerical_gradient(function_2, np.array([3.0, 0.0]))
array([ 6.,  0.])
```

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

**END**
