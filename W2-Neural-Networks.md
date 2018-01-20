ⓒ JMC 2017

---

## W2 :: 신경망 구성하기

**신경망의 내적**:

+ 넘파이 행렬을 써서 신경망을 구현한다.
+ [그림3-14]를 보고 코드로 짜본다.

```python
X = np.array([1,2])
W = np.array([[1,3,5], [2,4,6]])

Y = np.dot(X, W)
print(Y)
```

+ 넘파이 행렬을 써서 3층 신경망을 구현한다.
+ 아래 코드를 그림으로 그려본다. [그림3-15]

```python
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def identity_function(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3

    # 출력층의 함수로써 항등 함수를 썼다.
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
```

**출력층의 활성화 함수**:

+ binary classification : 일반적으로 sigmoid 사용
+ multiclass classificaiton : 일반적으로 softmax 사용

**소프트맥스 함수의 장점**:

```python
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
```

소프트맥스 함수를 출력층에 적용하면 확률적으로 문제를 대응할 수 있는 장점이 생긴다.
이전 레이어에서 계산된 score를 확률값으로 해석할 수 있도록 도와주기 때문이다.

**소프트맥스 함수 구현시 주의점**:

소프트맥스 함수는 지수 함수를 사용하고, 지수 함수는 매우 큰 값을 내뱉기 때문에 자칫 무한대의 숫자를 출력하기가 쉽다.
무한대의 숫자끼리 나눗셈을 하면 수치가 '불안정'해진다.
이렇게 표현할 값의 크기가 너무 커서 메모리를 지나치게 많이 쓰게 되는 문제를 '오버플로'라고 한다.

그래서 오버플로를 방지하기 위해서 소프트맥스 함수를 적용할 원소 중 최대값을 구해놓고, 각 원소마다 최대값을 뺀 후(일종의 정규화)에 지수함수를 적용하는 방법을 사용한다.

```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 오버플로 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
```

**데이터 전처리와 정규화**:

데이터를 특정 범위로 변환하는 처리를 정규화(normalization)라고 한다.
신경망의 입력 데이터에 특정 변환을 가하는 것을 전처리(pre-processing)라고 한다.
예를 들어, MNIST 데이터의 픽셀값을 0~255 범위인 것을 0.0~1.0 범위로 변환하는 것은 입력 이미지에 대한 전처리 작업으로 정규화를 수행한 것이다.

**데이터 전처리를 하는 이유와 백색화**:

현업에서 딥러닝에 전처리를 활발히 사용한다.
전처리를 통해 식별 능력을 개선하고 학습 속도를 높일 수 있기 때문이다.
실제로 전처리를 할 때는 데이터 전체의 분포를 고려한다.
예를 들어 데이터 전체 평균과 표준편차를 이용하여 데이터들을 0을 중심으로 분포하도록 이동하거나, 데이터의 확산 범위를 제한하는 정규화를 수행한다.
그 외에도 전체 데이터를 균일하게 분포시키는 데이터 백색화(whitening)등도 있다.

### Quote

+ `X = np.array([1,2])`: 로우벡터
+ 신경망 그림에서 `W = np.array([[1,3,5], [2,4,6]])`의 순서는 np.dot(X, W)의 연산순서에 따른 것이다.

### Further study (조교님들께 메일로 질문하기)

+ (1) 아래 코드에서 X의 row=1, column=2 인데 (2,)으로 표시되는 이유?

```python
X = np.array([1,2])
X.shape

# (2,)
```

+ (2) 출력층의 활성화 함수에서 binary classification에서는 sigmoid를 쓰고, multiclass classification에서는 softmax를 사용하는 이유? (생각해보니 softmax는 sigmoid 보다 exp 연산을 더 많이 하므로 연산이 더 비싸서 binary에서는 sigmoid를 쓰는 것 같다)

+ (3) 책에서 말하기를, 지수함수에 대한 연산을 생략해도 문제가 없으므로 추론 단계에서는 softmax 함수를 출력층에 적용하지 않는 게 일반적이라고 말했다. 그런데, 같은 논리대로 학습 단계에서도 softmax 함수를 생략해도 되지 않나? 학습 단계에서 softmax 함수를 굳이 적용하는 이유가 무엇일까?

**END (~3.6.2)**

---
