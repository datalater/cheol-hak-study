
ⓒ JMC 2017

---

## W1 :: MNIST 데이터를 분류하는 신경망

**딥러닝을 이용한 분류 알고리즘의 기본 접근법**:

1. 훈련 : 최적의 weight 학습
2. 추론 : 학습한 weight로 입력 데이터를 분류

**MNIST 데이터 분류**:

+ 먼저 추론에 필요한 함수를 정의합니다.

```python

# MNIST 데이터를 불러와서 training set과 test set으로 나눕니다.
def get_data():
    (x_train, t_test), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 미리 학습시킨 weight를 저장했던 pickle 파일을 불러옵니다.
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

# 레이어에서 input과 weight를 선형결합한 값에 적용할 비선형 함수로 sigmoid를 정의합니다.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 해당 class에 대한 score 값을 확률값으로 변형하는 softmax 함수를 정의합니다.
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# 새로운 test 데이터를 network에 넣어서 score 값을 출력합니다.
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y
```

+ 테스트 데이터에 대한 추론을 실행합니다.

```python
# x_test와 t_test를 리턴합니다.
x, t = get_data()

# 미리 저장해둔 hidden layer1 (50 units) + hidden layer2 (100 units)으로 구성된 network를 리턴합니다.
network = init_network()

# 정확도를 나타내는 변수 accuracy_cnt를 정수 타입으로 초기화합니다.
accuracy_cnt = 0

# x_test를 하나씩 순회합니다.
for i in range(len(x)):
    # 순서대로 x_test를 network에 넣어서 softmax를 통과한 10차원 벡터 y를 리턴합니다.
    y = predict(network, x[i])
    # 10차원 벡터 y 중에서 가장 값이 큰 index를 p에 저장합니다.
    p = np.argmax(y)
    # 해당 x_test의 label과 p가 일치하면 정확도에 1을 더합니다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```

### Quote

+ `sys.path.append(os.pardir)` : 부모 디렉토리부터 파일을 찾을 수 있도록 path를 설정한다.
+ 피클 : 프로그램 실행 중에 특정 객체를 파일로 저장해서, 다른 파일에서 즉시 로드할 수 있는 기능

### Further study

+ 네트워크 코드 직접 짜보기 (W2)
+ 피클로 저장하는 법 해보기
+ 네트워크를 짤 때 뉴런의 개수가 어떤 의도를 가진 것인지 의미 파악하기

---

**끝.**
