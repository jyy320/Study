# Study

Group 1. Basic Mathematical Concepts

	1.	Scalar(스칼라): 차원이 없는 값
	2.	Vector(벡터): 1차원으로 구성된 값
	3.	Matrix: 2차원으로 구성된 값
	4.	Matrix multiplication: 행렬 곱셈. ‘matmul’ 기능으로 수행.
	5.	Matrix element-wise product (Hadamard product): 행렬 곱셈과는 달리 동일한 크기의 행렬의 동일한 위치에 있는 원소끼리 곱한 값. ‘mul’ 기능으로 수행
	6.	Tensor (in deep learning, not mathematics): 3차원으로 구성된 (숫자)값.
	7.	Derivative(미분): 델타 x 가 0으로 수렴할 때의 변화율안 순간변화율, 또는 미분계수를 구한다.
	8.	Partial derivative(편미분): 다변수 함수의 특정 변수를 제외한 나머지 변수를 상수로 생각하여 미분
	9.	Gradient(그래디언트): 다변수 함수의 모든 입력값에서 모든 방향에 대한 순간변화율.
	10.	Convolution: 합성곱, 컨볼루션. 하나의 함수와 또 다른 함수를 반전 이동한 값을 곱한 다음, 구간에 대해 적분하여 새로운 함수를 구한다.
	11.	Cross-correlation (교차상관): 컨볼루션과 유사하나, 반전 이동하지 않은 함수값을 곱한 다음 구간에 대해 적분하여 새로운 함수를 구한다.
	12.	Probability distribution function (PDF): 연속적인 변수에 의한 확률 분포 함수. 특정 확률 변수 구간의 확률이 다른 구간에 비해 상대적으로 얼마나 높은가를 나타내며, 분포내에서 특정한 한 값에서의 확률은 0 이다. 항상 양의 값을 가져야 하며, 모든 범위의 PDF를 합하면 그 값이 1이여야 한다는 두 가지 조건을 충족해야 한다.
	13.	Entropy (Shannon Entropy in information theory): 확률분포의 모양을 설명하는 특징값. 확률분포가 가지고 있는 확신도/정보량을 수치로 표현한 것으로, 확률밀도가 특정값에 몰려있으면 엔트로피가 작다고 하고 반대로 여러가지 값에 골고루 퍼져 있다면 엔트로피가 크다고 한다.
	14.	KL Divergence (Kullback–Leibler Divergence, a.k.a. relative entropy): 쿨백-라이블러 발산으로 상대 엔트로피(relative entropy) 또는 정보 획득량(information gain)으로도 불린다. 두 확률분포의 정보 엔트로피 차이를 계산한다.
	15.	Cross-Entropy (also consider relationship with KL divergence): (확률분포로 된) 어떤 문제 p에 대해 (확률분포로 된) 특정 전략 q을 쓸 때 예측한 기댓값. 머신러닝에서 학습을 통해 p에 가까워질 수로 cross entropy 값은 작아지게 된다.

Group 2. Deep Learning Theory
	1.	Artificial Neuron: 생물학의 신경망을 본따서 만든 머신 러닝의 인공 신경망. 예시 모형 중에는 ‘Perceptron (퍼셉트론)’ 등이 있다.
	2.	Activation function (활성화 함수): 입력값에 대한 출력값을 활성화해주는 함수. 예시로는 Sigmoid, Leaky ReLU, tanh, Maxout, ReLU, ELU 등이 있다.
	3.	Weight (of neural network) (가중치): 각각의 입력신호에 부여된 고유한 숫자. 입력 데이터가 결과 출력에 주는 영향도를 조절하는 매개변수다. Wx+b의 ‘W’.
	4.	Bias (of neural network) (편향): Wx + b의 ‘b’ 절편. 뉴런이 얼마나 쉽게 활성화(activation)되는지 조정하는 매개변수로, 높으면 높을수록 분류의 기준이 엄격해지고, 낮아질수록 underfitting의 위험성이 높아진다.
	5.	Multi-layer perceptron (MLP; 다층 퍼셉트론): 가장 기본적인 형태의 인공신경망 구조로 하나의 입력층(input layer), 하나 이상의 은닉층(hidden layer), 그리고 하나의 출력층(output layer)로 구성된다.
	6.	Hidden layer (of deep neural network): Input layer나 Output layer와 달리 눈에 보이지 않는 은닉층.
	7.	Convolutional neural network (CNN; 컨볼루션 신경망): 이미지 처리에 탁월한 성능을 보이는 신경망. 크게 Convolution Layer와 Pooling Layer로 구성되어 있다. 
	8.	Kernel size (in CNNs): 커널(kernel)의 크기. 일반적으로 3 × 3 또는 5 × 5
	9.	Zero-Padding (in CNNs): convolution 연산 이후에도 특성 맵의 크기가 입력의 크기와 동일하게 유지되도록 하기 위한 패딩(padding)의 일종으로 테두리에 값을 0으로 채우는 것을 제로 패딩(zero padding)이라고 함.
	10.	Feature map (in CNNs) (특성맵): 입력으로부터 커널을 사용하여 합성곱 연산을 통해 나온 결과.
	11.	Channel (in CNNs): 색 성분
	12.	Forward propagation (순전파): 입력층에서 출력층으로 가중치를 업데이트
	13.	Loss function (손실함수): 신경망의 예측값과 타깃 실제값의 차이를 점수로 계산하는 함수. 비용 함수 (cost function)라고도 불림.
	14.	Backward propagation (Backpropagation; 역전파): Forward propagation과 달리 반대로 출력층에서 입력층 방향으로 계산하면서 가중치를 업데이트. 출력값에 대한 입력값의 기울기(미분값)부터 계산하여 거꾸로 전파시켜서 최종적으로 output값에 대한 입력층에서의 input data의 기울기 값을 구할 수 있다.
	15.	Optimizer (in deep learning (최적화 알고리즘): 정의한 비용 함수(Cost Function)의 값을 최소로 하는 W와 b를 찾는다.
	16.	Gradient descent (경사 하강법): Optimizer 알고리즘의 일종으로 cost가 최소화되는 기울기= 미분값 = 0의 지점까지 cost function을 미분하여 W값을 수정하는 과정.
	17.	Learning rate(학습률): W의 값을 변경할 때, 얼마나 크게 변경하는지.

Group 3. Deep Learning in Practice
	1.	Training dataset (학습 데이터): 결과 예측을 하는 모델을 학습시키기 위해 사용하는 데이터셋
	2.	Validation dataset: 학습이 완료된 모델을 검증하기 위한 데이터셋
	3.	Testing dataset: 모델이 얼마나 잘 작동하는지 판별하는 데이터셋. 최종 성능 평가용.
	4.	Ground Truth: 학습 데이터의 원본, 실제값
	5.	Overfitting (과적합): training data (학습 데이터)에서 잘 성능하는데 다른 데이터에서 성능을 못하는 경우.
	6.	Underfitting (과소적합): 모델이 training data (학습 데이터)에서 잘 성능하지 못하는 경우.
	7.	Parameter initialization (a.k.a. weight initialization): 파라미터 최적화, 또는 가중치 초기화. 제대로 된 최적화를 위해 학습 시작 시점에서 가중치를 잘 설정하기 위한 방법.
	8.	Epoch (in deep learning training): 전체 훈련 데이터가 학습에 한 번 사용된 주기/모델 학습 횟수.
	9.	Mini-batch (미니 배치): 전체 데이터를 나누어서 학습하는 더 작은 단위.
	10.	Supervised Learning (지도학습): 머신 러닝의 학습의 일종으로, unsupervised learning과 달리 정해진 정답이 있을 때 이에 가까운 예측을 하기 위한 학습이다. 크게 regression과 classification으로 분류된다. 
	11.	Classification(분류): Supervised learning 학습의 일종으로 기존에 존재하는 데이터 사이 관계를 파악하고 분별하여 분류한다.
	12.	Categorical Cross-Entropy: Multi-class classification에 주로 사용되며, 분류 문제에서 활성화함수 Softmax와 주로 사용하여 Softmax loss 라고도 불린다. 모든 category에 대한 크로스 엔트로피의 평균을 낸다.
	13.	Sigmoid (시그모이드/로지스틱 함수): ReLU 이전에 자주 쓰였던 활성화 함수로, S자 모양의 그래프를 만들 수 있는 로지스틱 회귀 가설 H(x)=f(Wx+b)에서의 f 함수다. 파이토치에서는 nn.Sigmoid를 통해 구현할 수 있다. 보통 2개의 선택지 중에서 1개를 고르는 이진 분류(Binary Classification)에서 사용한다.
	14.	Softmax (소프트맥스 함수): 로지스틱 회귀에서와 달리 3개 이상의 선택지로부터 1개를 선택하는 문제인 다중 클래스 분류(Multi-Class classification)에서 주로 사용하는 함수다. 
	15.	Logit (in machine learning, not in statistics/mathematics): Sigmoid 함수의 역함수로 In(P(x) / 1-P(x)). 로지스틱 회귀 분석에 쓰인다.
	16.	Rectified Linear Unit (ReLU; 렐루 함수): 오늘날 자주 사용되는 활성화 함수.  수식은 f(x)=max(0,x)로, 음수를 입력하면 0을 출력하고, 양수를 입력하면 입력값을 그대로 반환한다. 특정 양수값에 수렴하지 않으므로 깊은 신경망에서 시그모이드 함수보다 훨씬 더 잘 작동하고 이전 함수들의 문제들을 해결한다. 그러나 입력값이 음수면 기울기도 0이 되는 ‘죽은 렐루(dying ReLU)의 문제가 있어 이를 보완하기 위해 여러 ReLU의 변형 함수들이 등장.
