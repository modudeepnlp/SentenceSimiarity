# Paper Skim Sentence Simliarity with NLP implementation
+ 간단한 페이퍼 리뷰 및 업로드

## Target Dataset

+ [SNLI, Stanford](https://nlp.stanford.edu/projects/snli/)
+ [MNLI, NYU](https://www.nyu.edu/projects/bowman/multinli/)
+ [GLUE](https://gluebenchmark.com/leaderboard)
+ [QuestionPair, Song](https://github.com/songys/Question_pair)
+ ETC

### Paper List
* [ ] 448D Densely Interactive Inference Network (DIIN)
	+ https://arxiv.org/pdf/1709.04348.pdf
    + Summary
        + ICLR 논문으로 Model적인 의미보다는 타 Paper에 의해 검증 및 비교가 체계적임
        + Interactive Inference Netwrok (IIN)을 활용하여 semantic features를 좀 더 잘 뽑아 성능을 높혔다.
        + Input으로는 Glove, Char-feature (1D-Conv), Pos-tagger, Exact Matching (EM)을 활용
        + Embedding - Encoding - Interaction - Feature Extraction - Output Layer로 구성
        + 각 기능들 (Feature, 모델 구조 변형 등)에 대한 정확도 등에 설명이 잘 되있음

* [ ] DiSAN: Directional Self-Attention Network for RNN/CNN-Free Language Understanding
	+ https://arxiv.org/pdf/1709.04696.pdf
	+ Summary
		+ Attention 역사와 설명을 intro 에서 해줌
			+ premise = q, source = x
			+ multiply attention 과 additive attention 을 수식으로 설명 
				+ multiply attention - (비교) 메모리 효율성 좋음, 계산 효율 좋음
					+ f(xi, q) = <W1·xi, W2·q>  (<> - dot product symbol)
				+ additive attention - (비교) 성능 좋음
					+ f(xi, q) = wT·σ(W1·xi + W2·q)
		+ 논문은 additive self attention 이므로 f(xi, xj), 2가지 타입 
			+ token2token (self 로 token 끼리 interaction)
				+ f(xi, xj) = WT·σ(W1·xi + W2·xj + b1) + b2
			+ sorce2token (self 로 token 과 sequence 가 interaction)
				+ f(xi) =  WT·σ(W1·xi + b1) + b (수식이 이해안됨)	
			+ 여기서 중요한 포인트, 다차원 관점에서 보자. 
			+ 기존 additive attention 은 wT 로 소문자인 벡터
			+ 논문 additive attention 은 WT 로 대문자인 매트릭스
			+ attention score 가 여러개, 그 중 제일 best 인 것을 사용, 즉 다차원 관점에서 제일 좋은것을 사용하자
		+ (DiSA) Directional self-attention
			+ additive attention 을 변형, 
				+ reason 1. reduce parameter (상수 c의 역할 , c = 5 (always))
				+ reason 2. masked 를 씌어 interaction이 asymmetric 하게 
			+ Mask를 씌어 일시적 ordering 정보를 생성 (3가지 타입, fw,bw,disable)
			+ f(xi, xj) = c·tanh(W1·hi + W2·hj + b1/c)+ M·1. (1 = all-one vector)
		+ Fusion
			+ F 로 비율을 구하여 원본과 mask additive self attention 을 섞음
			+ fusion output이 DiSA 의 output
		+ (DiSAN) Directional Self-Attention Network
			+ equation 17 을 fw
			+ equation 18 을 bw 
			+ fw, bw 을 concat 한 것을 source로 보고 sorce2token으로 output 구함
		+ Remark 
			+ 이 논문의 특이점, 다차원관점을 사용했다. 
			
* [ ] HBMP: Hierarchical BiLSTM with Max Pooling
	+ https://arxiv.org/pdf/1808.08762.pdf
	+ Summary
        	+ promise, hypothesis 둘다 BiLSTM 3 layer를 쌓아서 각 레이이어 출력을 concat 해서 embedding 값으로 사용 함
        	+ embedding 출력 u(promise 출력), v(hypothesis 출력)를 [u; v; |u-v|; u * v] 형태로 concat 함
		* concat 한 출력을 linear-layer 3개를 사용해서 최종 출력을 만듬
        	+ 모델이 간단하고 구현하기 어렵지 않으면서도 나쁘지 않은 성능을 보여 줌
	+ Summary2
		+ intro
			+ iterative refinement strategy (hierarchy of BiLSTM and max pooling)
			+ model the inferential relationship between two or more given sentences
			+ In particular, given two sentences - the premise p and the hypothesis h
		+ Natural Language Inference by Tree-Based Convolution and Heuristic Matching (Mou et al. (2016))
			+ sentence embeddings are combined using a heuristic
			+ linear offset of vectors can capture relationships between two words
			+ but it has not been exploited in sentence-pair relation recognition.(Mikolov et al., 2013b),
			+ Our study verifies that vector offset is useful in capturing generic sentence relationships
			+ m = [h1; h2; h1 − h2; h1 ◦ h2]  (concat, difference, product)
			+ vector representations of individual sentences are combined to capture the relation between the p and h
			+ As the dataset is large, we prefer O(1) matching operations because of efficiency concerns. 
			+ first one (concat) = 
				+ the most standard procedure of the “Siamese” architectures, 
				+ W[h1, h2], where W = [W1, W2]
			+ latter two (difference, product) = 
				+ certain measures of “similarity” or “closeness.”
				+ special cases of concatenation
				+ element-wise product, 용량관점에서 concat에 포함 (same)
				+ element-wise difference, W0(h1−h2) = [W0, −W0][h1, h2]T
			+ heuristic significantly improves the performance
		+ Supervised Learning of Universal Sentence Representations from Natural Language Inference Data (Conneau et al. (2017))
			+ max-pooling - (4 conv layers) representations of the sentences at different level of abstractions에서 강한거 추출
			+ hierarchical convolutional network = blends different levels of abstraction.
			+ hierarchical 장점 
				+ (attentive 세심하게, 주의 깊게) 이전 레이어의 가중치를 받아 같은 동작을 다시 반복함으로써 attentive 해짐
				+ concat 으로 계층마다 다른 관점들을 블랜딩 함으로써 표현이 주의깊은 추상화가 됨
			+ fully connected layer 가 한개
		+ iterative refinement architecture
			+ 이전 LSTM 레이어의 information을 다음 레이어의 initialisation 함으로써 반복적 정제 아키텍쳐를 가진다고 함.
		+ 3-way softmax
			+ 3-class
		+ Max pooling is defined in the standard way of taking the highest value over each dimension of the hidden states = u
		+ sentence encoder 에서의 u1, u2, u3 은 concat 하여 나오고, premise와 hypothesis, 2개의 encoder가 있음.
	
* [ ] Learning Sentence Similarity with Siamese Recurrent Architectures
	+ https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023
* [ ] Fine-Tuned LM-Pretrained Transformer
	+ https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
