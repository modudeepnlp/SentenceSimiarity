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
	+Summary
		+ Attention 역사와 설명을 intro 에서 해줌
		+ premise = q, source = x
		+ multiply attention 과 additive attention 을 수식으로 설명 (이 논문은 additive attention 사용)
			+ multiply attention - (비교) 메모리 효율성 좋음, 계산 효율 좋음
				+ f(xi, q) = <W1*xi , W2*q>  (<> - dot product symbol)
			+ additive attention - (비교) 성능 좋음
				+ f(xi, q) = wT*σ(W1*xi + W2*q)
		+ additive self attention 이므로 f(xi, xj), attention을 2가지 타입으로 정의
			+ 1. token2token (self 로 token 끼리 interaction)
				+ f(xi, xj) = WT * σ(W1*xi + W2*xj + b1) + b2
			+ 2. sorce2token (self 로 token 과 sequence 가 interaction)
				+ f(xi) =  WT * σ(W1*xi + b1) + b
				+ 수식이 이해안됨
				
			+여기서 중요한 포인트, 다차원 관점에서 보자. 
			+기존 additive attention 은 wT 로 소문자인 벡터
			+논문 additive attention 은 WT 로 대문자인 매트릭스
			+값이 여러개가 나온다. 그 중 제일 best 인 것을 사용, 즉 다차원 관점에서 제일 좋은것을 사용
			
			

* [ ] Learning Sentence Similarity with Siamese Recurrent Architectures
	+ https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023
* [ ] Fine-Tuned LM-Pretrained Transformer
	+ https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
