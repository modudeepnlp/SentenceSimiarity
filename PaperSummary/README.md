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


* [ ] Learning Sentence Similarity with Siamese Recurrent Architectures
	+ https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023
* [ ] Fine-Tuned LM-Pretrained Transformer
	+ https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
