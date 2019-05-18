# Sentence Simliarity with NLP implementation
+ 문장 유사도 관련된 모델에 대해 R&D 리포

## Usage
+ 유사도를 기반으로 다양한 분야에 활용이 가능합니다. 

|      TASK        | X | Y | REF |
| :--------------- | :-------: | :------------: | :------: |
| Web Search       |  Search query | Web document|  Huang+ 13; Shen+ 14; Palangi+ 16 |
| Entity linkin    |  Entity mention and context |  Entity and its corresponding page | Gao+ 14b |
| Online recommendation  |  Doc in reading |     Interesting things / other docs|  Gao+ 14b  |
| Image captioning       |  Image |     Text     |  Fang+ 15 |
| Machine translation    |  Sentence in language A |     Translations in language B |  Gao+ 14a |
| Question answering     |  Question  |     Answer     |  Yih+ 15  |

## Target Dataset

+ [SNLI, Stanford](https://nlp.stanford.edu/projects/snli/)
+ [MNLI, NYU](https://www.nyu.edu/projects/bowman/multinli/)
+ [GLUE](https://gluebenchmark.com/leaderboard)
+ [QuestionPair, Song](https://github.com/songys/Question_pair)
+ ETC

### SNLI
+ Hyper-parameter was arbitrarily selected.

|                  | Train ACC | Validation ACC | Test ACC |
| :--------------- | :-------: | :------------: | :------: |
| Baseline (Feed Forward)         |  -  |     -     |  -  |
| MaLSTM           |  -  |     -     |  -  |

* [ ] Learning Sentence Similarity with Siamese Recurrent Architectures
	+ https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023
* [ ] Fine-Tuned LM-Pretrained Transformer
	+ https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf