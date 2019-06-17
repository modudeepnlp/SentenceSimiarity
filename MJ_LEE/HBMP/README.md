## python main.py 로 실행하면 됩니다.
- main.py → vocab.py → custom_dataset.py → main.py → model.py → main.py 
- https://arxiv.org/pdf/1808.08762.pdf
- 600D HBMP, paper accuracy : 86.6%
# 2019-06-16 
* [ ] (1)
- dev accuracy :  76.34762308998302 %
- test accuracy :  75.67911714770797 %
- config = {epoch:5, batch:256,  learning_rate:0.0005,  embedding_dim:32,  hidden_size:32,  linear_hidden_size:32}
* [ ] (2)
- dev accuracy :  78.46283783783784 %
- test accuracy :  77.2804054054054 %
- config = {epoch:10, batch:256,  learning_rate:0.0005,  embedding_dim:64,  hidden_size:64,  linear_hidden_size:64}

# 2019-06-17 
* [ ] (1)
- dev accuracy :  78.28336148648648 %
- test accuracy :  77.54434121621621 %
- config = {epoch:10, batch:256,  learning_rate:0.0005,  embedding_dim:128,  hidden_size:128,  linear_hidden_size:128}
* [ ] (2)
- dev accuracy :  79.5924831081081 %
- test accuracy :  78.63175675675676 %
- config = {epoch:5, batch:256,  learning_rate:0.0005,  embedding_dim:300,  hidden_size:128,  linear_hidden_size:128}
* [ ] (3)
- dev accuracy :  79.49746621621621 %
- test accuracy :  79.04349662162163 %
- config = {epoch:3, batch:256,  learning_rate:0.0005,  embedding_dim:300,  hidden_size:128,  linear_hidden_size:128}
