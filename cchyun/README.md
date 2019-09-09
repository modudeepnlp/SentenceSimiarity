# 2019-05-24 - baseline:
  - config: { "n_vocab": len(vocab), "n_embed": 32, "n_output": 3, "n_epoch": 10, n_batch": 256, "n_layer": 1, "cells": 2, "dropout": 0.1 ## HBMP }
  - dev:  66.7
  - test: 66.4


# 2019-06-02 - baseline + (p;h;|p-h|;p*h):
  - config: { "n_vocab": len(vocab), "n_embed": 32, "n_output": 3, "n_epoch": 10, n_batch": 256, "n_layer": 1, "cells": 2, "dropout": 0.1 ## HBMP }
  - dev:  75.3
  - test: 75.8


# 2019-06-07 - Prevous + HBMP:
  - config: { "n_vocab": len(vocab), "n_embed": 32, "n_output": 3, "n_epoch": 10, n_batch": 256, "n_layer": 1, "cells": 2, "dropout": 0.1 ## HBMP }
  - dev:  77.6
  - test: 77.5


# 2019-06-07 - Prevous + layer3:
  - config: { "n_vocab": len(vocab), "n_embed": 32, "n_output": 3, "n_epoch": 10, n_batch": 256, "n_layer": 1, "cells": 2, "dropout": 0.1 ## HBMP }
  - dev:  77.3
  - test: 77.2


# 2019-06-07 - Prevous + RELU(2):
  - config: { "n_vocab": len(vocab), "n_embed": 32, "n_output": 3, "n_epoch": 10, n_batch": 256, "n_layer": 1, "cells": 2, "dropout": 0.1 ## HBMP }
  - dev:  78.6
  - test: 78.6


# 2019-06-16 - Prevous + { "n_layer": 2, "cells": 4 }
  - config: { "n_vocab": len(vocab), "n_embed": 32, "n_output": 3, "n_epoch": 10, "n_batch": 256,  "n_layer": 2, "cells": 4, "dropout": 0.1 ## HBMP }
  - dev:  79.3
  - test: 78.9


# 2019-06-16 - Prevous + dropout 2
  - config: { "n_vocab": len(vocab), "n_embed": 32, "n_output": 3, "n_epoch": 15, "n_batch": 256,  "n_layer": 2, "cells": 4, "dropout": 0.1 ## HBMP }
  - dev:  78.1
  - test: 78.5


# 2019-06-18 - Prevous + various n_embed
  - n_embed':  32, loss: 0.531, dev: 79.029, test: 78.196
  - n_embed':  64, loss: 0.559, dev: 77.677, test: 77.626
  - n_embed':  96, loss: 0.590, dev: 76.956, test: 76.557
  - n_embed': 128, loss: 1.099, dev: 33.824, test: 34.283
  - n_embed': 160, loss: 0.575, dev: 77.566, test: 77.351
  - n_embed': 192, loss: 0.615, dev: 75.940, test: 76.252
  - n_embed': 224, loss: 0.627, dev: 75.483, test: 75.428
  - n_embed': 256, loss: 0.616, dev: 75.930, test: 75.631
  - n_embed': 288, loss: 0.600, dev: 76.214, test: 76.110
  - n_embed': 320, loss: 0.704, dev: 74.670, test: 74.379


# 2019-06-18 - Prevous + train vocab only (minfreq=1)
  - config: {'n_vocab': 33000, 'n_embed': 32, 'n_output': 3, 'n_epoch': 20, 'n_batch': 256, 'n_layer': 2, 'cells': 4, 'dropout': 0.1}
  - dev:  78.7
  - test: 78.4


# 2019-06-18 - Prevous + train vocab only (minfreq=2)
  - config: {'n_vocab': 23381, 'n_embed': 32, 'n_output': 3, 'n_epoch': 20, 'n_batch': 256, 'n_layer': 2, 'cells': 4, 'dropout': 0.1}
  - dev:  78.7
  - test: 78.6


# 2019-06-18 - Prevous + train vocab only (minfreq=3)
  - config: {'n_vocab': 21100, 'n_embed': 32, 'n_output': 3, 'n_epoch': 20, 'n_batch': 256, 'n_layer': 2, 'cells': 4, 'dropout': 0.1}
  - dev:  78.8
  - test: 78.4


# 2019-06-18 - Prevous + train vocab only (minfreq=4)
  - config: {'n_vocab': 17653, 'n_embed': 32, 'n_output': 3, 'n_epoch': 20, 'n_batch': 256, 'n_layer': 2, 'cells': 4, 'dropout': 0.1}
  - dev:  78.1
  - test: 78.3


# 2019-06-18 - Prevous + train vocab only (minfreq=5)
  - config: {'n_vocab': 16024, 'n_embed': 32, 'n_output': 3, 'n_epoch': 20, 'n_batch': 256, 'n_layer': 2, 'cells': 4, 'dropout': 0.1}
  - dev:  79.2
  - test: 78.7


# 2019-06-18 - Prevous + 10% discount sql len
  - config: {'n_vocab': 16024, 'n_embed': 32, 'n_output': 3, 'n_epoch': 20, 'n_batch': 256, 'n_layer': 2, 'cells': 4, 'dropout': 0.1}
  - dev:  78.7
  - test: 78.7


# 2019-06-21 - HBMP basic config
  - config: { n_vocab': 33795, 'd_embed': 300, 'd_hidden': 600, 'n_output': 3, 'n_epoch': 20, 'n_batch': 64, 'learning_rate': 0.0005, 'n_layer': 1, 'dropout': 0.1}
  - dev:  82.0
  - test: 81.6


# 2019-07-03 - DiSAN
  - config: { 'n_vocab': 33797, 'd_embed': 300, 'd_hidden': 300, 'n_output': 3, 'n_epoch': 20, 'n_batch': 64, 'learning_rate': 0.0005, 'dropout': 0.1, 'n_layer': 1, 'i_pad': 0}
  - dev: 82.087
  - test: 81.749


# 2019-07-05 - DiSAN ([s1; s2, |s1 - s2|, s1 * s2])
  - config: { 'n_vocab': 33797, 'd_embed': 300, 'd_hidden': 300, 'n_output': 3, 'n_epoch': 20, 'n_batch': 64, 'learning_rate': 0.0005, 'dropout': 0.1, 'n_layer': 1, 'i_pad': 0}
  - dev: 82.422
  - test: 82.176


# 2019-07-19 - Transformer ([s1; s2, |s1 - s2|, s1 * s2])
  - config: { 'n_enc_vocab': 33797, 'n_dec_vocab': 33797, 'n_enc_seq': 82, 'n_dec_seq': 82, 'n_layer': 6, 'd_embed': 300, 'i_pad': 0, 'd_ff': 64, 'n_heads': 4, 'd_k': 32, 'd_v': 32, 'dropout': 0.1 }
  - dev: 73.349
  - test: 73.198


# GPT (Decoder)
0.0   : epoch: 15, loss: 0.423, dev: 76.306, test: 77.372
0.5   : epoch: 18, loss: 1.932, dev: 82.920, test: 82.339
GPT2  : epoch: 20, loss: 1.907, dev: 82.260, test: 82.044
LAST  : epoch: 34, loss: 1.927, dev: 83.448, test: 83.377
FIRST : 학습안됨
MAX   : epoch: 50, loss: 1.920, dev: 83.164, test: 83.143
MEAN  : epoch: 25, loss: 2.010, dev: 81.996, test: 82.685
{'n_layer': 6, 'd_embed': 256, 'i_pad': 0, 'd_ff': 1024, 'n_heads': 8, 'd_k': 32, 'd_v': 32, n_batch: 256}
        epoch: 49, loss, 1.686, dev: 84.495, test: 84.171
{''n_layer': 6, 'd_embed': 512, 'i_pad': 0, 'd_ff': 2048, 'n_heads': 8, 'd_k': 64, 'd_v': 64,'n_batch': 192}
        epoch: 48, loss: 1.553, dev: 85.277, test: 84.884
add layer-normal
        epoch: 48, loss: 1.553, dev: 85.247, test: 84.731
        epoch: 36, loss: 1.528, dev: 85.125, test: 84.314
gelu:
        epoch: 38, loss: 1.615, dev: 84.830, test: 84.660
        epoch: 90, loss: 1.462, dev: 85.704, test: 85.088

# GPT (Encoder)
0.0  : epoch: 14, loss: 0.406, dev: 78.958, test: 79.601
0.5  : epoch: 24, loss: 0.404, dev: 81.731, test: 82.156
 
 # BERT
 No pretrain: epoch 48:, loss: 0.901, dev: 59.266, test: 59.986