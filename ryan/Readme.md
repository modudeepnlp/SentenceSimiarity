## Sentence Simiarity

## MALSTM

549367/549367 [==============================] - 18s 33us/sample - loss: 0.4061 - acc: 0.8473 - val_loss: 0.6200 - val_acc: 0.7748
9824/9824 [==============================] - 0s 8us/sample - loss: 0.5542 - acc: 0.7845
Test loss / test accuracy = 0.5542 / 0.7845

Train Acc: 0.8473
Val Acc: 0.7748
Test Acc: 0.7845

Epoch 12/42
549367/549367 [==============================] - 18s 34us/sample - loss: 0.3835 - acc: 0.8568 - val_loss: 0.5713 - val_acc: 0.7899
9824/9824 [==============================] - 0s 7us/sample - loss: 0.5499 - acc: 0.7906
Test loss / test accuracy = 0.5499 / 0.7906

Train Acc: 0.8568
Val Acc: 0.7899
Test Acc: 0.7906

## HBMP (Hierarchical BiLSTM max pooling)

LSTM (W/O: Glove):
549367/549367 [==============================] - 37s 68us/sample - loss: 0.4006 - acc: 0.8444 - val_loss: 0.6272 - val_acc: 0.7727
9824/9824 [==============================] - 0s 24us/sample - loss: 0.6041 - acc: 0.7629
Test loss / test accuracy = 0.6041 / 0.7629

LSTM (W/Glove):
549367/549367 [==============================] - 38s 69us/sample - loss: 0.3747 - acc: 0.8558 - val_loss: 0.5559 - val_acc: 0.7974
9824/9824 [==============================] - 0s 24us/sample - loss: 0.5510 - acc: 0.7911
Test loss / test accuracy = 0.5510 / 0.7911

MAXPOOLINGLSTM:
(W/Glove)
549367/549367 [==============================] - 73s 132us/sample - loss: 0.4511 - acc: 0.8280 - val_loss: 0.6033 - val_acc: 0.7620
9824/9824 [==============================] - 0s 46us/sample - loss: 0.5928 - acc: 0.7673
Test loss / test accuracy = 0.5928 / 0.7673

HBMP:
(W/O Glove)
549367/549367 [==============================] - 293s 534us/sample - loss: 0.4033 - acc: 0.8473 - val_loss: 0.5672 - val_acc: 0.7834
Test loss / test accuracy = 0.5753 / 0.7724

(W/Glove)
549367/549367 [==============================] - 295s 537us/sample - loss: 0.3187 - acc: 0.8856 - val_loss: 0.5279 - val_acc: 0.8035
9824/9824 [==============================] - 2s 206us/sample - loss: 0.5123 - acc: 0.8033
Test loss / test accuracy = 0.5123 / 0.8033