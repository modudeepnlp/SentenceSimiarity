import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

# train 데이터 path
train_data = Path.cwd() / 'data_in' / 'snli_1.0_train.txt'

# train data load (document, label 컬럼을 tab으로 구별)
data = pd.read_csv(train_data, sep='\t').loc[:, ['sentence1', 'sentence2', 'gold_label']]
# train data에서 Nan 값 지우고
#data = data.loc[data['gold_label'].isin(['-']).apply(lambda elm: not elm), :]
# train, valid 파일을 8:2로 나눔
train, eval = train_test_split(data, test_size=0.2, shuffle=True, random_state=777)
# train 데이터 tab으로 저장
train.to_csv(Path.cwd() / 'data_in' / 'train.txt', sep='\t', index=False, header=False)
# valid 데이터 tab으로 저장
eval.to_csv(Path.cwd() / 'data_in' / 'val.txt', sep='\t', index=False, header=False)

#test 데이터 path
test_data = Path.cwd() / 'data_in' / 'snli_1.0_test.txt'
# test data load (document, label 컬럼을 tab으로 구별)
data = pd.read_csv(test_data, sep='\t').loc[:, ['sentence1', 'sentence2', 'gold_label']]
# test data에서 Nan 값 지우고
#data = data.loc[data['gold_label'].isin(['-']).apply(lambda elm: not elm), :]
# test 데이터 tab으로 저장
data.to_csv(Path.cwd() / 'data_in' / 'test.txt', sep='\t', index=False, header=False)