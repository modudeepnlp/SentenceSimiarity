import torch
import torch.nn as nn
import custom_dataset as custom_dataset
import model as model
from torch.autograd import Variable
from torch.utils.data import DataLoader
import config as config
import numpy as np

if __name__ == "__main__":
    # 데이터 처리
    dataset = custom_dataset.Custom_dataset()
    train_data, test_data = dataset.get_data()

    train_loader =  DataLoader(train_data,
                                batch_size=config.batch ,
                                shuffle=True,
                                num_workers=config.cpu_processor,
                                drop_last=True)

    test_loader =   DataLoader(test_data,
                                batch_size=config.batch,
                                shuffle=True,
                                num_workers=config.cpu_processor,
                                drop_last=True)

    # 모델 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.Classifier(dataset.vocab_size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()
    print("--model set--")

    # 훈련
    for i in range(config.epoch):
        for n, (label, sent1, sent2) in enumerate(train_loader):
            optimizer.zero_grad() # 초기화
            label = Variable(label.to(device))
            sent1 = Variable(torch.stack(sent1).to(device))
            sent2 = Variable(torch.stack(sent2).to(device))
            logit = model(sent1, sent2)

            # break
            loss = loss_function(logit, label)
            loss.backward()
            optimizer.step()

            if n % 1000 == 0:
                print(n , " loss : ", loss)

    # 평가
    total = 0
    correct = 0
    for n, (label, sent1, sent2) in enumerate(test_loader):
        label = Variable(label.to(device))
        sent1 = Variable(torch.stack(sent1).to(device))
        sent2 = Variable(torch.stack(sent2).to(device))
        out =  model(sent1, sent2)
        if n == 0:
            print(out.data)
        _, pred = torch.max(out.data, 1)
        total += label.size(0) # batch size

        print(pred)
        print(label)
        print("-------------")

        if n == 30:
            break

        correct += (pred == label).sum() 

    # correct 는 gpu 에 있는 pred 과 label 을 사용하기에 다시 cpu로 가져와야한다
    print("accuracy : " , 100 * (correct.cpu().numpy()/total), "%")
