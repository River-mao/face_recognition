import torch
import numpy as np
from dataset import ReadData
from Net import AlexNet

trainPath = '../../data/net/train/'
testPath = '../../data/net/test/'
paramSavepath = '../../checkpoints/alexnetParam'

if __name__ == '__main__':

    BatchSize = 16
    lr = 0.05
    Epoch = 101
    trainSet = ReadData(imgPath=trainPath)
    testSet = ReadData(imgPath=testPath)
    trainsize = len(trainSet)
    testsize = len(testSet)
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=BatchSize, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=BatchSize, shuffle=True, num_workers=1)
    criterion = torch.nn.CrossEntropyLoss()
    net = AlexNet()
    #net = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.99)

    for epoch in range(Epoch):
        net.train()
        total_loss = 0
        total_accnum = 0

        print("----------------------------train---------------------------------")
        for step, (data, label) in enumerate(trainloader):
            # print("input shape:{}".format(np.shape(data)))
            optimizer.zero_grad()
            input = torch.tensor(data, dtype=torch.float)
            label = label.clone().detach()
            label = torch.tensor(label, dtype=torch.long)

            score = net(input)
            loss = criterion(score, label)
            total_loss +=loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            score = score.data.numpy()
            label = label.data.numpy()
            predict = np.zeros(len(label))
            for i in range(len(label)):
                predict[i] = np.argmax(score[i])
            print("{}/{}. loss:{}".format(step, int(trainsize/BatchSize+1), loss))

            accnum = np.sum(predict == label)
            total_accnum += accnum

        acc = total_accnum/trainsize
        loss = total_loss / (trainsize // BatchSize)

        print("Epoch:{}/{}, loss:{}, train acc:{},".format(epoch, Epoch, loss, acc))


        if epoch % 2 ==0:
            net.eval()
            total_accnum = 0
            for step, (data, label) in enumerate(testloader):
                #input=data.clone().detach().requires_grad_(True)
                input = torch.tensor(data, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.long)
                #label = label.clone().detach()
                #label = torch.from_numpy(label)
                score = net(input)

                score = score.data.numpy()
                label = label.data.numpy()
                predict = np.zeros(len(label))
                for i in range(len(label)):
                    predict[i] = np.argmax(score[i])

                accnum = np.sum(predict == label)
                total_accnum += accnum

            accuracy = total_accnum/testsize
            if accuracy > 0.5:
                savename ='epoch_' + '_' + str(epoch)+ '_acc_' + str(accuracy)
                torch.save(net.state_dict(), savename+ "net_param.pkl")
            print("---------------------------test-----------------------------------")
            print("Epoch{}/{}, accuracy number: {}/{}, test acc {}".format(epoch, Epoch, total_accnum, testsize, accuracy))