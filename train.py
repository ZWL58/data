import torch
import torch.nn as nn
import Model as Mod
import Dataloader as Data
from matplotlib import pyplot as plt
import os.path as osp
import datetime
import numpy as np
import os

# 固定随机种子，保证实验结果是可以复现的
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# 加注意力机制后对比四种情况曲线
def show_contrast_loss_acc(acc, loss, sava_dir):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='CNN-SK_acc')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='dashed', alpha=0.5)
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('acc contrast')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='CNN-SK_loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='dashed', alpha=0.5)
    plt.ylabel('Cross Entropy')
    plt.title('loss contrast')
    plt.xlabel('epoch')
    # 保存在savedir目录下。
    save_path = osp.join(sava_dir, "CNN-SK.png")
    plt.savefig(save_path, dpi=300)

# ----------------------------
# Training Loop
# ----------------------------
def training_12(Model_12, train_dl, num_epochs, num_class):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model_12.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    # Repeat for each epoch
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    train_real_labels = []
    train_pre_labels = []

    # Repeat for each batch in the training set
    for i in range(1):
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            # c = datetime.datetime.now()
            # d = (c - starttime).seconds
            # print(d, end='')
            Model_12.to('cuda:0')

            inputs, labels = data[0].to(device), data[1].to(device)
            label = labels.cpu().numpy()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()    # 计算张量的平均值和标准差
            inputs = (inputs - inputs_m) / inputs_s    # torch.Size([16, 2, 64, 344])

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + back
            outputs_SKAttention = Model_12(inputs)       # torch.Size([16, 3])

            y_pred = torch.softmax(outputs_SKAttention, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()


            loss = criterion(outputs_SKAttention, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs_SKAttention, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            # Print stats at the end of the epoch
            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            train_real_labels.extend(label)
            train_pre_labels.extend(y_pred)

            print("\rEpoch: {:d}    Train_12:batch: {:d}           loss: {:.4f}            acc: {:.4f} | {:.2%}"
                  .format(epoch+1, num_epochs,avg_loss, acc, (i+1)*1.0/num_batches), end='', flush=True)
        print("\rEpoch: {:d}/{:d}       Train_12:loss: {:.4f}        acc: {:.4f}"
              .format(epoch + 1, num_epochs, avg_loss, acc), end='\n')

        break

    train_12_acc.append(acc)
    train_12_loss.append(avg_loss)

    return train_12_acc, train_12_loss

def training_23(Model_23, train_dl, num_epochs, num_class):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model_23.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    # Repeat for each epoch
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    train_real_labels = []
    train_pre_labels = []
    # Repeat for each batch in the training set
    for i in range(1):
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            # c = datetime.datetime.now()
            # d = (c - starttime).seconds
            # print(d, end='')
            Model_23.to('cuda:0')

            inputs, labels = data[0].to(device), data[1].to(device)
            label = labels.cpu().numpy()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()    # 计算张量的平均值和标准差
            inputs = (inputs - inputs_m) / inputs_s    # torch.Size([16, 2, 64, 344])

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + back
            outputs_SKAttention = Model_23(inputs)       # torch.Size([16, 3])

            y_pred = torch.softmax(outputs_SKAttention, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()


            loss = criterion(outputs_SKAttention, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs_SKAttention, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            # Print stats at the end of the epoch
            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            train_real_labels.extend(label)
            train_pre_labels.extend(y_pred)

            print("\rEpoch: {:d}    Train_23:batch: {:d}           loss: {:.4f}            acc: {:.4f} | {:.2%}"
                  .format(epoch+1, num_epochs,avg_loss, acc, (i+1)*1.0/num_batches), end='', flush=True)
        print("\rEpoch: {:d}/{:d}       Train_23:loss: {:.4f}        acc: {:.4f}"
              .format(epoch + 1, num_epochs, avg_loss, acc), end='\n')

        break

    train_23_acc.append(acc)
    train_23_loss.append(avg_loss)

    return train_23_acc, train_23_loss

def training_34(Model_34, train_dl, num_epochs, num_class):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model_34.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    # Repeat for each epoch
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    train_real_labels = []
    train_pre_labels = []
    # Repeat for each batch in the training set
    for i in range(1):
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            # c = datetime.datetime.now()
            # d = (c - starttime).seconds
            # print(d, end='')
            Model_34.to('cuda:0')

            inputs, labels = data[0].to(device), data[1].to(device)
            label = labels.cpu().numpy()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()    # 计算张量的平均值和标准差
            inputs = (inputs - inputs_m) / inputs_s    # torch.Size([16, 2, 64, 344])

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + back
            outputs_SKAttention = Model_34(inputs)       # torch.Size([16, 3])

            y_pred = torch.softmax(outputs_SKAttention, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()


            loss = criterion(outputs_SKAttention, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs_SKAttention, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            # Print stats at the end of the epoch
            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            train_real_labels.extend(label)
            train_pre_labels.extend(y_pred)

            print("\rEpoch: {:d}    Train_34:batch: {:d}           loss: {:.4f}            acc: {:.4f} | {:.2%}"
                  .format(epoch+1, num_epochs,avg_loss, acc, (i+1)*1.0/num_batches), end='', flush=True)
        print("\rEpoch: {:d}/{:d}       Train_34:loss: {:.4f}        acc: {:.4f}"
              .format(epoch + 1, num_epochs, avg_loss, acc), end='\n')

        break

    train_34_acc.append(acc)
    train_34_loss.append(avg_loss)

    return train_34_acc, train_34_loss

def training_45(Model_45, train_dl, num_epochs, num_class):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model_45.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    # Repeat for each epoch
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    train_real_labels = []
    train_pre_labels = []
    # Repeat for each batch in the training set
    for i in range(1):
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            # c = datetime.datetime.now()
            # d = (c - starttime).seconds
            # print(d, end='')
            Model_45.to('cuda:0')

            inputs, labels = data[0].to(device), data[1].to(device)
            label = labels.cpu().numpy()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()    # 计算张量的平均值和标准差
            inputs = (inputs - inputs_m) / inputs_s    # torch.Size([16, 2, 64, 344])

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + back
            outputs_SKAttention = Model_45(inputs)       # torch.Size([16, 3])

            y_pred = torch.softmax(outputs_SKAttention, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()


            loss = criterion(outputs_SKAttention, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs_SKAttention, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            # Print stats at the end of the epoch
            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            train_real_labels.extend(label)
            train_pre_labels.extend(y_pred)

            print("\rEpoch: {:d}    Train_6:batch: {:d}           loss: {:.4f}            acc: {:.4f} | {:.2%}"
                  .format(epoch+1, num_epochs,avg_loss, acc, (i+1)*1.0/num_batches), end='', flush=True)
        print("\rEpoch: {:d}/{:d}       Train_6:loss: {:.4f}        acc: {:.4f}"
              .format(epoch + 1, num_epochs, avg_loss, acc), end='\n')

        break

    train_45_acc.append(acc)
    train_45_loss.append(avg_loss)

    return train_45_acc, train_45_loss

def training_56(Model_56, train_dl, num_epochs, num_class):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model_56.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    # Repeat for each epoch
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    train_real_labels = []
    train_pre_labels = []
    # Repeat for each batch in the training set
    for i in range(1):
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            # c = datetime.datetime.now()
            # d = (c - starttime).seconds
            # print(d, end='')
            Model_56.to('cuda:0')

            inputs, labels = data[0].to(device), data[1].to(device)
            label = labels.cpu().numpy()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()    # 计算张量的平均值和标准差
            inputs = (inputs - inputs_m) / inputs_s    # torch.Size([16, 2, 64, 344])

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + back
            outputs_SKAttention = Model_56(inputs)       # torch.Size([16, 3])

            y_pred = torch.softmax(outputs_SKAttention, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()


            loss = criterion(outputs_SKAttention, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs_SKAttention, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            # Print stats at the end of the epoch
            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            train_real_labels.extend(label)
            train_pre_labels.extend(y_pred)

            print("\rEpoch: {:d}    Train_6:batch: {:d}           loss: {:.4f}            acc: {:.4f} | {:.2%}"
                  .format(epoch+1, num_epochs,avg_loss, acc, (i+1)*1.0/num_batches), end='', flush=True)
        print("\rEpoch: {:d}/{:d}       Train_6:loss: {:.4f}        acc: {:.4f}"
              .format(epoch + 1, num_epochs, avg_loss, acc), end='\n')

        break

    train_56_acc.append(acc)
    train_56_loss.append(avg_loss)

    return train_56_acc, train_56_loss

def training_no(Model_no, train_dl, num_epochs, num_class):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model_no.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    # Repeat for each epoch
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    train_real_labels = []
    train_pre_labels = []
    # Repeat for each batch in the training set
    for i in range(1):
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            # c = datetime.datetime.now()
            # d = (c - starttime).seconds
            # print(d, end='')
            Model_no.to('cuda:0')

            inputs, labels = data[0].to(device), data[1].to(device)
            label = labels.cpu().numpy()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()    # 计算张量的平均值和标准差
            inputs = (inputs - inputs_m) / inputs_s    # torch.Size([16, 2, 64, 344])

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + back
            outputs = Model_no(inputs)       # torch.Size([16, 3])

            y_pred = torch.softmax(outputs, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()


            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            # Print stats at the end of the epoch
            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            train_real_labels.extend(label)
            train_pre_labels.extend(y_pred)

            print("\rEpoch: {:d}    Train_no:batch: {:d}           loss: {:.4f}            acc: {:.4f} | {:.2%}"
                  .format(epoch+1, num_epochs,avg_loss, acc, (i+1)*1.0/num_batches), end='', flush=True)
        print("\rEpoch: {:d}/{:d}       Train_no:loss: {:.4f}        acc: {:.4f}"
              .format(epoch + 1, num_epochs, avg_loss, acc), end='\n')

        break

    train_no_acc.append(acc)
    train_no_loss.append(avg_loss)

    return train_no_acc, train_no_loss


# ----------------------------
# Valing Loop
# ----------------------------
# 注意力机制对比
def Valing_12(Model_12, val_dl, num_epochs, num_class):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model_12.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(val_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    val_real_labels4 = []
    val_pre_labels4 = []

    # Repeat for each batch in the training set
    for i in range(1):
        for i, data in enumerate(val_dl):
            # Get the input features and target labels, and put them on the GPU
            Model_12.to('cuda:0')

            inputs, labels = data[0].to(device), data[1].to(device)
            label = labels.cpu().numpy()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs_SKAttention = Model_12(inputs)

            y_pred = torch.softmax(outputs_SKAttention, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            loss = criterion(outputs_SKAttention, labels)

            loss.backward()

            optimizer.step()
            scheduler.step()

            num_batches = len(val_dl)

            # Keep stats for Loss and Accuracy
            # Get the predicted class with the highest score

            running_loss += loss.item()
            _, prediction = torch.max(outputs_SKAttention, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            avg_loss4 = running_loss / num_batches
            acc4 = correct_prediction / total_prediction
            val_real_labels4.extend(label)
            val_pre_labels4.extend(y_pred)



            print("\rEpoch: {:d}    Val_12:batch: {:d}            loss: {:.4f}            acc: {:.4f} |   {:.2%}"
                  .format(epoch + 1, num_epochs, avg_loss4, acc4, (i + 1) * 1.0 / num_batches), end='', flush=True)
        print("\rEpoch: {:d}/{:d}       Val_12:loss: {:.4f}        acc: {:.4f}"
              .format(epoch + 1, num_epochs, avg_loss4, acc4), end='\n')

        break

    val_accs.append(acc4)
    val_loss.append(avg_loss4)

    return val_accs, val_loss

def Valing_23(Model_23, val_dl, num_epochs, num_class):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model_23.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(val_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    val_real_labels4 = []
    val_pre_labels4 = []

    # Repeat for each batch in the training set
    for i in range(1):
        for i, data in enumerate(val_dl):
            # Get the input features and target labels, and put them on the GPU
            Model_23.to('cuda:0')

            inputs, labels = data[0].to(device), data[1].to(device)
            label = labels.cpu().numpy()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs_SKAttention = Model_23(inputs)

            y_pred = torch.softmax(outputs_SKAttention, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            loss = criterion(outputs_SKAttention, labels)

            loss.backward()

            optimizer.step()
            scheduler.step()

            num_batches = len(val_dl)

            # Keep stats for Loss and Accuracy
            # Get the predicted class with the highest score

            running_loss += loss.item()
            _, prediction = torch.max(outputs_SKAttention, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            avg_loss4 = running_loss / num_batches
            acc4 = correct_prediction / total_prediction
            val_real_labels4.extend(label)
            val_pre_labels4.extend(y_pred)

            # # if i % 10 == 0:    # print every 10 mini-batches
            # #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            print("\rEpoch: {:d}    Val_23:batch: {:d}            loss: {:.4f}            acc: {:.4f} |   {:.2%}"
                  .format(epoch + 1, num_epochs, avg_loss4, acc4, (i + 1) * 1.0 / num_batches), end='', flush=True)
        print("\rEpoch: {:d}/{:d}       Val_23:loss: {:.4f}        acc: {:.4f}"
              .format(epoch + 1, num_epochs, avg_loss4, acc4), end='\n')
        break

    val_accs1.append(acc4)
    val_loss1.append(avg_loss4)

    return val_accs1, val_loss1

def Valing_34(Model_34, val_dl, num_epochs, num_class):
    criterion = nn.CrossEntropyLoss()
    optimizer_VGG = torch.optim.Adam(Model_34.parameters(), lr=0.001)
    scheduler_VGG = torch.optim.lr_scheduler.OneCycleLR(optimizer_VGG, max_lr=0.001,
                                                    steps_per_epoch=int(len(val_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    val_real_labels = []
    val_pre_labels = []

    # Repeat for each batch in the training set
    for i in range(1):
        for i, data in enumerate(val_dl):
            # Get the input features and target labels, and put them on the GPU
            Model_34.to('cuda:0')
            torch.cuda.empty_cache()

            inputs, labels = data[0].to(device), data[1].to(device)
            label = labels.cpu().numpy()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer_VGG.zero_grad()

            # forward + backward + optimize
            outputs_SKAttention = Model_34(inputs)

            y_pred = torch.softmax(outputs_SKAttention, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            loss = criterion(outputs_SKAttention, labels)

            loss.backward()

            optimizer_VGG.step()
            scheduler_VGG.step()

            num_batches = len(val_dl)

            # Keep stats for Loss and Accuracy
            # Get the predicted class with the highest score

            running_loss += loss.item()
            _, prediction = torch.max(outputs_SKAttention, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            val_real_labels.extend(label)
            val_pre_labels.extend(y_pred)


            # # if i % 10 == 0:    # print every 10 mini-batches
            # #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            print("\rEpoch: {:d}    Val_34:batch: {:d}            loss: {:.4f}            acc: {:.4f} |   {:.2%}"
                  .format(epoch + 1, num_epochs, avg_loss, acc, (i + 1) * 1.0 / num_batches), end='', flush=True)
        print("\rEpoch: {:d}/{:d}       Val_34:loss: {:.4f}        acc: {:.4f}"
              .format(epoch + 1, num_epochs, avg_loss, acc), end='\n')
        break

    val_accs2.append(acc)
    val_loss2.append(avg_loss)

    return val_accs2, val_loss2

def Valing_45(Model_45, val_dl, num_epochs, num_class):
    criterion = nn.CrossEntropyLoss()
    optimizer_AlexNet = torch.optim.Adam(Model_45.parameters(), lr=0.001)       # to do
    scheduler_AlexNet = torch.optim.lr_scheduler.OneCycleLR(optimizer_AlexNet, max_lr=0.001,             # to do
                                                    steps_per_epoch=int(len(val_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    val_real_labels = []
    val_pre_labels = []

    # Repeat for each batch in the training set
    for i in range(1):
        for i, data in enumerate(val_dl):
            # Get the input features and target labels, and put them on the GPU
            Model_45.to('cuda:0')                  # to do
            torch.cuda.empty_cache()

            inputs, labels = data[0].to(device), data[1].to(device)
            label = labels.cpu().numpy()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer_AlexNet.zero_grad()                    # to do

            # forward + backward + optimize
            outputs_SKAttention = Model_45(inputs)                       # to do

            y_pred = torch.softmax(outputs_SKAttention, dim=1)               # to do
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            loss = criterion(outputs_SKAttention, labels)                     # to do

            loss.backward()

            optimizer_AlexNet.step()                            # to do
            scheduler_AlexNet.step()                              # to do

            num_batches = len(val_dl)

            # Keep stats for Loss and Accuracy
            # Get the predicted class with the highest score

            running_loss += loss.item()
            _, prediction = torch.max(outputs_SKAttention, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            val_real_labels.extend(label)
            val_pre_labels.extend(y_pred)


            # # if i % 10 == 0:    # print every 10 mini-batches
            # #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            print("\rEpoch: {:d}    Val_6:batch: {:d}            loss: {:.4f}            acc: {:.4f} |   {:.2%}"
                  .format(epoch + 1, num_epochs, avg_loss, acc, (i + 1) * 1.0 / num_batches), end='', flush=True)
        print("\rEpoch: {:d}/{:d}       Val_45:loss: {:.4f}        acc: {:.4f}"
              .format(epoch + 1, num_epochs, avg_loss, acc), end='\n')                       # to do
        break

    val_accs3.append(acc)                             # to do
    val_loss3.append(avg_loss)                         # to do

    return val_accs3, val_loss3                        # to do

def Valing_56(Model_56, val_dl, num_epochs, num_class):
    criterion = nn.CrossEntropyLoss()
    optimizer_AlexNet = torch.optim.Adam(Model_56.parameters(), lr=0.001)       # to do
    scheduler_AlexNet = torch.optim.lr_scheduler.OneCycleLR(optimizer_AlexNet, max_lr=0.001,             # to do
                                                    steps_per_epoch=int(len(val_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    val_real_labels = []
    val_pre_labels = []

    # Repeat for each batch in the training set
    for i in range(1):
        for i, data in enumerate(val_dl):
            # Get the input features and target labels, and put them on the GPU
            Model_56.to('cuda:0')                  # to do
            torch.cuda.empty_cache()

            inputs, labels = data[0].to(device), data[1].to(device)
            label = labels.cpu().numpy()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer_AlexNet.zero_grad()                    # to do

            # forward + backward + optimize
            outputs_SKAttention = Model_56(inputs)                       # to do outputs, outputs_SKAttention, outputs_TripletAttention, outputs_ShuffleAttention

            y_pred = torch.softmax(outputs_SKAttention, dim=1)               # to do
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            loss = criterion(outputs_SKAttention, labels)                     # to do

            loss.backward()

            optimizer_AlexNet.step()                            # to do
            scheduler_AlexNet.step()                              # to do

            num_batches = len(val_dl)

            # Keep stats for Loss and Accuracy
            # Get the predicted class with the highest score

            running_loss += loss.item()
            _, prediction = torch.max(outputs_SKAttention, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            val_real_labels.extend(label)
            val_pre_labels.extend(y_pred)


            # # if i % 10 == 0:    # print every 10 mini-batches
            # #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            print("\rEpoch: {:d}    Val_6:batch: {:d}            loss: {:.4f}            acc: {:.4f} |   {:.2%}"
                  .format(epoch + 1, num_epochs, avg_loss, acc, (i + 1) * 1.0 / num_batches), end='', flush=True)
        print("\rEpoch: {:d}/{:d}       Val_56:loss: {:.4f}        acc: {:.4f}"
              .format(epoch + 1, num_epochs, avg_loss, acc), end='\n')                       # to do
        break

    val_accs4.append(acc)                             # to do
    val_loss4.append(avg_loss)                         # to do

    return val_accs4, val_loss4                        # to do

def Valing_no(Model_no, val_dl, num_epochs, num_class):
    criterion = nn.CrossEntropyLoss()
    optimizer_AlexNet = torch.optim.Adam(Model_no.parameters(), lr=0.001)       # to do
    scheduler_AlexNet = torch.optim.lr_scheduler.OneCycleLR(optimizer_AlexNet, max_lr=0.001,             # to do
                                                    steps_per_epoch=int(len(val_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    val_real_labels = []
    val_pre_labels = []

    # Repeat for each batch in the training set
    for i in range(1):
        for i, data in enumerate(val_dl):
            # Get the input features and target labels, and put them on the GPU
            Model_no.to('cuda:0')                  # to do
            torch.cuda.empty_cache()

            inputs, labels = data[0].to(device), data[1].to(device)
            label = labels.cpu().numpy()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer_AlexNet.zero_grad()                    # to do

            # forward + backward + optimize
            outputs = Model_no(inputs)                       # to do

            y_pred = torch.softmax(outputs, dim=1)               # to do
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            loss = criterion(outputs, labels)                     # to do

            loss.backward()

            optimizer_AlexNet.step()                            # to do
            scheduler_AlexNet.step()                              # to do

            num_batches = len(val_dl)

            # Keep stats for Loss and Accuracy
            # Get the predicted class with the highest score

            running_loss += loss.item()
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            val_real_labels.extend(label)
            val_pre_labels.extend(y_pred)


            # # if i % 10 == 0:    # print every 10 mini-batches
            # #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            print("\rEpoch: {:d}    Val_no:batch: {:d}            loss: {:.4f}            acc: {:.4f} |   {:.2%}"
                  .format(epoch + 1, num_epochs, avg_loss, acc, (i + 1) * 1.0 / num_batches), end='', flush=True)
        print("\rEpoch: {:d}/{:d}       Val_no:loss: {:.4f}        acc: {:.4f}"
              .format(epoch + 1, num_epochs, avg_loss, acc), end='\n')                       # to do
        break

    val_accs5.append(acc)                             # to do
    val_loss5.append(avg_loss)                         # to do

    return val_accs5, val_loss5                        # to do

if __name__=='__main__':
    sava_dir = r'path'

    train_dl, val_dl, num_class = Data.Dataloader()
    Model_12 = Mod.AudioClassifier12()
    Model_23 = Mod.AudioClassifier23()
    Model_34 = Mod.AudioClassifier34()
    Model_45 = Mod.AudioClassifier45()
    Model_56 = Mod.AudioClassifier56()
    Model_no = Mod.AudioClassifier()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 70  # Just for demo, adjust this higher.
    train_12_acc = []
    train_12_loss = []
    train_23_acc = []
    train_23_loss = []
    train_34_acc = []
    train_34_loss = []
    train_45_acc = []
    train_45_loss = []
    train_56_acc = []
    train_56_loss = []
    train_no_acc = []
    train_no_loss = []

    val_accs = []
    val_loss = []
    val_accs1 = []
    val_loss1 = []
    val_accs2 = []
    val_loss2 = []
    val_accs3 = []
    val_loss3 = []
    val_accs4 = []
    val_loss4 = []
    val_accs5 = []
    val_loss5 = []

    print("Begin train")
    starttime = datetime.datetime.now()
    # Loss Function, Optimizer and Scheduler
    for epoch in range(num_epochs):
        train_12_acc, train_12_loss = training_12(Model_12, train_dl, num_epochs, num_class)
        # train_23_acc, train_23_loss = training_23(Model_23, train_dl, num_epochs, num_class)
        # train_34_acc, train_34_loss = training_34(Model_34, train_dl, num_epochs, num_class)
        # train_45_acc, train_45_loss = training_45(Model_45, train_dl, num_epochs, num_class)
        # train_56_acc, train_56_loss = training_56(Model_56, train_dl, num_epochs, num_class)
        # train_no_acc, train_no_loss = training_no(Model_no, train_dl, num_epochs, num_class)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        val_accs, val_losss = Valing_12(Model_12, val_dl, num_epochs, num_class)
        # val_accs1, val_loss1 = Valing_23(Model_23, val_dl, num_epochs, num_class)
        # val_accs2, val_loss2 = Valing_34(Model_34, val_dl, num_epochs, num_class)
        # val_accs3, val_loss3 = Valing_45(Model_45, val_dl, num_epochs, num_class)
        # val_accs4, val_loss4 = Valing_56(Model_56, val_dl, num_epochs, num_class)
        # val_accs5, val_loss5 = Valing_no(Model_no, val_dl, num_epochs, num_class)

        torch.save(Model_12, "weight/CNN-SK-{}.pth".format(epoch))

    show_contrast_loss_acc(val_accs, val_losss, sava_dir)

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print(time)
    print('Finished Training')