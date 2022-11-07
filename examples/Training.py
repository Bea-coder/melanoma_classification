import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from melanoma.benig_mal.analysis import Analysis
import torch.optim as optim

from melanoma.benig_mal import ImageDataset
from melanoma.benig_mal import NetBodyParts, NetMelanoma

if __name__ == '__main__':
    # importing results
    max_files = 512 * 2
    print("Files to train {}".format(max_files))
    num_workers = 0
    batch_size = 32
    valid_size = 0.2
    n_epochs = 150
    num_train = max_files
    resolution = 512
    load_model = True
    keep_loss_file = True  # initialize start=last epoch
    start = 30
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dict_categories = {'benign': 0, 'malignant': 1}

    cuda1 = torch.device('cuda:3')
    print(cuda1)

    # Generating equally weighted results

    file_list = pd.read_csv('/data/bea/Data/siim-isic-melanoma-classification/train.csv',
                            header=0, index_col=False, usecols=[0, 1, 2, 3, 4, 5, 6, 7])
    labels, filenames = Analysis.generating_indices(file_list, 0, max_files)

    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 0}

    # Partition Training and Testing
    split = int(np.floor(valid_size * num_train))
    partition = {}
    indices = range(max_files)
    partition['train'] = indices[:split]
    partition['valid'] = indices[split:]

    # Generators
    training_set = ImageDataset(partition['train'], labels, filenames, transform=transform)
    training_generator = DataLoader(training_set, **params)

    validation_set = ImageDataset(partition['valid'], labels, filenames, transform=transform)
    validation_generator = DataLoader(validation_set, **params)

    # create a complete CNN
    modelA = NetBodyParts(resolution)
    n = 0
    for param in modelA.parameters():
        print(n, len(param))
        if n < 6:
            param.requires_grad = False
        n += 1
    modelA.load_state_dict(torch.load('model-BP.pt'))

    print(modelA)
    model = NetMelanoma(modelA, resolution)

    # load model
    if load_model is True:
        state_dict = torch.load('model_melanoma.pt')
        model.load_state_dict(state_dict)
        print("model loaded")

    print(model)

    model.cuda(cuda1)

    # Loss and Optimizer
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    filename = 'loss-celoss-adam-ben-lr-2'

    # training
    valid_loss_min = np.Inf
    if keep_loss_file == False:
        with open(filename + '.dat', "w") as f:
            line = "#train loss\tvalid_loss\n"
            f.write(line)
        start = 0
    # scheduler=lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,40], gamma=0.1)
    for epoc in range(start, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0
        accuracy = 0.0
        n = 0
        model.train()
        print("{}/{}".format(epoc, n_epochs))
        for local_data, local_labels in training_generator:
            local_data = local_data.to(cuda1)
            #        print(len(local_labels))
            local_labels = local_labels.to(cuda1)
            #        print("-->{}/{}".format(n,len(training_generator)))
            optimizer.zero_grad()
            output = model(local_data)
            loss = criterion(output, local_labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * local_data.size(0)
            n += 1
        n = 0
        # validate the model
        model.eval()
        with torch.set_grad_enabled(False):
            for local_data, local_labels in validation_generator:
                local_data, local_labels = local_data.to(cuda1), local_labels.to(cuda1)
                #            print("-->{}/{}".format(n,len(validation_generator)))
                output = model(local_data)
                loss = criterion(output, local_labels)
                valid_loss += loss.item() * local_data.size(0)

                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == local_labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                n += 1
                #            print("prediction: ".format(top_class.view(*local_labels.shape)))
                #            print("values    : ".format(local_labels))
                Analysis.confusion_matrix(top_class, local_labels, equals, len(dict_categories), cuda1)
        #            print(torch.sum(equals))
        train_loss = train_loss / (len(training_generator))
        valid_loss = valid_loss / (len(validation_generator))
        accuracy = accuracy / (len(validation_generator))
        print("train loss: {} valid loss {} {}".format(train_loss, valid_loss, accuracy))
        #   scheduler.step()
        with open(filename + '.dat', "a+") as f:
            line = "{}\t{}\t{}\t{}\n".format(epoc, train_loss, valid_loss, accuracy)
            f.write(line)

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            name = 'model_melanoma.pt'
            torch.save(model.state_dict(), name)
            valid_loss_min = valid_loss
