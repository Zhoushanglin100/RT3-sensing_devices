'''
    This file is for the pattern pruning using the extracted representative patterns.
    This pattern pruning is based on the first-step block-structured pruning.
    That' to say, BP+PP, however, this is just one episode in the RL process.
'''

import copy
import time
import torch
import torchtext
from torchtext.data import get_tokenizer
import numpy as np
from Pruning.mnist_pattern_pruning_CPU import changed_pattern_pruning
from Generate_patterns.mnist_extract_info_from_precompression import extract_original_layers_whole_pattern
import torch.nn as nn

import sys

# #keep batch data
# def batchify(data,TEXT,device,bsz):
#     data = TEXT.numericalize([data.examples[0].text])
#     nbatch = data.size(0) // bsz # Divide the dataset into bsz parts.
#     data = data.narrow(0, 0, nbatch * bsz) #Trim off any extra elements that wouldn't cleanly fit (remainders).
#     data = data.view(bsz, -1).t().contiguous()# Evenly divide the data across the bsz batches.
#     sequence_length = nbatch * bsz
#     return data.to(device),sequence_length


# bptt = 35
# #bptt is a max sentence number of one batch
# def get_batch(source, i):
#     seq_len = min(bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
#     return data, target


# emsize = 800
# nhid = 200
# nlayers = 2
# nhead = 4
# dropout = 0.2

# # dataset--WikiText2
# TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
#                             init_token='<sos>',
#                             eos_token='<eos>',
#                             lower=True)
# train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
# TEXT.build_vocab(train_txt)

# train_data,train_length = batchify(train_txt,TEXT,device,bsz=20)
# val_data,val_length = batchify(val_txt,TEXT,device,bsz=10)
# test_data,test_length = batchify(test_txt,TEXT,device,bsz=10)

# ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
# model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)



# -------------------------------
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from LeNet_5 import Model

# batch_size = 256
# train_dataset = mnist.MNIST(root='./train', train=True, download=True, transform=ToTensor())
# test_dataset = mnist.MNIST(root='./test', train=False, download=True, transform=ToTensor())
# train_loader = DataLoader(train_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_train = datasets.MNIST('temp_nn/data/mnist/', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataset_test = datasets.MNIST('temp_nn/data/mnist/', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)


model = Model()
model.load_state_dict(torch.load("../LeNet5-MNIST-PyTorch/models/mnist_0.99.pt"))
model.to(device)

# -------------------------------

def train(model, train_loader, optimizer, criterion, scheduler, epoch, device, mask_dict_set, every_mask_whole_pattern, original_whole_pattern):

    # print("-----------------")
    # print(mask_dict_set)
    # print(every_mask_whole_pattern)
    # print(original_whole_pattern)
    # print("-----------------")

    model.train()
    start_time = time.time()
    total_loss = 0.

    for idx, (train_x, train_label) in enumerate(train_loader):
        train_x, train_label = train_x.to(device), train_label.to(device)
        optimizer.zero_grad()
        average_loss = torch.tensor(0,dtype=torch.float).to(device)
        model_replica = copy.deepcopy(model)            # deepcopy model

        for mask_j in range(len(mask_dict_set)):

            changed_pattern_pruning(model,every_mask_whole_pattern[mask_j],device)#pattern pruning
            
            predict_y = model(train_x.float())
            sub_loss = criterion(predict_y, train_label.long())

            #give different sub-loss different weight
            if mask_j == 0:
                loss = sub_loss * 0.5
            if mask_j == 1:
                loss = sub_loss * 0.3
            if mask_j == 2:
                loss = sub_loss * 0.2

            loss.backward()
            average_loss += loss
            #recovery original model
            assign_model = copy.deepcopy(model_replica)
            for pre_name, pre_weight in assign_model.named_parameters():
                for name, weight in model.named_parameters():
                    if pre_name == name:
                        weight.data = pre_weight.data
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        for update_name, update_weight in model.named_parameters():
            # if ('in_proj_weight' in update_name) or ('out_proj.weight' in update_name) \
            #         or ('linear1.weight' in update_name) or ('linear2.weight' in update_name) \
            #         or ('encoder.weight' in update_name) or ('decoder.weight' in update_name):
            if ("fc" in update_name) and ("weight" in name):
                update_weight.data = update_weight.data.mul_(original_whole_pattern[update_name].to(device))
        total_loss += average_loss.item()
        #print train process
        log_interval = 900
        if idx % log_interval == 0 and idx > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | {:5.2f} ms/batch  | '
                    'loss {:5.2f}'.format(
                    epoch, idx, len(train_loader), scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()

#model evaluate
def evaluate(eval_model, test_loader, criterion):
    correct = 0
    _sum = 0
    total_loss = 0.

    eval_model.eval()

    with torch.no_grad():
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x, test_label = test_x.to(device), test_label.to(device)
            log_probs = eval_model(test_x)
            total_loss += criterion(log_probs, test_label).item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(test_label.data.view_as(y_pred)).sum()

    # print('accuracy: {:.2f}'.format(correct / _sum))
    return total_loss / len(test_loader.dataset), correct / len(test_loader.dataset)


#compute weighted accuracy in every epoch at eval_data
def epoch_weighted_accuracy(model,mask_dict_set,every_mask_whole_pattern,device,test_loader,criterion,epoch):
    weighted_accuracy = torch.tensor(0,dtype=torch.float).to(device)
    for mask_j in range(len(mask_dict_set)):
        whole_weight_pattern = every_mask_whole_pattern[mask_j]
        #cpoy a submodel to evaluate
        sub_model = copy.deepcopy(model)
        changed_pattern_pruning(sub_model,whole_weight_pattern,device)
        # sub_loss,sub_accuracy = evaluate(bptt,sub_model,data_source,TEXT,criterion)
        sub_loss,sub_accuracy = evaluate(sub_model, test_loader, criterion)

        print('| end of epoch {:3d} | sub loss {:5.2f} | sub accuracy {:8.4f}'.format(epoch,
                                                            sub_loss, sub_accuracy))
        if mask_j == 0:
            sub_accuracy *= 0.5
        if mask_j == 1:
            sub_accuracy *= 0.3
        if mask_j == 2:
            sub_accuracy *= 0.2
        weighted_accuracy += sub_accuracy
    weighted_accuracy = weighted_accuracy.cpu().numpy()
    return weighted_accuracy


#compute weighted accuracy after fine-tune training in test_data
def eval_weighted_accuracy(model,mask_dict_set,every_mask_whole_pattern,device,test_loader,criterion):
    weighted_accuracy = torch.tensor(0,dtype=torch.float).to(device)
    level_accuracy = []
    for mask_j in range(len(mask_dict_set)):
        whole_weight_pattern = every_mask_whole_pattern[mask_j]
        # cpoy a submodel to evaluate
        sub_model = copy.deepcopy(model)
        changed_pattern_pruning(sub_model,whole_weight_pattern,device)
        # sub_loss,sub_accuracy = evaluate(bptt,sub_model,data_source,TEXT,criterion)
        sub_loss,sub_accuracy = evaluate(sub_model, test_loader, criterion)
        level_accuracy.append(sub_accuracy.cpu().numpy())#store sub accuracy
        print('| test_data evaluate | sub loss {:5.2f} | sub accuracy {:8.4f}'.format(
                                                            sub_loss, sub_accuracy))
        if mask_j == 0:
            sub_accuracy *= 0.5
        if mask_j == 1:
            sub_accuracy *= 0.3
        if mask_j == 2:
            sub_accuracy *= 0.2
        weighted_accuracy += sub_accuracy
    weighted_accuracy = weighted_accuracy.cpu().numpy()
    return weighted_accuracy,level_accuracy



def train_prune(mask_dict_set,model,epochs,every_mask_whole_pattern):
    best_accuracy= float(0)
    best_model = None

    lr = 0.23

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    original_whole_pattern = extract_original_layers_whole_pattern(model,device)

    print('-' * 89)
    # original_loss,original_accuracy = evaluate(bptt,model,test_data,TEXT,criterion)
    original_loss,original_accuracy = evaluate(model, test_loader, criterion)

    print('| original model evaluate | loss {:5.2f} | accuracy {:8.4f}'.format(
        original_loss, original_accuracy))

    print('-' * 89)
    test_accuracy,three_sub_accuracy = eval_weighted_accuracy(model,mask_dict_set,every_mask_whole_pattern,device,test_loader,criterion)
    print('| test before training | weighted_accuracy {:8.4f}'.format(test_accuracy))

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        print('-' * 89)
        train(model, train_loader, optimizer, criterion, scheduler, epoch, device, mask_dict_set, every_mask_whole_pattern, original_whole_pattern)
        weighted_accuracy = epoch_weighted_accuracy(model, mask_dict_set, every_mask_whole_pattern, device, test_loader,criterion,epoch)
        print('| end of epoch {:3d} | time: {:5.2f}s | weighted_accuracy {:8.4f}'.format(epoch, (time.time() - epoch_start_time),
                                                            weighted_accuracy))
        # save best model
        if weighted_accuracy > best_accuracy:
            best_accuracy = weighted_accuracy
            best_model = model

        scheduler.step()
    print('-' * 89)
    reward_weighted_accuracy,reward_sub_accuracy = eval_weighted_accuracy(best_model,mask_dict_set,every_mask_whole_pattern,device,test_loader,criterion)
    print('| end of training | weighted_accuracy {:8.4f}'.format(reward_weighted_accuracy))
    print('-' * 89)

    return reward_weighted_accuracy,reward_sub_accuracy