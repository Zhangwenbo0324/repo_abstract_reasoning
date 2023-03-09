import os
import random
import torch
import numpy as np
import argparse
import time
import torch.nn.functional as F

from tqdm import trange
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

from dataset_utility import dataset, ToTensor

from model import ARII



parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='normal')

parser.add_argument('--fig_type', type=str, default='interpolation')
parser.add_argument('--dataset', type=str, default='pgm')
parser.add_argument('--root', type=str, default='/mnt/data/zwb/PGM/')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--seed', type=int, default=123)


# parser.add_argument('--wd', type=float, default=0.01)

parser.add_argument('--cuda', type=str, default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

tf = transforms.Compose([ToTensor()])


train_set = dataset(args.root, 'train', args.fig_type, args.img_size, tf)
valid_set = dataset(args.root, 'val', args.fig_type, args.img_size, tf)
test_set = dataset(args.root, 'test', args.fig_type, args.img_size, tf)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                          pin_memory=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)


save_name = args.model_name + '_' + args.fig_type + str(args.img_size)

save_path_model = os.path.join('/mnt/data/zwb/', args.dataset, 'models', save_name)
if not os.path.exists(save_path_model):
    os.makedirs(save_path_model)

save_path_log = os.path.join('/mnt/data/zwb/', args.dataset, 'logs')
if not os.path.exists(save_path_log):
    os.makedirs(save_path_log)


model = ARII(image_size=args.img_size).to(device)


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
#                               weight_decay=args.wd)

time_now = datetime.now().strftime('%D-%H:%M:%S')
save_log_name = os.path.join(save_path_log, 'log_{:s}.txt'.format(save_name))
with open(save_log_name, 'a') as f:
    f.write('\n------ lr: {:f}, batch_size: {:d}, img_size: {:d}, time: {:s} ------\n'.format(
        args.lr, args.batch_size, args.img_size, time_now))
f.close()



def train(epoch):
    model.train()
    metrics = {'loss': [], 'correct': [], 'count': [], 'EntropyLoss': [], 'VQLoss': [], 'ReconLoss': []}

    train_loader_iter = iter(train_loader)
    for batch_idx in trange(len(train_loader_iter)):

        image, target = next(train_loader_iter)

        image = Variable(image, requires_grad=True).to(device)
        target = Variable(target, requires_grad=False).to(device)


        logit1, logit2, vq_loss, recon_loss = model(image)

        s1 = F.softmax(logit1, dim=-1)
        s2 = F.softmax(logit2, dim=-1)
        predict = 1 / 2 * (s1 + s2)
        pred = torch.max(predict, 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()

        loss1 = F.cross_entropy(logit1, target)
        loss2 = F.cross_entropy(logit2, target)
        loss = 1 / 2 * (loss1 + loss2) + vq_loss + recon_loss
        # loss = 1 / 2 * (loss1 + loss2) * 0.1 + vq_loss * 0.1 + recon_loss * 0.8


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metrics['EntropyLoss'].append((1 / 2 * (loss1 + loss2)).item())
        metrics['VQLoss'].append(vq_loss.item())
        metrics['ReconLoss'].append(recon_loss.item())
        metrics['loss'].append(loss.item())
        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))

        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count'])

        if (batch_idx + 1) % 1000 == 0:
            print(
                'Training Epoch: {:d}/{:d}, Iteration: {:d}, Loss: {:.5f}, EntropyLoss:{:.5f}, VQLoss:{:.5f}, ReconLoss:{:.5f}, Accuracy: {:.3f} \n'.format(
                    epoch, args.epochs, batch_idx + 1, np.mean(metrics['loss']), np.mean(metrics['EntropyLoss']),
                    np.mean(metrics['VQLoss']), np.mean(metrics['ReconLoss']), accuracy))
            save_name = os.path.join(save_path_model, 'model_epoch_{:02d}_iter_{:d}.pth'.format(epoch, batch_idx+1))
            torch.save(model.state_dict(), save_name)
            metrics_val = validate(epoch, iteration=batch_idx + 1)
            metrics_test = test(epoch, iteration=batch_idx + 1)

            loss_train = np.mean(metrics['loss'])
            loss_train_Entropy = np.mean(metrics['EntropyLoss'])
            loss_train_VQ = np.mean(metrics['VQLoss'])
            loss_train_Recon = np.mean(metrics['ReconLoss'])
            acc_train = 100 * np.sum(metrics['correct']) / np.sum(metrics['count'])

            loss_val = np.mean(metrics_val['loss'])
            loss_val_Entropy = np.mean(metrics_val['EntropyLoss'])
            loss_val_VQ = np.mean(metrics_val['VQLoss'])
            loss_val_Recon = np.mean(metrics_val['ReconLoss'])
            acc_val = 100 * np.sum(metrics_val['correct']) / np.sum(metrics_val['count'])

            loss_test = np.mean(metrics_test['loss'])
            loss_test_Entropy = np.mean(metrics_test['EntropyLoss'])
            loss_test_VQ = np.mean(metrics_test['VQLoss'])
            loss_test_Recon = np.mean(metrics_test['ReconLoss'])
            acc_test = 100 * np.sum(metrics_test['correct']) / np.sum(metrics_test['count'])

            time_now = datetime.now().strftime('%H:%M:%S')
            with open(save_log_name, 'a') as f:
                f.write('Epoch {:02d}, Iteration {:d}: Accuracy: {:.3f} ({:.3f}, {:.3f})\n '
                        'TrainLoss: ({:.5f}, {:.5f}, {:.5f}, {:.5f})\n '
                        'ValLoss: ({:.5f}, {:.5f}, {:.5f}, {:.5f})\n'
                        'TestLoss: ({:.5f}, {:.5f}, {:.5f}, {:.5f}), Time: {:s}\n'.format(
                    epoch, batch_idx + 1, acc_test, acc_train, acc_val,
                    loss_train, loss_train_Entropy, loss_train_VQ, loss_train_Recon,
                    loss_val, loss_val_Entropy, loss_val_VQ, loss_val_Recon,
                    loss_test, loss_test_Entropy, loss_test_VQ, loss_test_Recon, time_now))
            f.close()
            model.train()
            print(model.training)


    print(
        'Training Epoch: {:d}/{:d}, Iteration: all, Loss: {:.5f}, EntropyLoss:{:.5f}, VQLoss:{:.5f}, ReconLoss:{:.5f}, Accuracy: {:.3f} \n'.format(
            epoch, args.epochs, np.mean(metrics['loss']), np.mean(metrics['EntropyLoss']), np.mean(metrics['VQLoss']),
            np.mean(metrics['ReconLoss']), accuracy))

    return metrics


def validate(epoch, iteration=None):
    model.eval()

    metrics = {'loss': [], 'correct': [], 'count': [], 'EntropyLoss': [], 'VQLoss': [], 'ReconLoss': []}

    valid_loader_iter = iter(valid_loader)
    for _ in trange(len(valid_loader_iter)):

        image, target = next(valid_loader_iter)


        image = Variable(image, requires_grad=True).to(device)
        target = Variable(target, requires_grad=False).to(device)


        with torch.no_grad():
            logit1, logit2, vq_loss, recon_loss = model(image)

        s1 = F.softmax(logit1, dim=-1)
        s2 = F.softmax(logit2, dim=-1)
        predict = 1 / 2 * (s1 + s2)
        pred = torch.max(predict, 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()

        loss1 = F.cross_entropy(logit1, target)
        loss2 = F.cross_entropy(logit2, target)

        loss = 1 / 2 * (loss1 + loss2) + vq_loss + recon_loss
        # loss = 1 / 2 * (loss1 + loss2) * 0.1 + vq_loss * 0.1 + recon_loss * 0.8


        metrics['EntropyLoss'].append((1 / 2 * (loss1 + loss2)).item())
        metrics['VQLoss'].append(vq_loss.item())
        metrics['ReconLoss'].append(recon_loss.item())
        metrics['loss'].append(loss.item())
        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))

        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count'])


    if iteration:
        print(
            'Validation Epoch: {:d}/{:d}, Iteration: {:d}, Loss: {:.5f}, EntropyLoss:{:.5f}, VQLoss:{:.5f}, ReconLoss:{:.5f}, Accuracy: {:.3f} \n'.format(
                epoch, args.epochs, iteration, np.mean(metrics['loss']), np.mean(metrics['EntropyLoss']),
                np.mean(metrics['VQLoss']), np.mean(metrics['ReconLoss']), accuracy))
    else:
        print(
            'Validation Epoch: {:d}/{:d}, Iteration: all, Loss: {:.5f}, EntropyLoss:{:.5f}, VQLoss:{:.5f}, ReconLoss:{:.5f}, Accuracy: {:.3f} \n'.format(
                epoch, args.epochs, np.mean(metrics['loss']), np.mean(metrics['EntropyLoss']),
                np.mean(metrics['VQLoss']), np.mean(metrics['ReconLoss']), accuracy))

    return metrics


def test(epoch, iteration=None):
    model.eval()
    metrics = {'loss': [], 'correct': [], 'count': [], 'EntropyLoss': [], 'VQLoss': [], 'ReconLoss': []}

    test_loader_iter = iter(test_loader)
    for _ in trange(len(test_loader_iter)):

        image, target = next(test_loader_iter)


        image = Variable(image, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)


        with torch.no_grad():
            logit1, logit2, vq_loss, recon_loss = model(image)

        s1 = F.softmax(logit1, dim=-1)
        s2 = F.softmax(logit2, dim=-1)
        predict = 1 / 2 * (s1 + s2)
        pred = torch.max(predict, 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()

        loss1 = F.cross_entropy(logit1, target)
        loss2 = F.cross_entropy(logit2, target)
        loss = 1 / 2 * (loss1 + loss2) + vq_loss + recon_loss
        # loss = 1 / 2 * (loss1 + loss2) * 0.1 + vq_loss * 0.1 + recon_loss * 0.8



        metrics['EntropyLoss'].append((1 / 2 * (loss1 + loss2)).item())
        metrics['VQLoss'].append(vq_loss.item())
        metrics['ReconLoss'].append(recon_loss.item())
        metrics['loss'].append(loss.item())
        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))

        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count'])

    if iteration:
        print(
            'Testing Epoch: {:d}/{:d}, Iteration: {:d}, Loss: {:.5f}, EntropyLoss:{:.5f}, VQLoss:{:.5f}, ReconLoss:{:.5f}, Accuracy: {:.3f} \n'.format(
                epoch, args.epochs, iteration, np.mean(metrics['loss']), np.mean(metrics['EntropyLoss']),
                np.mean(metrics['VQLoss']), np.mean(metrics['ReconLoss']), accuracy))
    else:
        print(
            'Testing Epoch: {:d}/{:d}, Iteration: all, Loss: {:.5f}, EntropyLoss:{:.5f}, VQLoss:{:.5f}, ReconLoss:{:.5f}, Accuracy: {:.3f} \n'.format(
                epoch, args.epochs, np.mean(metrics['loss']), np.mean(metrics['EntropyLoss']),
                np.mean(metrics['VQLoss']), np.mean(metrics['ReconLoss']), accuracy))

    return metrics


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        metrics_train = train(epoch)
        metrics_val = validate(epoch)
        metrics_test = test(epoch)

        # Save model
        if epoch > 0:
            save_name = os.path.join(save_path_model, 'model_epoch_{:02d}_iter_all.pth'.format(epoch))
            torch.save(model.state_dict(), save_name)

        loss_train = np.mean(metrics_train['loss'])
        loss_train_Entropy = np.mean(metrics_train['EntropyLoss'])
        loss_train_VQ = np.mean(metrics_train['VQLoss'])
        loss_train_Recon = np.mean(metrics_train['ReconLoss'])
        acc_train = 100 * np.sum(metrics_train['correct']) / np.sum(metrics_train['count'])

        loss_val = np.mean(metrics_val['loss'])
        loss_val_Entropy = np.mean(metrics_val['EntropyLoss'])
        loss_val_VQ = np.mean(metrics_val['VQLoss'])
        loss_val_Recon = np.mean(metrics_val['ReconLoss'])
        acc_val = 100 * np.sum(metrics_val['correct']) / np.sum(metrics_val['count'])

        loss_test = np.mean(metrics_test['loss'])
        loss_test_Entropy = np.mean(metrics_test['EntropyLoss'])
        loss_test_VQ = np.mean(metrics_test['VQLoss'])
        loss_test_Recon = np.mean(metrics_test['ReconLoss'])
        acc_test = 100 * np.sum(metrics_test['correct']) / np.sum(metrics_test['count'])

        time_now = datetime.now().strftime('%H:%M:%S')
        with open(save_log_name, 'a') as f:
            f.write('Epoch {:02d}, Iteration all: Accuracy: {:.3f} ({:.3f}, {:.3f})\n '
                    'TrainLoss: ({:.5f}, {:.5f}, {:.5f}, {:.5f})\n '
                    'ValLoss: ({:.5f}, {:.5f}, {:.5f}, {:.5f})\n'
                    'TestLoss: ({:.5f}, {:.5f}, {:.5f}, {:.5f}), Time: {:s}\n'.format(
                epoch, acc_test, acc_train, acc_val,
                loss_train, loss_train_Entropy, loss_train_VQ, loss_train_Recon,
                loss_val, loss_val_Entropy, loss_val_VQ, loss_val_Recon,
                loss_test, loss_test_Entropy, loss_test_VQ, loss_test_Recon, time_now))
        f.close()

