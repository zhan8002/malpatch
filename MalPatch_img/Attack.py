# MalPatch-img patch-whitebox

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import argparse
import csv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from PIL import Image
from patch_utils import*
from utils import*

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
parser.add_argument('--train_size', type=int, default=2000, help="number of training images")
parser.add_argument('--test_size', type=int, default=2000, help="number of test images")
parser.add_argument('--pad_row', type=float, default=10, help="row of the patch size")
parser.add_argument('--probability_threshold', type=float, default=0.9, help="minimum target probability")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--max_iteration', type=int, default=100, help="max iteration")
parser.add_argument('--target', type=int, default=0, help="target label")
parser.add_argument('--epochs', type=int, default=1, help="total epoch")
parser.add_argument('--data_train_dir', type=str, default='/home/ubuntu/zhan/dataset/figshare_img/val/', help="dir of the dataset")
parser.add_argument('--data_test_dir', type=str, default='/home/ubuntu/zhan/dataset/figshare_img/test/val/', help="dir of the dataset")
args = parser.parse_args()


def patch_attack(image, pad_trans, mask, target, probability_threshold, model, lr=1, max_iteration=100):
    model.eval()
    target_probability, count = 0, 0
    g = 0
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), pad_trans.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    while target_probability < probability_threshold and count < max_iteration:
        count += 1
        # Optimize the patch
        perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        per_image = per_image.cuda()
        output = model(per_image)
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()

        g = g + patch_grad / torch.norm(patch_grad, p=1)
        pad_trans = pad_trans.type(torch.FloatTensor) + lr * torch.sign(g)

        pad_trans = torch.clamp(pad_trans, min=0, max=1)
        # Test the patch
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), pad_trans.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=0, max=1)
        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
    perturbated_image = perturbated_image.cpu().numpy()
    return perturbated_image, pad_trans

device = torch.device('cuda:0')

# Load the model
model = torch.load('./cnn_model/densenet.pth', map_location='cuda:0')
model.eval()

transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=3),
     transforms.Resize(size=(224, 224)),
     transforms.ToTensor(),
     ])

trainset= torchvision.datasets.ImageFolder(
    root= args.data_train_dir,
    transform=transform)

testset = torchvision.datasets.ImageFolder(
    root= args.data_test_dir,
    transform=transform)

# Load the datasets
train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)


# Initialize the patch
patch = np.zeros((1, 3, 224, 224))
patch = torch.from_numpy(patch)

pad_row = args.pad_row

padding_rows_set = np.linspace(10, 1, 10)
# create mask
mask = np.zeros([3, 224, 224])
mask = torch.from_numpy(mask)
for pad_row in padding_rows_set:
    best_patch_epoch, best_patch_success_rate = 0, 0
    log_train_success, log_train_single = 0, 0

    pad_row = int(pad_row)
    mask[:, :, :] = 0 # initial mask
    mask[:, -pad_row:, :] = 1
    # Generate the patch
    for epoch in range(args.epochs):
        train_total, train_actual_total, train_success = 0, 0, 0
        for (image, label) in train_loader:
            train_total += label.shape[0]
            assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            _, predicted = torch.max(output.data, 1)

            if predicted[0] == label and predicted[0].data.cpu().numpy() != args.target:
                 train_actual_total += 1

                 if train_actual_total % 100 == 0:
                     print("Training ", str(train_actual_total), " samples with", str(train_success), "succeed")

                 ori_x = np.asarray((image.cpu()).squeeze())

                 pad_image = np.pad(ori_x[0], ((0, pad_row), (0, 0)), constant_values=0)
                 pad_image = Image.fromarray(np.uint8(pad_image*255))
                 pad_trans = transform(pad_image)
                 pad_trans = pad_trans.unsqueeze(0)
                 pad_trans = pad_trans.to(device)

                 pad_trans = torch.mul(mask.type(torch.FloatTensor), patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), pad_trans.type(torch.FloatTensor))

                 perturbated_image, applied_patch = patch_attack(image, pad_trans, mask, args.target, args.probability_threshold, model, args.lr, args.max_iteration)
                 perturbated_image = torch.from_numpy(perturbated_image).cuda()
                 output = model(perturbated_image)
                 _, predicted = torch.max(output.data, 1)
                 if predicted[0].data.cpu().numpy() == args.target:
                     train_success += 1
                 patch = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor))

        print("Epoch:{} Patch attack success rate on trainset with different patch: {:.3f}%".format(epoch, 100 * train_success / train_actual_total))
        train_success_rate = test_patch(pad_row, args.target, patch, train_loader, model, mask)
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success_rate))
        test_success_rate = test_patch(pad_row, args.target, patch, test_loader, model, mask)
        print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))

        # Record the statistics
        if test_success_rate > best_patch_success_rate:
            best_patch_success_rate = test_success_rate
            best_patch_epoch = epoch
    print("Pad_row {} with success rate {}% on testset".format(pad_row, 100 * best_patch_success_rate))