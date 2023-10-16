# MalPatch-img patch-blackbox

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import cv2
import argparse
import csv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from PIL import Image
from patch_utils import*
from utils import*
from ga import ga_padding

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
parser.add_argument('--train_size', type=int, default=2000, help="number of training images")
parser.add_argument('--test_size', type=int, default=2000, help="number of test images")
parser.add_argument('--pad_row', type=float, default=10, help="row of the patch size")
parser.add_argument('--max_iteration', type=int, default=100, help="max iteration")
parser.add_argument('--target', type=int, default=0, help="target label, 0:benign、1:malware")
parser.add_argument('--epochs', type=int, default=3, help="total epoch")
parser.add_argument('--data_train_dir', type=str, default='', help="dir of the dataset")
parser.add_argument('--data_test_dir', type=str, default='', help="dir of the dataset")
parser.add_argument('--model_path', type=str, default='', help="path of the target model")
parser.add_argument('--log_dir', type=str, default='patch_den_log.csv', help='dir of the log')
args = parser.parse_args()

CUDA_VISIBLE_DEVICES= 1
device = torch.device('cuda:0')

# Load the model
model = torch.load(args.model_path, map_location='cuda:0')
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

padding_rows_set = np.linspace(10, 1, pad_row)
# create mask
mask = np.zeros([3, 224, 224])
mask = torch.from_numpy(mask)
for pad_row in padding_rows_set:
    best_patch_epoch, best_patch_success_rate = 0, 0
    log_train_success, log_train_single = 0, 0
    pad_row = int(pad_row)
    mask[:, :, :] = 0 # initial mask
    mask[:, -pad_row:, :] = 1
    # initial elite population
    et_pop = []
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

                 if train_actual_total%100 == 0:
                     print("Training ", str(train_actual_total), " samples with", str(train_success), "success")

                 padding_image, applied_patch, update_et_pop = ga_padding(image, net=model, et_pop=et_pop, padding_row=pad_row, target_class=0)

                 et_pop = update_et_pop
                 perturbated_image = cv2.resize(np.transpose(padding_image, (1, 2, 0)), (224, 224))
                 perturbated_image = np.transpose(perturbated_image, (2, 0, 1))
                 perturbated_image = torch.from_numpy(perturbated_image)
                 perturbated_image = perturbated_image.unsqueeze(0)

                 pre_output = model(perturbated_image.type(torch.FloatTensor).cuda())
                 _, final_predicted = torch.max(pre_output.data, 1)

                 print(str(output[0][0].data.cpu().numpy())+'  to   '+ str(pre_output[0][0].data.cpu().numpy()))
                 if final_predicted[0].data.cpu().numpy() == args.target:
                     print(str(train_actual_total)+'_attack success')
                     train_success += 1

                 applied_patch = torch.from_numpy(applied_patch)
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
            log_train_success = train_success_rate
            log_train_single = train_success / train_actual_total


    with open(args.log_dir, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([pad_row, log_train_single, log_train_success, best_patch_success_rate])
        # Load the statistics and generate the line
        log_generation(args.log_dir)

    print("Pad_row {} with success rate {}% on testset".format(pad_row, 100 * best_patch_success_rate))