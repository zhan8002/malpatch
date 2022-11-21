# Adversarial Patch: patch_utils
# utils for patch initialization and mask generation
# Created by Junbo Zhao 2020/3/19

import numpy as np
import torch
from PIL import Image
from utils import*
import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device('cuda:0')

transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=3),
     transforms.Resize(size=(224, 224)),
     transforms.ToTensor(),
     ])

# Initialize the patch
# TODO: Add circle type
def patch_initialization(patch_type='rectangle', image_size=(3, 224, 224), noise_percentage=0.03):
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

# Generate the mask and apply the patch
# TODO: Add circle type
def mask_generation(mask_type='rectangle', patch=None, image_size=(3, 224, 224)):
    applied_patch = np.zeros(image_size)
    if mask_type == 'rectangle':
        # patch rotation
        # rotation_angle = np.random.choice(4)
        # for i in range(patch.shape[0]):
        #     patch[i] = np.rot90(patch[i], rotation_angle)  # The actual rotation angle is rotation_angle * 90
        # patch location
        x_location, y_location = np.random.randint(low=0, high=image_size[1]-patch.shape[1]), np.random.randint(low=0, high=image_size[2]-patch.shape[2])
        for i in range(patch.shape[0]):
            applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location, y_location

# Test the patch on dataset
def test_patch(pad_row, target, patch, test_loader, model, mask):

    padding_row = pad_row

    model.eval()
    test_total, test_actual_total, test_success = 0, 0, 0
    for (image, label) in test_loader:
        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] == label and predicted[0].data.cpu().numpy() != target:
            test_actual_total += 1
            # applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
            # applied_patch = torch.from_numpy(applied_patch)
            # mask = torch.from_numpy(mask)
            ori_x = np.asarray((image.cpu()).squeeze())

            pad_image = np.pad(ori_x[0], ((0, padding_row), (0, 0)), constant_values=0)
            pad_image = Image.fromarray(np.uint8(pad_image * 255))
            pad_trans = transform(pad_image)
            pad_trans = pad_trans.unsqueeze(0)
            pad_trans = pad_trans.to(device)

            perturbated_image = torch.mul(mask.type(torch.FloatTensor), patch.type(torch.FloatTensor)) + torch.mul(
                (1 - mask.type(torch.FloatTensor)), pad_trans.type(torch.FloatTensor))

            # perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.cuda()
            output = model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1
    return test_success / test_actual_total

def test_base(pad_row, target, patch, test_loader, model, mask, base):



    padding_row = pad_row

    model.eval()
    test_total, test_actual_total, test_success = 0, 0, 0
    for (image, label) in test_loader:

        ###random patch########
        if base == 'random':
            patch_random = np.random.uniform(0, 1, [1, 3, 224, 224])
            patch_random = torch.from_numpy(patch_random)
            # patch_random = patch_random.to(device)
            patch = torch.mul(mask.type(torch.FloatTensor), patch_random.type(torch.FloatTensor))

        ###benign patch#####
        if base == 'benign':
            imagepath = '/home/ubuntu/zhan/dataset/figshare_img/val/benign'
            pathdir = os.listdir(imagepath)
            sample = random.sample(pathdir, 1)
            be_image = Image.open(imagepath + '/' + sample[0])
            be_image = np.asarray(be_image)

            be_transformed = transform(Image.fromarray(be_image))
            be_transformed = np.asarray(be_transformed)
            be_transformed = torch.from_numpy(be_transformed)
            # be_transformed = be_transformed.to(device)
            patch[0, : , -pad_row: , :] = be_transformed[:,:pad_row,:]
        ########################
        if base == 'transfer':
            # transfer_path = './img_patch/res/' + str(pad_row)
            # patchname = random.sample(os.listdir(transfer_path), 1)
            # patch_path = os.path.join(transfer_path, patchname[0])
            # patch = np.load(patch_path)
            # patch = torch.from_numpy(patch)

            transfer_path = './log/squ'
            patchname = '/row_' + str(pad_row) +'_'+ os.path.split(transfer_path)[1]+'.npy'
            patch_path = transfer_path + patchname
            patch = np.load(patch_path)
            patch = torch.from_numpy(patch)

        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] == label and predicted[0].data.cpu().numpy() != target:
            test_actual_total += 1
            # applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
            # applied_patch = torch.from_numpy(applied_patch)
            # mask = torch.from_numpy(mask)
            ori_x = np.asarray((image.cpu()).squeeze())

            pad_image = np.pad(ori_x[0], ((0, padding_row), (0, 0)), constant_values=0)
            pad_image = Image.fromarray(np.uint8(pad_image * 255))
            pad_trans = transform(pad_image)
            pad_trans = pad_trans.unsqueeze(0)
            pad_trans = pad_trans.to(device)

            perturbated_image = torch.mul(mask.type(torch.FloatTensor), patch.type(torch.FloatTensor)) + torch.mul(
                (1 - mask.type(torch.FloatTensor)), pad_trans.type(torch.FloatTensor))

            # perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.cuda()
            output = model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1
    return test_success / test_actual_total