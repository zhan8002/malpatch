#!/usr/bin/env python
# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES']='1,2'
import magic
import numpy as np
from secml.array import CArray
import random
from secml_malware.models.malconv import MalConv, DNN_Net
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware, End2EndModel
from secml_malware.attack.whitebox.c_headerPlus_evasion import CHeaderPlusEvasion



# define target model
net_choice = 'MalConv' # MalConv/AvastNet

if net_choice == 'MalConv':
    ##################attack Malconv####################
    net = MalConv()
    net = CClassifierEnd2EndMalware(net)
    net.load_pretrained_model() # Pr-MalConv

    # if attack fine-tuned MalConv
    net.load_pretrained_model('./secml_malware/data/trained/finetuned_malconv.pth')

    partial_dos = CHeaderPlusEvasion(net, random_init=False, iterations=10, header_and_padding=True, threshold=0.5, how_many=144, is_debug=False)

elif net_choice == 'AvastNet':
    ###################attack AvastNet##################
    net = DNN_Net()
    net = CClassifierEnd2EndMalware(net)
    net.load_pretrained_model('./secml_malware/data/trained/dnn_pe.pth')

    partial_dos = CHeaderPlusEvasion(net, random_init=False, iterations=10, header_and_padding=False, threshold=0.5, how_many=0, is_debug=False)

#####################################################

Train_folder = '' # dir of malware samples for generating adversarial patch
Test_folder = '' # dir of malware samples for evaluating adversarial patch

# load Train-set samples
Train_X = []
Train_y = []
train_file_names = []

for i, f in enumerate(os.listdir(Train_folder)):
    path = os.path.join(Train_folder, f)

    if "PE32" not in magic.from_file(path):
        continue
    with open(path, "rb") as file_handle:
        code = file_handle.read()

    if net.model.__class__.__name__ != 'AvastNet':
        x = End2EndModel.bytes_to_numpy(
            code, net.get_input_max_length(), 256, False
        )
    else:
        x = End2EndModel.bytes_to_numpy_2(
            code, net.get_input_max_length(), 256, False)
    _, confidence = net.predict(CArray(x), True)

    if confidence[0, 1].item() < 0.5:
        continue

    print(f"> Original sample size: {len(code)}")
    print(f"> Added {f} with confidence {confidence[0,1].item()}")
    Train_X.append(x)
    conf = confidence[1][0].item()
    Train_y.append([1 - conf, conf])
    train_file_names.append(path)

# load Test-set samples
Test_X = []
Test_y = []
test_file_names = []

for i, f in enumerate(os.listdir(Test_folder)):


    path = os.path.join(Test_folder, f)

    if "PE32" not in magic.from_file(path):
        continue
    with open(path, "rb") as file_handle:
        code = file_handle.read()
    if net.model.__class__.__name__ != 'AvastNet':
        x = End2EndModel.bytes_to_numpy(
            code, net.get_input_max_length(), 256, False
        )
    else:
        x = End2EndModel.bytes_to_numpy_2(
            code, net.get_input_max_length(), 256, False
        )

    _, confidence = net.predict(CArray(x), True)

    if confidence[0, 1].item() < 0.5:
        continue

    print(f"> Original sample size: {len(code)}")
    print(f"> Added {f} with confidence {confidence[0,1].item()}")
    Test_X.append(x)
    conf = confidence[1][0].item()
    Test_y.append([1 - conf, conf])
    test_file_names.append(path)

success, best_success_rate = 0, 0
for epoch in range(5):
    for sample, label in zip(Train_X, Train_y):

        y_pred, adv_score, adv_ds, f_obj, byte_change = partial_dos.run(CArray(sample), CArray(label[1]))
        print(partial_dos.confidences_)
        # print(f_obj)
        if f_obj < 0.5:
            success += 1

        adv_x = adv_ds.X[0, :]
        real_adv_x = partial_dos.create_real_sample_from_adv(train_file_names[0], adv_x)
    avg_success = success/(len(train_file_names)*(epoch+1))
    print("Epoch:{} Average evasion rate on trainset with different patch: {:.3f}%" .format(epoch, 100 * avg_success))

    #### patch on the train_set
    train_success = 0
    for sample, label in zip(Train_X, Train_y):
        padding_positions = CArray(sample).find(CArray(sample) == 256)
        if not padding_positions:
            indexes_to_perturb = [i for i in range(2, 0x3C)]
        else:
            indexes_to_perturb = [i for i in range(2, 0x3C)] + list(range(
                padding_positions[0],
                min(len(sample), padding_positions[0] + 144)
            ))
        for b in range(len(indexes_to_perturb)):
            sample[indexes_to_perturb[b]] = byte_change[b]
        _, confidence = net.predict(CArray(sample), True)
        print(confidence[0, 1].item() )
        if confidence[0, 1].item() < 0.5:
            train_success += 1
    avg_train_success = train_success/len(train_file_names)
    print("Epoch:{} Average evasion rate on trainset: {:.3f}%" .format(epoch, 100 * avg_train_success))

    #### patch on the test_set
    test_success = 0
    for sample, label in zip(Test_X, Test_y):
        padding_positions = CArray(sample).find(CArray(sample) == 256)
        if not padding_positions:
            indexes_to_perturb = [i for i in range(2, 0x3C)]
        else:
            indexes_to_perturb = [i for i in range(2, 0x3C)] + list(range(
                padding_positions[0],
                min(len(sample), padding_positions[0] + 144)
            ))
        for b in range(len(indexes_to_perturb)):
            sample[indexes_to_perturb[b]] = byte_change[b]
        _, confidence = net.predict(CArray(sample), True)
        print(confidence[0, 1].item() )
        if confidence[0, 1].item() < 0.5:
            test_success += 1
    avg_test_success = test_success/len(test_file_names)
    print("Epoch:{} Average evasion rate on testset: {:.3f}%" .format(epoch, 100 * avg_test_success))

    if avg_test_success > best_success_rate:
        best_success_rate = avg_test_success

print("Best evasion rate on testset: {:.3f}%" .format(100 * best_success_rate))