#!/usr/bin/env python
# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES']='1,2'
import magic
import numpy as np
from secml.array import CArray

from secml_malware.attack.blackbox.c_wrapper_phi import CEnd2EndWrapperPhi
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm
from secml_malware.models.malconv import MalConv, DNN_Net
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware, End2EndModel

# define target model
net_choice = 'MalConv' # MalConv/AvastNet

if net_choice == 'MalConv':
    ##################attack Malconv####################
    net = MalConv()
    net = CClassifierEnd2EndMalware(net)
    net.load_pretrained_model() # Pr-MalConv

    # if attack fine-tuned MalConv
    net.load_pretrained_model('./secml_malware/data/trained/finetuned_malconv.pth')


elif net_choice == 'AvastNet':
    ###################attack AvastNet##################
    net = DNN_Net()
    net = CClassifierEnd2EndMalware(net)
    net.load_pretrained_model('./secml_malware/data/trained/dnn_pe.pth')


#####################################################


net = CEnd2EndWrapperPhi(net)

from secml_malware.attack.blackbox.c_blackbox_malpatch import CBlackBoxMalPatchProblem
attack = CBlackBoxMalPatchProblem(net, population_size=30, iterations=100, is_debug=False)
# attack = CBlackBoxHeaderEvasionProblem(net, population_size=30, iterations=100, is_debug=True)

engine = CGeneticAlgorithm(attack)

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
    x = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
    _, confidence = net.predict(x, True)

    if confidence[0, 1].item() < 0.5:
        continue

    print(f"> Added {f} with confidence {confidence[0,1].item()}")
    Train_X.append(code)
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
    x = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
    _, confidence = net.predict(x, True)

    if confidence[0, 1].item() < 0.5:
        continue

    print(f"> Added {f} with confidence {confidence[0,1].item()}")
    Test_X.append(code)
    conf = confidence[1][0].item()
    Test_y.append([1 - conf, conf])
    test_file_names.append(path)

success, best_success_rate = 0, 0
for epoch in range(3):
    for sample, label in zip(Train_X, Train_y):

        sample = CArray(np.frombuffer(sample, dtype=np.uint8)).atleast_2d()
        y_pred, adv_score, adv_ds, f_obj, byte_change = engine.run(sample, CArray(label[1]))
        # print(engine.confidences_)
        # print(f_obj)

        adv_x = adv_ds.X[0, :]
        engine.write_adv_to_file(adv_x, 'adv_exe')
        with open('adv_exe', 'rb') as h:
            code = h.read()
        real_adv_x = CArray(np.frombuffer(code, dtype=np.uint8))
        _, confidence = net.predict(CArray(real_adv_x), True)
        print(confidence[0, 1].item())
        if confidence[0, 1].item() < 0.5:
            success += 1

    avg_success = success/(len(train_file_names)*(epoch+1))
    print("Epoch:{} Average evasion rate on trainset with different patch: {:.3f}%" .format(epoch, 100 * avg_success))

    #### patch on the train_set
    train_success = 0
    for sample, label in zip(Train_X, Train_y):
        if net_choice != 'DNN_Net':
            sample = End2EndModel.bytes_to_numpy(
                sample, 2**20, 256, False
            )
        else:
            sample = End2EndModel.bytes_to_numpy_2(
                sample, 2**20, 256, False
            )

        # _, confidence = net.predict(CArray(sample), True)

        padding_positions = CArray(sample).find(CArray(sample) == 256)
        if not padding_positions:
            indexes_to_perturb = [i for i in range(2, 0x3C)]
        else:
            indexes_to_perturb = [i for i in range(2, 0x3C)] + list(range(
                padding_positions[0],
                min(len(sample), padding_positions[0] + 144)
            ))

        for b in range(len(indexes_to_perturb)):
            index = indexes_to_perturb[b]
            sample[index] = (byte_change[b]* 255).astype(np.int)
        _, confidence = net.predict(CArray(sample), True)
        print(confidence[0, 1].item())
        if confidence[0, 1].item() < 0.5:
            train_success += 1
    avg_train_success = train_success / len(train_file_names)
    print("Epoch:{} Average evasion rate on trainset: {:.3f}%".format(epoch, 100 * avg_train_success))

    #### patch on the test_set
    test_success = 0
    for sample, label in zip(Test_X, Test_y):

        if net_choice != 'DNN_Net':
            sample = End2EndModel.bytes_to_numpy(
                sample, 2 ** 20, 256, False
            )
        else:
            sample = End2EndModel.bytes_to_numpy_2(
                sample, 2 ** 20, 256, False
            )

        padding_positions = CArray(sample).find(CArray(sample) == 256)
        if not padding_positions:
            indexes_to_perturb = [i for i in range(2, 0x3C)]
        else:
            indexes_to_perturb = [i for i in range(2, 0x3C)] + list(range(
                padding_positions[0],
                min(len(sample), padding_positions[0] + 144)
            ))
        for b in range(len(indexes_to_perturb)):
            index = indexes_to_perturb[b]
            sample[index] = (byte_change[b]* 255).astype(np.int)
        _, confidence = net.predict(CArray(sample), True)
        print(confidence[0, 1].item())
        if confidence[0, 1].item() < 0.5:
            test_success += 1
    avg_test_success = test_success / len(test_file_names)
    print("Epoch:{} Average evasion rate on testset: {:.3f}%".format(epoch, 100 * avg_test_success))

    if avg_test_success > best_success_rate:
        best_success_rate = avg_test_success
        file = 'iter_malconv_patch.npy'
        np.save(file, byte_change)
print("Best evasion rate on testset: {:.3f}%".format(100 * best_success_rate))