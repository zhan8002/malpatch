import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import pandas as pd
import torch
import time
from dnn_net import DNN_Net
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import lief
import struct

class ExeDataset(Dataset):
    def __init__(self, fp_list, data_path, label_list, first_n_byte=4096):
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        # try:
        #     with open(self.data_path+self.fp_list[idx],'rb') as f:
        #         tmp = [i for i in f.read()[:self.first_n_byte]]
        #         tmp = tmp+[0]*(self.first_n_byte-len(tmp))
        # except:
        #     with open(self.data_path+self.fp_list[idx].lower(),'rb') as f:
        #         tmp = [i for i in f.read()[:self.first_n_byte]]
        #         tmp = tmp+[0]*(self.first_n_byte-len(tmp))
        #
        # return np.array(tmp),np.array([self.label_list[idx]])
        with open(self.data_path+self.fp_list[idx], "rb") as file_handle:
            bytez = file_handle.read()
            b = np.ones((self.first_n_byte,), dtype=np.uint16) * 0

            bytez = np.frombuffer(bytez, dtype=np.uint8)

            # liefpe = lief.PE.parse(bytez.tolist())
            # first_content_offset = liefpe.dos_header.addressof_new_exeheader

            # pe_position = bytez[0x3C:0x40].astype(np.uint16)
            pe_position = struct.unpack("<I", bytez[0x3C:0x40])
            pe_position = pe_position[0]

            if pe_position > len(bytez):
                bytez = bytez[:4096]
                b[: min(4096, len(bytez))] = bytez[: min(4096, len(bytez))]
                return np.array(b, dtype=float), np.array([self.label_list[idx]])


            else:
                optional_header_size = bytez[pe_position + 20]

                coff_header_size = 24

                content_offset = pe_position + optional_header_size + coff_header_size + 12

                first_content_offset = struct.unpack("<I", bytez[content_offset:content_offset+4])

                bytez = bytez[:first_content_offset[0]]


                b[: min(4096, len(bytez))] = bytez[: min(4096, len(bytez))]
                return np.array(b, dtype=float), np.array([self.label_list[idx]])


if __name__ == "__main__":

    is_support = torch.cuda.is_available()
    if is_support:
        device = torch.device('cuda:0')

    use_gpu = True

    chkpt_acc_path = 'dnn_pe.pth'

    train_data_path = ''
    valid_data_path = ''

    train_label_path = './train_label.csv'
    valid_label_path = './test_label.csv'

    # Load Ground Truth.
    tr_label_table = pd.read_csv(train_label_path, header=None, index_col=0)
    #tr_label_table.index = tr_label_table.index.str.upper()
    tr_label_table = tr_label_table.rename(columns={1: 'ground_truth'})
    val_label_table = pd.read_csv(valid_label_path, header=None, index_col=0)
    #val_label_table.index = val_label_table.index.str.upper()
    val_label_table = val_label_table.rename(columns={1: 'ground_truth'})

    # Merge Tables and remove duplicate
    tr_table = tr_label_table.groupby(level=0).last()
    del tr_label_table
    val_table = val_label_table.groupby(level=0).last()
    del val_label_table
    tr_table = tr_table.drop(val_table.index.join(tr_table.index, how='inner'))

    dataloader = DataLoader(
        ExeDataset(list(tr_table.index), train_data_path, list(tr_table.ground_truth), 4096),
        batch_size=32, shuffle=True, num_workers=8)
    validloader = DataLoader(
        ExeDataset(list(val_table.index), valid_data_path, list(val_table.ground_truth), 4096),
        batch_size=1, shuffle=False, num_workers=8)

    valid_idx = list(val_table.index)
    del tr_table
    del val_table

    dnn = DNN_Net(input_length=4096)
    bce_loss = nn.BCEWithLogitsLoss()
    adam_optim = optim.Adam([{'params': dnn.parameters()}], lr=0.001)
    sigmoid = nn.Sigmoid()

    print("start train")

    if use_gpu:
        dnn = dnn.cuda()
        bce_loss = bce_loss.cuda()
        sigmoid = sigmoid.cuda()


    step_msg = 'step-{}-loss-{:.6f}-acc-{:.4f}-time-{:.2f}'
    valid_msg = 'step-{}-tr_loss-{:.6f}-tr_acc-{:.4f}-val_loss-{:.6f}-val_acc-{:.4f}'
    log_msg = '{}, {:.6f}, {:.4f}, {:.6f}, {:.4f}, {:.2f}'
    history = {}
    history['tr_loss'] = []
    history['tr_acc'] = []

    valid_best_acc = 0.0
    total_step = 0
    step_cost_time = 0

    while total_step < 20000:

        # Training
        for step, batch_data in enumerate(dataloader):
            start = time.time()

            adam_optim.zero_grad()

            cur_batch_size = batch_data[0].size(0)

            exe_input = batch_data[0].cuda() if use_gpu else batch_data[0]
            exe_input = Variable(exe_input.long(), requires_grad=False)

            label = batch_data[1].cuda() if use_gpu else batch_data[1]
            label = Variable(label.float(), requires_grad=False)

            pred = dnn(exe_input)
            loss = bce_loss(pred, label)
            loss.backward()
            adam_optim.step()

            history['tr_loss'].append(loss.cpu().data.numpy())
            history['tr_acc'].extend(
                list(label.cpu().data.numpy().astype(int) == (sigmoid(pred).cpu().data.numpy() + 0.5).astype(int)))
            step_cost_time = time.time() - start

            # if (step + 1) % 2 == 0:
            #     print(step_msg.format(total_step, np.mean(history['tr_loss']),
            #                           np.mean(history['tr_acc']), step_cost_time), end='\r', flush=True)
            total_step += 1

            # Interupt for validation
            if total_step % 1000 == 0:
                break


        # Testing
        history['val_loss'] = []
        history['val_acc'] = []
        history['val_pred'] = []

        for _, val_batch_data in enumerate(validloader):
            cur_batch_size = val_batch_data[0].size(0)

            exe_input = val_batch_data[0].cuda() if use_gpu else val_batch_data[0]
            exe_input = Variable(exe_input.long(), requires_grad=False)

            label = val_batch_data[1].cuda() if use_gpu else val_batch_data[1]
            label = Variable(label.float(), requires_grad=False)

            pred = dnn(exe_input)
            loss = bce_loss(pred, label)

            history['val_loss'].append(loss.cpu().data.numpy())
            history['val_acc'].extend(
                list(label.cpu().data.numpy().astype(int) == (sigmoid(pred).cpu().data.numpy() + 0.5).astype(int)))
            history['val_pred'].append(list(sigmoid(pred).cpu().data.numpy()))

        # print(log_msg.format(total_step, np.mean(history['tr_loss']), np.mean(history['tr_acc']),
        #                      np.mean(history['val_loss']), np.mean(history['val_acc']), step_cost_time),
        #        flush=True)

        print(valid_msg.format(total_step, np.mean(history['tr_loss']), np.mean(history['tr_acc']),
                               np.mean(history['val_loss']), np.mean(history['val_acc'])))
        if valid_best_acc < np.mean(history['val_acc']):
            valid_best_acc = np.mean(history['val_acc'])
            torch.save(dnn.state_dict(), chkpt_acc_path)
            print('Checkpoint saved at', chkpt_acc_path)

        history['tr_loss'] = []
        history['tr_acc'] = []