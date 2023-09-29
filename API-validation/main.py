import os

import json_parser as jp
import analyzer as analyzer
import sys
import csv
import json
import random


class LinerMap(object):

    def __init__(self):
        self.items = []

    # 往表中添加元素
    def add(self, k, v):
        self.items.append((k, v))

    # 线性方式查找元素
    def get(self, k):
        for key, value in self.items:
            if key == k:  # 键存在，返回值
                return int(value) % 307
            continue
        return -1  # 返回-1，API函数完整后就不会有键不存在的情况


def get_sample_name(list_file_path):
    list_file = open(list_file_path, 'r')
    samples = []
    for line in list_file.readlines():
        curline = line.rstrip("\n")
        samples.append(curline)
    #    print(samples)
    return samples


def trans(json_path, csv_path, mp: LinerMap):
    with open(json_path, 'rb') as f_ojb:
        json_file = json.load(f_ojb, )

    csv_file = open(csv_path, 'a')
    csv_file.write(json_path + ',')
    writer = csv.writer(csv_file)
    # writer.writerow(str(json_path))
    # print(json_data)

    sequence = []
    apis = []
    json_calls_list = json_file['calls']
    for json_calls_data in json_calls_list:
        json_api = json_calls_data['api']

        if mp.get(str(json_api).lower()) != -1:

            sequence.append(json_api)
            apis.append(str(mp.get(str(json_api).lower())))
        # print(json_api)
        # writer.writerow([json_api])
    # writer.writerow(apis)

    csv_file.write(','.join(apis[:]) + "\n")
    csv_file.write('name' + ',')
    csv_file.write(','.join(sequence[:]) + "\n")

    f_ojb.close()
    csv_file.close()


def main():
    report_path = report_root_path + "/report.json"
    jp.report(report_path)
    F_NAME = jp.get_file_name()
    analyzer.set_file(F_NAME)
    analyzer.main()

    # 存放API函数，以便后续索引
    m = LinerMap()
    csv_reader = csv.reader(open("./api.csv"))
    for line in csv_reader:
        m.add(str(line[0]).lower(), line[1])
        print(str(line[0]).lower(), line[1])

    # 将LIST.TXT中列举的文件夹中json文件提取为数字串
    sample_names = F_NAME
    for i in range(0, 1):
        # print(sample_names[i])
        # sample_names[i] = sample_names[i].rstrip("\r")
        filename = './storage/' + sample_names + '/processes.json'
        # filename = 'D:/VMShare/cuckoo_report_analyzer-master/storage/' + sample_names[i] + '/processes.json'
        trans(filename, os.path.join('.', sample_names.split('.')[0]+'.csv'), m)

    # # 将test.csv中的每行API字符串转为test2.csv中的数字串
    # # test.csv中格式：hash,API_name[100]
    # csv_file = open("./test2.csv", 'a')
    # api_csv_reader = csv.reader(open("./test.csv"))
    # for line in api_csv_reader:
    #     csv_file.write(line[0] + ',')
    #     api_num = []
    #     for i in range(1, 100):
    #         api_num.append(str(m.get(str(line[i]).lower())))
    #     csv_file.write(','.join(api_num) + "\n")

if __name__ == "__main__":
    for i in range(1,135):
        report_root_path = os.path.join('/home/ubuntu/.cuckoo/storage/analyses', str(i) + '/reports')
        main()
