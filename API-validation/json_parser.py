import json
import sys
import ipapi
from pandas.io.json import json_normalize
import hashlib
from termcolor import colored
import os, errno

F_PATH = ""
F_NAME = ""


class Report(object):
    def __init__(self, data):
        self.__dict__ = data


def load_data(file):
    print
    file
    with open(file, 'rb') as f:
        report_dict = json.load(f)
    return report_dict


def get_behavior(obj):
    i = 0
    for b in obj.keys():
        if isinstance(obj[b], dict):
            for z in obj[b]:
                if isinstance(obj[b], list):
                    print
                    "|----------------------------list"
                    object_list = obj[b]
                    for a in object_list:
                        print
                        "|--"
                        if isinstance(a, dict):
                            print
                            "|----------------------------list->dict"
                            for keys in a.keys():
                                print
                                "|--" + colored(str(keys), 'cyan')
                                if isinstance(a[keys], dict):
                                    print
                                    "|----------------------------list->dict->dict"
                                    for b in a[keys]:
                                        print
                                        "|--"
                                        print
                                        "|--|--" + colored(str(b), 'red')
                                        for c in a[keys][b]:
                                            print
                                            "|--|--|--" + str(c)
                                elif isinstance(a[keys], list):
                                    print
                                    "|----------------------------list->dict->list"

                                    temp = a[keys]
                                    write_file(temp, str(keys))
                                    del temp

        elif isinstance(obj[b], list):
            print
            "|----------------------------list" + str(b)
            object_list = obj[b]
            for a in object_list:
                print
                "|--"
                if isinstance(a, dict):
                    print
                    "|----------------------------list->dict->" + str(b)
                    temp = a
                    write_file(temp, str(b))
                    temp = {}


def get_networkActivities(obj):
    i = 0
    temp = obj['udp']
    write_file(temp, "network")
    temp = {}


def get_static(obj):
    temp = obj['pe_imports']
    write_file(temp, "pe_imports")
    temp = obj['pe_sections']
    write_file(temp, "pe_sections")


def get_info(obj):
    i = 0
    temp = obj['file']
    write_file(temp, "executable_info")
    temp = {}


def get_string(obj):
    temp = obj
    # print temp
    write_file(temp, "string_debug")


def get_file_name():
    return F_NAME


def set_path(path):
    global F_PATH
    # F_PATH = os.getcwd() + path
    F_PATH = path


def write_file(obj, title):
    completeName = os.path.join(F_PATH, title + ".json")
    with open(completeName, 'w') as f:
        json.dump(obj, f)


def create_directory(directory):
    try:
        os.makedirs(directory)
        set_path(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def report(file):
    global F_PATH, F_NAME

    report = Report(load_data(file))
    F_NAME = report.target['file']['name']
    # F_PATH = os.getcwd() + "/storage/" + report.target['file']['name']
    F_PATH = "./storage/" + report.target['file']['name']
    create_directory(F_PATH)

    try:
        get_behavior(report.behavior)
#        print("behavior saved in dir =" + F_PATH)

    except:
        print("tidak dapat menemukan behavior")

    try:
        get_static(report.static)
#        print("PE imports saved in dir =" + F_PATH)

    except:
        print("tidak dapat menemukan static")

    try:
        get_networkActivities(report.network)
 #       print("network activities saved in dir =" + F_PATH)

    except:
        print("tidak dapat menemukan network activity")

    try:
        get_info(report.target)
#        print("file information saved in dir =" + F_PATH)

    except:
        print("tidak dapat menemukan informasi file")

    try:
        get_string(report.strings)
 #       print("debug information saved in dir =" + F_PATH)

    except:
        print
        "tidak dapat menemukan debug file"
