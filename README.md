# malpatch
MalPatch: Evading DNN-Based Malware Detection With Adversarial Patches

MalPatch_binary contains the code that attacks the MalConv&CNN.

MalPatch_img contains the code that attacks the Grayscale detectors.

In each Folder, there are two main attack.py file corresponding to white-box attack and black-box attack.

To run MalPatch_binary, you need to replace this file in library ../secml/adv/attacks/evasion/ with c_attack_evasion.py in the folder.

