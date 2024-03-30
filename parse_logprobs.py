import json
import numpy as np
import fnmatch
import os
from tqdm import tqdm
import numpy as np
import collections
import matplotlib.pyplot as plt
from functools import cmp_to_key
import pandas as pd


def kl_util(p, q):
    p = np.exp(p)
    q = np.exp(q)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

arr1 = np.load('/root/azure-storage/eval_outputs/logprobs/mmlu/pretrainedrootazurestoragellamaweightsabhinavpretrainedLlama27bhfuseaccelerateTruefewshot0arc_easy_batch_size_1_.npy',allow_pickle=True) #always in memory

arr2 = np.load('/root/azure-storage/eval_outputs/logprobs/mmlu/rootazurestoragequantenginespretrainedtrtLlama27bhfsmoothquantsq051gpupertokenchannelfewshot0arc_easy_batch_size_1_.npy',allow_pickle=True)
#flush after use 


breakpoint()
kl_option = []
# req corresponds to question, option combo
for i_req in range(0,len(arr1)):
    kl_tokens =[]
    for j_option_tokens in range(0,len(arr1[i_req])):
        base_vocab_prob = arr1[i_req][j_option_tokens]
        qtz_vocab_prob = arr2[i_req][j_option_tokens]
        kl_answer = kl_util(base_vocab_prob,qtz_vocab_prob)
        kl_tokens.append(kl_answer)
    kl_option.append(np.mean(kl_tokens))

print(np.mean(kl_option))
breakpoint()