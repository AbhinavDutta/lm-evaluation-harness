import fnmatch
import os
from matplotlib import pyplot as plt
import numpy as np
from astropy.visualization import hist

prefix = '*pretrained*'

def getinfo(file):
    model=''
    if file.find('Llama27b')!=-1:
       model='llama2-7b'
    elif file.find('Llama213b')!=-1:
       model=''
    
    quantization=''
    if file.find('8bit')!=-1:
      quantization='8'
    elif file.find('4bit')!=-1:
      quantization='4'
    else:
      quantization='16'

    
    type=''
    
    if file.find('wrong')!=-1:
      type='wrong'
    elif file.find('correct')!=-1:
      type='correct'
    else:
      type='normal'
    


    shot=''
    if file.find('fewshot1')!=-1:
      shot='1shot'
    elif file.find('fewshot3')!=-1:
      shot='3shot'
    elif file.find('fewshot5')!=-1:
      shot='5shot'
    else:
      shot='0shot'
    
    task=''
    possible_tasks= {'best_likelihoods': 'best likelihood',
                    'correct_exp_loglikelihoods':'ground truth likelihood',
                    'top_margin':'margin between best 2',
                    'a_exp_loglikelihoods':'A option likelihood',
                    'b_exp_loglikelihoods':'B option likelihood',
                    'c_exp_loglikelihoods':'C option likelihood',
                    'd_exp_loglikelihoods':'D option likelihood',
                    }
    for pt,common_name in possible_tasks.items():
        if file.find(pt)!=-1:
            task=common_name
            break

    return model,quantization,type,shot,task





#write code to iterate over all files in the directory that have the above prefix
for file in os.listdir('./aggregated_logs'):
    if fnmatch.fnmatch(file, prefix):
        
    
        data_collected = np.loadtxt('./aggregated_logs/'+file)
        if len(data_collected)<3:
            continue
        #data_collected=-np.sort(-data_collected)
        print(file,len(data_collected))
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        
        
        #ax.hist(data_collected, bins = 10)
        hist(data_collected, bins='freedman', ax=ax, histtype='stepfilled',alpha=1.0, density=False)
        label='total= '+str(len(data_collected))+str(' mean= %.4f'%np.mean(data_collected))+str(' std dev= %.4f '%np.std(data_collected))
        ax.set_xlabel(label)
        model,quantization,type,shot,task=getinfo(file)
        plt.title(model+', '+quantization+'bit, '+type+', '+shot+', '+task)
        fig.savefig('./artifacts/'+file+'.png')   # save the figure to file
        plt.close(fig)    # close the figure window

