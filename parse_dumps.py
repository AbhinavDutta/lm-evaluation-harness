import fnmatch
import os
from matplotlib import pyplot as plt
import numpy as np
from astropy.visualization import hist
import tqdm

prefix = 'pretrained*'
DIR = './hostvm/logs/'
ART_PATH_BATCH = './hostvm/artifacts/batch/'
ART_PATH_QUANT = './hostvm/artifacts/quantization/'
ART_PATH_HISTOGRAM = './hostvm/artifacts/histogram/'

def getinfo(file):
    model=''
    if file.find('Llama27bhf')!=-1:
       model='llama2-7b'
    elif file.find('Llama27bchat')!=-1:
       model='llama2-7b-chat'
    elif file.find('Llama213bchat')!=-1:
       model='llama2-13b-chat'
    elif file.find('Llama213bhf')!=-1:
       model='llama2-13b'
    else:
       assert(False)
    
    quantization=''
    if file.find('8bit')!=-1:
      quantization='8bit'
    elif file.find('4bit')!=-1:
      quantization='4bit'
    else:
      quantization='16bit'

    
    wrong_corr_norm=''
    
    if file.find('wrong')!=-1:
      wrong_corr_norm='wrong'
    elif file.find('correct_likelihoods')==-1 and file.find('correct_likelihoods')==-1 and file.find('correct')!=-1:
      wrong_corr_norm='correct'
    else:
      wrong_corr_norm='normal'
    


    shot=''
    if file.find('fewshot1')!=-1:
      shot='1shot'
    elif file.find('fewshot3')!=-1:
      shot='3shot'
    elif file.find('fewshot5')!=-1:
      shot='5shot'
    elif file.find('fewshot0')!=-1:
       shot='0shot'
    else:
      assert(False)

    batch_size=''
    if file.find('batch_size_1')!=-1:
       batch_size='batch=1'
    elif file.find('batch_size_5')!=-1:
       batch_size='batch=5'
    elif file.find('batch_size_32')!=-1:
       batch_size='batch=32'
    else:
       assert(False)
    
    possible_tasks = {'hendrycksTest-abstract_algebra': 'abstract algebra',
                      'hendrycksTest-machine_learning': 'machine learning', 'hendrycksTest-philosophy': 'philosophy', 'hendrycksTest-high_school_government_and_politics': 'govt politics',
    }

    task=''
    for key_,value in possible_tasks.items():
       if file.find(key_)!=-1:
          task=value
          break
                          

    artifacts=''
    possible_artifacts= {'best_likelihoods': 'best likelihood',
                    'correct_likelihoods':'ground truth likelihood',
                    'top_margin':'margin between best 2',
                    'a_likelihoods':'A option likelihood',
                    'b_likelihoods':'B option likelihood',
                    'c_likelihoods':'C option likelihood',
                    'd_likelihoods':'D option likelihood',
                    'clearance': 'clearance',
                    '_mask': 'mask',
                    }
    for key_,value in possible_artifacts.items():
        if file.find(key_)!=-1:
            artifacts=value
            break
    if task=='' or artifacts=='':
       assert(False)

    return model,quantization,wrong_corr_norm,shot,batch_size,task,artifacts


data_table={}
for file in (os.listdir(DIR)):
    if fnmatch.fnmatch(file, prefix):
        model,quantization,wrong_corr_norm,shot,batch_size,task,artifacts=getinfo(file) 
        data_collected = np.loadtxt(DIR+file)
        key = model+quantization+wrong_corr_norm+shot+batch_size+task+artifacts
        data_table[key]=data_collected


#Scatter Plots batch size variation
for file in tqdm.tqdm(os.listdir(DIR)):
    if fnmatch.fnmatch(file, prefix):
        model,quantization,wrong_corr_norm,shot,batch_size,task,artifacts=getinfo(file) 
        
        #print(model,quantization,wrong_corr_norm,shot,batch_size,task,artifacts)
        if artifacts=='mask': #dont scatter plots for these
           continue
        data_collected = np.loadtxt(DIR+file)
        if len(data_collected)<3: #skip over empty files
            continue
        comment=''
        if batch_size!='batch=1' and (artifacts=='A option likelihood' or artifacts=='B option likelihood' or artifacts=='C option likelihood' or artifacts=='D option likelihood'):
           comment='_DIFF_WRT_bs1_'
           data_collected=np.subtract(data_collected,data_table[model+quantization+wrong_corr_norm+shot+'batch=1'+task+artifacts])
           
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        x = np.arange(0,len(data_collected))
        ax.plot(x, data_collected)
    
        label='total= '+str(len(data_collected))+str(' mean= %.5f'%np.mean(data_collected))+str(' std dev= %.5f '%np.std(data_collected))
        ax.set_xlabel(label)
        
        plt.title(model+', '+quantization+', '+wrong_corr_norm+', '+shot+', '+batch_size+', '+artifacts+' '+comment+', '+task,loc='center', wrap=True)
        
        fig.savefig(ART_PATH_BATCH+file+comment+'_scatter_plot.png')   # save the figure to file
        plt.close(fig)    # close the figure window


#Scatter Plots quantization variation
for file in tqdm.tqdm(os.listdir(DIR)):
    if fnmatch.fnmatch(file, prefix):
        model,quantization,wrong_corr_norm,shot,batch_size,task,artifacts=getinfo(file) 
        
        #print(model,quantization,wrong_corr_norm,shot,batch_size,task,artifacts)
        if artifacts=='mask': #dont scatter plots for these
           continue
        data_collected = np.loadtxt(DIR+file)
        if len(data_collected)<3: #skip over empty files
            continue
        comment=''
        if quantization!='16bit' and (artifacts=='A option likelihood' or artifacts=='B option likelihood' or artifacts=='C option likelihood' or artifacts=='D option likelihood'):
           comment='_DIFF_WRT_qt_16bit_'
           data_collected=np.subtract(data_collected,data_table[model+'16bit'+wrong_corr_norm+shot+batch_size+task+artifacts])
           
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        x = np.arange(0,len(data_collected))
        ax.plot(x, data_collected)
    
        label='total= '+str(len(data_collected))+str(' mean= %.5f'%np.mean(data_collected))+str(' std dev= %.5f '%np.std(data_collected))
        ax.set_xlabel(label)
        
        plt.title(model+', '+quantization+', '+wrong_corr_norm+', '+shot+', '+batch_size+', '+artifacts+' '+comment+', '+task,loc='center', wrap=True)
        
        fig.savefig(ART_PATH_QUANT+file+comment+'_scatter_plot.png')   # save the figure to file
        plt.close(fig)    # close the figure window




#histogram
for file in tqdm.tqdm(os.listdir(DIR)):
    if fnmatch.fnmatch(file, prefix):
        model,quantization,wrong_corr_norm,shot,batch_size,task,artifacts=getinfo(file) 
        
        #print(model,quantization,wrong_corr_norm,shot,batch_size,task,artifacts)
        data_collected = np.loadtxt(DIR+file)
        if len(data_collected)<3: #skip over empty files
            continue

        if artifacts=='mask':
           continue  
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        hist(data_collected, bins='freedman', ax=ax, histtype='stepfilled',alpha=1.0, density=False)
    
        label='total= '+str(len(data_collected))+str(' mean= %.5f'%np.mean(data_collected))+str(' std dev= %.5f '%np.std(data_collected))
        ax.set_xlabel(label)
        
        plt.title(model+', '+quantization+', '+wrong_corr_norm+', '+shot+', '+batch_size+', '+artifacts+', '+task,loc='center', wrap=True)
        
        fig.savefig(ART_PATH_HISTOGRAM+file+'_histogram.png')   # save the figure to file
        plt.close(fig)    # close the figure window
