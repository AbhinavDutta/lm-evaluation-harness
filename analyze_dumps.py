import fnmatch
import os
from matplotlib import pyplot as plt
import numpy as np
from astropy.visualization import hist
import tqdm
import collections

prefix = 'pretrained*'
DIR = './hostvm/logs_1/'

POSSIBLE_TASKS=['abstract algebra','machine learning', 'philosophy', 'govt politics']
POSSIBLE_MODELS=['llama2-70b','llama2-70b-chat']
POSSIBLE_QTZ=['16bit', '8bit', '4bit']
POSSIBLE_CAT=['normal', 'wrong', 'correct']
POSSIBLE_SHOT=['0shot',  '1shot', '3shot', '5shot']
POSSIBLE_BATCH=['batch=1', 'batch=5', 'batch=32']
ARTIFACT_TO_ENGLISH= {'best_likelihoods': 'best likelihood',
                    'correct_likelihoods':'ground truth likelihood',
                    'top_margin':'margin between best 2',
                    'a_likelihoods':'A option likelihood',
                    'b_likelihoods':'B option likelihood',
                    'c_likelihoods':'C option likelihood',
                    'd_likelihoods':'D option likelihood',
                    'clearance': 'clearance',
                    '_mask': 'mask',
                    }
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
    elif file.find('Llama270bhf')!=-1:
       model='llama2-70b'
    elif file.find('Llama270bchat')!=-1:
       model='llama2-70b-chat'
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
    elif file.find('correct_likelihoods')==-1 and file.find('correct')!=-1:
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

def get_map_config_to_data(DIR):
  map_config_to_data=collections.defaultdict(lambda:np.array([]))
  for file in (os.listdir(DIR)):
    if fnmatch.fnmatch(file,'*.txt') and fnmatch.fnmatch(file,'pretrained*'):
        model,quantization,wrong_corr_norm,shot,batch_size,task,_artifacts=getinfo(file)
        arr=np.loadtxt(DIR+file)
        map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+_artifacts]=arr

  return map_config_to_data

map_config_to_data=get_map_config_to_data(DIR)

def number_of_answers_changed(**kwargs):
  resp2=[]
  resp1=[]
  for a,b,c,d in zip(kwargs['a1'],kwargs['b1'],kwargs['c1'],kwargs['d1']):  
    resp1.append((a,b,c,d))
  for a,b,c,d in zip(kwargs['a2'],kwargs['b2'],kwargs['c2'],kwargs['d2']):
    resp2.append((a,b,c,d))

  change=[]
  for x,y in zip(resp1,resp2):
    if np.argmax(x)!=np.argmax(y):
      change.append(np.max(x)-sorted(x)[-2])
  return len(change)

x_arr=[]
y_arr=[]
freq=collections.defaultdict(lambda:0)
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis 
#color={'llama2-70b':'red','llama2-70b-chat':'blue'}
#color={'0shot':'red','1shot':'blue', '3shot':'green', '5shot':'yellow'}
#color={'16bit':'red','8bit':'blue', '4bit':'green'}
#color={'abstract algebra':'red','machine learning':'blue', 'philosophy':'green', 'govt politics':'yellow'}
#color={'batch=1':'red','batch=5':'blue', 'batch=32':'green'}
color={'0-30%':'red','30-50%':'orange','50-70%':'skyblue','70-100%':'blue'}
for model in POSSIBLE_MODELS:
    for shot in POSSIBLE_SHOT:
        for quantization in POSSIBLE_QTZ:            
            for task in POSSIBLE_TASKS:
                for batch_size in POSSIBLE_BATCH:
                    current_config=model+quantization+'normal'+shot+batch_size+task
                    base_config=model+'16bit'+'normal'+shot+batch_size+task
                    mask2=map_config_to_data[current_config+'mask']
                    if mask2.size < 3:
                        continue
                    mask1=map_config_to_data[base_config+'mask']
                    if mask1.size < 3:
                        continue    
                    acc2=np.mean(mask2)
                    acc1=np.mean(mask1)
                    clearance1= np.mean(map_config_to_data[base_config+'clearance'])
                    clearance2= np.mean(map_config_to_data[current_config+'clearance'])
                    top_margin1= np.mean(map_config_to_data[base_config+'margin between best 2'])
                    top_margin2= np.mean(map_config_to_data[current_config+'margin between best 2'])
                    best_likelihood1= np.mean(map_config_to_data[base_config+'best likelihood'])
                    best_likelihood2= np.mean(map_config_to_data[current_config+'best likelihood'])
                    correct_likelihood1= np.mean(map_config_to_data[base_config+'ground truth likelihood'])
                    correct_likelihood2= np.mean(map_config_to_data[current_config+'ground truth likelihood'])
                    x = correct_likelihood2-correct_likelihood1
                    a1=map_config_to_data[base_config+ARTIFACT_TO_ENGLISH['a_likelihoods']]
                    b1=map_config_to_data[base_config+ARTIFACT_TO_ENGLISH['b_likelihoods']]
                    c1=map_config_to_data[base_config+ARTIFACT_TO_ENGLISH['c_likelihoods']]
                    d1=map_config_to_data[base_config+ARTIFACT_TO_ENGLISH['d_likelihoods']]

                    a2=map_config_to_data[current_config+ARTIFACT_TO_ENGLISH['a_likelihoods']]
                    b2=map_config_to_data[current_config+ARTIFACT_TO_ENGLISH['b_likelihoods']]
                    c2=map_config_to_data[current_config+ARTIFACT_TO_ENGLISH['c_likelihoods']]
                    d2=map_config_to_data[current_config+ARTIFACT_TO_ENGLISH['d_likelihoods']]
                    
                    y = number_of_answers_changed(a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2)
                    y= y*100.0/len(a1)
                    if acc2<0.3:
                        region='0-30%'
                    elif acc2<0.5:  
                        region='30-50%'
                    elif acc2<0.7:
                        region='50-70%'
                    else:
                        region='70-100%'
                    if region!='70-100%':
                        continue
                    ax.scatter(x,y,c=color[region],label=region,alpha=0.5)
                    freq[region]+=1
                    x_arr.append(x)
                    y_arr.append(y)
                    

handles, labels = ax.get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
ax.legend(*zip(*unique))   
print(freq)
print(len(x_arr))
metric='accuracy'
plt.title(f'flips vs change in {metric} wrt qtz=16bit')
plt.xlabel(f'change in {metric}')
plt.ylabel('% flips')
fig.savefig(f'flips_vs_{metric}_qtz.png')   # #flips vs change in accuracy

plt.close(fig)    # close the figure window
               