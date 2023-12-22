import xlsxwriter
import fnmatch
import os
from tqdm import tqdm
import numpy as np
import collections

workbook_qtz = xlsxwriter.Workbook("SmoothQuant_per_tensor.xlsx")
POSSIBLE_AGG_ARTIFACTS= ['accuracy','clearance','number of answers changed','number of answers same','#answers changed from correct','#answers changed from wrong', 'maximum variation']
POSSIBLE_QTZ=['16bit','8bit']



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


POSSIBLE_TASKS=['abstract algebra','machine learning', 'philosophy', 'govt politics']
POSSIBLE_MODELS=['llama7bchat','llama13bchat','llama70bchat']
POSSIBLE_CAT=['normal', 'wrong', 'correct']
POSSIBLE_SHOT=['0shot',  '1shot', '3shot', '5shot']



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
  return len(change), (np.mean(change) if len(change)>0 else 0.0)
  
def number_of_answers_same(**kwargs):
  resp2=[]
  resp1=[]
  for a,b,c,d in zip(kwargs['a1'],kwargs['b1'],kwargs['c1'],kwargs['d1']):  
    resp1.append((a,b,c,d))
  for a,b,c,d in zip(kwargs['a2'],kwargs['b2'],kwargs['c2'],kwargs['d2']):
    resp2.append((a,b,c,d))

  change=[]
  for x,y in zip(resp1,resp2):
    if np.argmax(x)==np.argmax(y):
      change.append(np.max(x)-sorted(x)[-2])
  return len(change),(np.mean(change) if len(change)>0 else 0.0)

def number_of_answers_changed_from_correct(**kwargs):
  resp2=[]
  resp1=[]
  
  for a,b,c,d in zip(kwargs['a1'],kwargs['b1'],kwargs['c1'],kwargs['d1']):  
    resp1.append((a,b,c,d))
  for a,b,c,d in zip(kwargs['a2'],kwargs['b2'],kwargs['c2'],kwargs['d2']):
    resp2.append((a,b,c,d))

  change=[]
  for x,y,mask1 in zip(resp1,resp2,kwargs['mask1']):
    if np.argmax(x)!=np.argmax(y) and mask1==1:
      change.append(np.max(x)-sorted(x)[-2])
  return len(change), (np.mean(change) if len(change)>0 else 0.0)

def number_of_answers_changed_from_wrong(**kwargs):
  resp2=[]
  resp1=[]
  
  for a,b,c,d in zip(kwargs['a1'],kwargs['b1'],kwargs['c1'],kwargs['d1']):  
    resp1.append((a,b,c,d))
  for a,b,c,d in zip(kwargs['a2'],kwargs['b2'],kwargs['c2'],kwargs['d2']):
    resp2.append((a,b,c,d))

  change=[]
  for x,y,mask1 in zip(resp1,resp2,kwargs['mask1']):
    if np.argmax(x)!=np.argmax(y) and mask1==0:
      change.append(np.max(x)-sorted(x)[-2])
  return len(change), (np.mean(change) if len(change)>0 else 0.0)

def maximum_variation(**kwargs):
  resp2=[]
  resp1=[]
  
  for a,b,c,d in zip(kwargs['a1'],kwargs['b1'],kwargs['c1'],kwargs['d1']):  
    resp1.append((a,b,c,d))
  for a,b,c,d in zip(kwargs['a2'],kwargs['b2'],kwargs['c2'],kwargs['d2']):
    resp2.append((a,b,c,d))
  
  max_var=[]
  for x,y in zip(resp1,resp2):
    var=[]
    for opt1,opt2 in zip(x,y):
      var.append(opt2-opt1)
      #print(opt2-opt1,end=' ')
    max_var.append(np.max(var))
  
  return max_var[np.abs(max_var).argmax()],np.mean(max_var)


ARTIFACT_TO_FUNCTION={'number of answers changed':number_of_answers_changed,
                      'number of answers same':number_of_answers_same,
                      '#answers changed from correct':number_of_answers_changed_from_correct,
                      '#answers changed from wrong':number_of_answers_changed_from_wrong,
                      'maximum variation':maximum_variation,
}

#rootazurestoragequantenginestrtllama70bchatsmoothquantsq081gpuhendrycksTest-high_school_government_and_politics_batch_size_None__best_option_length.txt
def getinfo(file):
    model=''
    if file.find('llama70bchat')!=-1 or file.find('Llama270bchat') !=-1:
       model='llama70bchat'
    elif file.find('llama7bchat')!=-1 or file.find('Llama27bchat') !=-1:
       model='llama7bchat'
    elif file.find('llama13bchat')!=-1 or file.find('Llama213bchat') !=-1:
       model='llama13bchat'
    else:
       assert(False)
    

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
                    'best_option_length':'best option length',
                    'best_option_norm_length':'best option norm length',
                    'correct_option_length':'correct option length',
                    }
    for key_,value in possible_artifacts.items():
        if file.find(key_)!=-1:
            artifacts=value
            break
    if task=='' or artifacts=='':
       assert(False)

    return model,wrong_corr_norm,shot,task,artifacts


def get_map_config_to_data(agg_artifact,DIR):
  map_config_to_data=collections.defaultdict(lambda:np.array([]))
  #map_config_to_data={}
  for file in sorted(os.listdir(DIR)):
    
    if fnmatch.fnmatch(file,'*.txt') and fnmatch.fnmatch(file,'rootazurestorage*'):
      if fnmatch.fnmatch(file,'*gpupertokenchannel*'):
        continue
      model,wrong_corr_norm,shot,task,_artifacts=getinfo(file)
      #print(model,shot,wrong_corr_norm,task,_artifacts)
      if agg_artifact=='accuracy':
        if file.find('mask')!=-1:
          arr=np.loadtxt(DIR+file)
          map_config_to_data[model+'8bit'+wrong_corr_norm+shot+task+'accuracy']=np.mean(arr)*100.0
      elif agg_artifact=='clearance':
        if file.find('clearance')!=-1:
          arr=np.loadtxt(DIR+file)
          map_config_to_data[model+'8bit'+wrong_corr_norm+shot+task+'clearance']=np.mean(arr)*100.0
      elif agg_artifact=='number of answers changed' or agg_artifact=='number of answers same' or file.find('_mask')!=-1 or agg_artifact=='#answers changed from correct' or agg_artifact=='#answers changed from wrong' or agg_artifact=='maximum variation':
        if file.find('a_likelihoods')!=-1 or file.find('b_likelihoods')!=-1 or file.find('c_likelihoods')!=-1 or file.find('d_likelihoods')!=-1 or file.find('_mask')!=-1:
          arr=np.loadtxt(DIR+file)
          map_config_to_data[model+'8bit'+wrong_corr_norm+shot+task+_artifacts+agg_artifact]=arr
    
    if fnmatch.fnmatch(file,'*.txt') and fnmatch.fnmatch(file,'pretrained*'):
      model,wrong_corr_norm,shot,task,_artifacts=getinfo(file)
      if agg_artifact=='accuracy':
        if file.find('mask')!=-1:
          arr=np.loadtxt(DIR+file)
          map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+'accuracy']=np.mean(arr)*100.0
      elif agg_artifact=='clearance':
        if file.find('clearance')!=-1:
          arr=np.loadtxt(DIR+file)
          map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+'clearance']=np.mean(arr)*100.0
      elif agg_artifact=='number of answers changed' or agg_artifact=='number of answers same' or file.find('_mask')!=-1 or agg_artifact=='#answers changed from correct' or agg_artifact=='#answers changed from wrong' or agg_artifact=='maximum variation':
        if file.find('a_likelihoods')!=-1 or file.find('b_likelihoods')!=-1 or file.find('c_likelihoods')!=-1 or file.find('d_likelihoods')!=-1 or file.find('_mask')!=-1:
          arr=np.loadtxt(DIR+file)
          map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+_artifacts+agg_artifact]=arr



  return map_config_to_data






def print_info_qtz(agg_artifact,map_config_to_data,worksheet,row,col,model,quantization,wrong_corr_norm,shot,task):
  if agg_artifact=='accuracy' or agg_artifact=='clearance':
    if model+quantization+wrong_corr_norm+shot+task+agg_artifact in map_config_to_data:
        worksheet.write_number(row,col,map_config_to_data[model+quantization+wrong_corr_norm+shot+task+agg_artifact])
    else:
        worksheet.write_string(row,col,'NA')
  elif agg_artifact=='number of answers changed' or agg_artifact=='number of answers same'  or agg_artifact=='#answers changed from correct' or agg_artifact=='#answers changed from wrong' or agg_artifact=='maximum variation':
    a1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['a_likelihoods']+agg_artifact]
    b1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['b_likelihoods']+agg_artifact]
    c1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['c_likelihoods']+agg_artifact]
    d1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['d_likelihoods']+agg_artifact]

    a2=map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['a_likelihoods']+agg_artifact]
    b2=map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['b_likelihoods']+agg_artifact]
    c2=map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['c_likelihoods']+agg_artifact]
    d2=map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['d_likelihoods']+agg_artifact]

    mask1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['_mask']+agg_artifact]


    if a1.size==0 or a2.size==0:
      worksheet.write_string(row,col,'NA')
      worksheet.write_string(row,col+1,'NA')
    else:
        worksheet.write_number(row,col,ARTIFACT_TO_FUNCTION[agg_artifact](a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1)[0])
        worksheet.write_number(row,col+1,ARTIFACT_TO_FUNCTION[agg_artifact](a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1)[1])
  
    
    



def print_headings_qtz(worksheet):
  col=5
  for task in POSSIBLE_TASKS:
    worksheet.write_string(0,col+2,task)
    worksheet.write_string(1,col+2,'QUANTIZATION')
    worksheet.write_string(2,col,'16bit')
    worksheet.write_string(2,col+2,'8bit')
    col=col+4
    col=col+1



def dump_qtz_variation(agg_artifact,DIR,worksheet):
  print_headings_qtz(worksheet)

  worksheet.write_string(3,0,'Model')
  worksheet.write_string(3,1,'#shot')
  map_config_to_data=get_map_config_to_data(agg_artifact,DIR)
  
  row=7
  for model in POSSIBLE_MODELS:
    for shot in POSSIBLE_SHOT:
      worksheet.write_string(row,0,model)
      worksheet.write_string(row,1,shot)
      col=5
      for task in POSSIBLE_TASKS:
        for quantization in POSSIBLE_QTZ:
            print_info_qtz(agg_artifact,map_config_to_data,worksheet,row,col,model,quantization,'normal',shot,task)
            col=col+2
        col=col+1   
      row=row+1
    row=row+1
  row=row+1


for agg_artifact in (POSSIBLE_AGG_ARTIFACTS):
  #print(agg_artifact)
  worksheet_qtz = workbook_qtz.add_worksheet(agg_artifact)

  dump_qtz_variation(agg_artifact,'./hostvm/logs/',worksheet_qtz)

workbook_qtz.close()
