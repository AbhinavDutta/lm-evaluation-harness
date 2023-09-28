import xlsxwriter
import fnmatch
import os
from tqdm import tqdm
import numpy as np

workbook_batch = xlsxwriter.Workbook("Numbers Batch Size Variation.xlsx")
workbook_qtz = xlsxwriter.Workbook("Numbers Quantization Variation.xlsx")
POSSIBLE_AGG_ARTIFACTS= ['accuracy','clearance','number of answers changed','number of answers same']


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
POSSIBLE_MODELS=['llama2-7b','llama2-7b-chat','llama2-13b','llama2-13b-chat']
POSSIBLE_QTZ=['16bit', '8bit', '4bit']
POSSIBLE_CAT=['normal', 'wrong', 'correct']
POSSIBLE_SHOT=['0shot',  '1shot', '3shot', '5shot']
POSSIBLE_BATCH=['batch=1', 'batch=5', 'batch=32']



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
      change.append(np.max(x))
  return len(change), (np.mean(change) if len(change)>0 else 0.0)
  
def number_of_answers_same(**kwargs):
  resp2=[]
  resp1=[]
  for a,b,c,d in zip(kwargs['a1'],kwargs['b1'],kwargs['c1'],kwargs['d1']):  
    resp1.append((a,b,c,d))
  for a,b,c,d in zip(kwargs['a1'],kwargs['b1'],kwargs['c1'],kwargs['d1']):
    resp2.append((a,b,c,d))

  change=[]
  for x,y in zip(resp1,resp2):
    if np.argmax(x)==np.argmax(y):
      change.append(np.max(x))
  return len(change),(np.mean(change) if len(change)>0 else 0.0)

ARTIFACT_TO_FUNCTION={'number of answers changed':number_of_answers_changed,
                      'number of answers same':number_of_answers_same,
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


def get_map_config_to_data(agg_artifact,DIR):
  map_config_to_data={}
  for file in (os.listdir(DIR)):
    if fnmatch.fnmatch(file,'*.txt') and fnmatch.fnmatch(file,'pretrained*'):
      model,quantization,wrong_corr_norm,shot,batch_size,task,_artifacts=getinfo(file)
      if agg_artifact=='accuracy':
        if file.find('mask')!=-1:
          arr=np.loadtxt(DIR+file)
          map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+'accuracy']=np.mean(arr)*100.0
      elif agg_artifact=='clearance':
        if file.find('clearance')!=-1:
          arr=np.loadtxt(DIR+file)
          map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+'clearance']=np.mean(arr)*100.0
      elif agg_artifact=='number of answers changed' or agg_artifact=='number of answers same':
        if file.find('a_likelihoods')!=-1 or file.find('b_likelihoods')!=-1 or file.find('c_likelihoods')!=-1 or file.find('d_likelihoods')!=-1:
          arr=np.loadtxt(DIR+file)
          map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+_artifacts+agg_artifact]=arr



  return map_config_to_data




def print_info_batch(agg_artifact,map_config_to_data,worksheet,row,col,model,quantization,wrong_corr_norm,shot,batch_size,task):
  if agg_artifact=='accuracy' or agg_artifact=='clearance':
    worksheet.write_number(row,col,map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+agg_artifact])
  elif agg_artifact=='number of answers changed' or agg_artifact=='number of answers same':
    a1=map_config_to_data[model+quantization+wrong_corr_norm+shot+'batch=1'+task+ARTIFACT_TO_ENGLISH['a_likelihoods']+agg_artifact]
    b1=map_config_to_data[model+quantization+wrong_corr_norm+shot+'batch=1'+task+ARTIFACT_TO_ENGLISH['b_likelihoods']+agg_artifact]
    c1=map_config_to_data[model+quantization+wrong_corr_norm+shot+'batch=1'+task+ARTIFACT_TO_ENGLISH['c_likelihoods']+agg_artifact]
    d1=map_config_to_data[model+quantization+wrong_corr_norm+shot+'batch=1'+task+ARTIFACT_TO_ENGLISH['d_likelihoods']+agg_artifact]

    a2=map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+ARTIFACT_TO_ENGLISH['a_likelihoods']+agg_artifact]
    b2=map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+ARTIFACT_TO_ENGLISH['b_likelihoods']+agg_artifact]
    c2=map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+ARTIFACT_TO_ENGLISH['c_likelihoods']+agg_artifact]
    d2=map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+ARTIFACT_TO_ENGLISH['d_likelihoods']+agg_artifact]
    worksheet.write_number(row,col,ARTIFACT_TO_FUNCTION[agg_artifact](a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2)[0])
    worksheet.write_number(row,col+1,ARTIFACT_TO_FUNCTION[agg_artifact](a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2)[1])
   

def print_info_qtz(agg_artifact,map_config_to_data,worksheet,row,col,model,quantization,wrong_corr_norm,shot,batch_size,task):
  if agg_artifact=='accuracy' or agg_artifact=='clearance':
    worksheet.write_number(row,col,map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+agg_artifact])
  elif agg_artifact=='number of answers changed' or agg_artifact=='number of answers same':
    a1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+'batch=1'+task+ARTIFACT_TO_ENGLISH['a_likelihoods']+agg_artifact]
    b1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+'batch=1'+task+ARTIFACT_TO_ENGLISH['b_likelihoods']+agg_artifact]
    c1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+'batch=1'+task+ARTIFACT_TO_ENGLISH['c_likelihoods']+agg_artifact]
    d1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+'batch=1'+task+ARTIFACT_TO_ENGLISH['d_likelihoods']+agg_artifact]

    a2=map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+ARTIFACT_TO_ENGLISH['a_likelihoods']+agg_artifact]
    b2=map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+ARTIFACT_TO_ENGLISH['b_likelihoods']+agg_artifact]
    c2=map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+ARTIFACT_TO_ENGLISH['c_likelihoods']+agg_artifact]
    d2=map_config_to_data[model+quantization+wrong_corr_norm+shot+batch_size+task+ARTIFACT_TO_ENGLISH['d_likelihoods']+agg_artifact]
    worksheet.write_number(row,col,ARTIFACT_TO_FUNCTION[agg_artifact](a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2)[0])
    worksheet.write_number(row,col+1,ARTIFACT_TO_FUNCTION[agg_artifact](a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2)[1])
    #print(ARTIFACT_TO_FUNCTION[agg_artifact](a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2)[1])
    


def print_headings_batch(worksheet):
  col=5
  for task in POSSIBLE_TASKS:
    worksheet.write_string(0,col+2,task)
    worksheet.write_string(1,col+2,'BATCH SIZE')
    worksheet.write_number(2,col,1)
    worksheet.write_number(2,col+2,5)
    worksheet.write_number(2,col+4,32)
    col=col+6
    col=col+1

def print_headings_qtz(worksheet):
  col=5
  for task in POSSIBLE_TASKS:
    worksheet.write_string(0,col+2,task)
    worksheet.write_string(1,col+2,'QUANTIZATION')
    worksheet.write_string(2,col,'16bit')
    worksheet.write_string(2,col+2,'8bit')
    worksheet.write_string(2,col+4,'4bit')
    col=col+6
    col=col+1

def dump_batch_size_variation(agg_artifact,DIR,worksheet):
  print_headings_batch(worksheet)


  worksheet.write_string(3,0,'Model')
  worksheet.write_string(3,1,'#shot')
  worksheet.write_string(3,2,'qtz')
  
  map_config_to_data=get_map_config_to_data(agg_artifact,DIR)

  row=7
  for model in POSSIBLE_MODELS:
    for shot in POSSIBLE_SHOT:
      for quantization in POSSIBLE_QTZ:            
        worksheet.write_string(row,0,model)
        worksheet.write_string(row,1,shot)
        worksheet.write_string(row,2,quantization)
        col=5
        for task in POSSIBLE_TASKS:
            for batch_size in POSSIBLE_BATCH:
                print_info_batch(agg_artifact,map_config_to_data,worksheet,row,col,model,quantization,'normal',shot,batch_size,task)
                col=col+2    
            col=col+1
        row=row+1
      row=row+1
    row=row+1
  row=row+1
      




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
            print_info_qtz(agg_artifact,map_config_to_data,worksheet,row,col,model,quantization,'normal',shot,'batch=1',task)
            col=col+2
        col=col+1   
      row=row+1
    row=row+1
  row=row+1


for agg_artifact in (POSSIBLE_AGG_ARTIFACTS):
  print(agg_artifact)
  worksheet_batch = workbook_batch.add_worksheet(agg_artifact)
  worksheet_qtz = workbook_qtz.add_worksheet(agg_artifact)

  dump_batch_size_variation(agg_artifact,'./hostvm/logs/',worksheet_batch)
  dump_qtz_variation(agg_artifact,'./hostvm/logs/',worksheet_qtz)

workbook_batch.close()
workbook_qtz.close()