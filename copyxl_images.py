import xlsxwriter
import fnmatch
import os
from tqdm import tqdm
import numpy as np

workbook_batch = xlsxwriter.Workbook("Charts Batch Size Variation.xlsx")
workbook_qtz = xlsxwriter.Workbook("Charts Quantization Variation.xlsx")
POSSIBLE_ARTIFACTS= ['best_likelihoods','correct_likelihoods','top_margin','a_likelihoods','b_likelihoods',   'c_likelihoods','d_likelihoods','clearance']

POSSIBLE_TASKS=['abstract algebra','machine learning', 'philosophy', 'govt politics']
POSSIBLE_MODELS=['llama2-7b','llama2-7b-chat','llama2-13b','llama2-13b-chat']
POSSIBLE_QTZ=['16bit', '8bit', '4bit']
POSSIBLE_CAT=['normal', 'wrong', 'correct']
POSSIBLE_SHOT=['0shot',  '1shot', '3shot', '5shot']
POSSIBLE_BATCH=['batch=1', 'batch=5', 'batch=32']

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

def get_map_config_to_image(artifact,DIR):
  map_config_to_image={}
  for file in (os.listdir(DIR)):
    if fnmatch.fnmatch(file,'*.png'):
      if file.find(artifact)==-1:
        continue
      model,quantization,wrong_corr_norm,shot,batch_size,task,_artifacts=getinfo(file)
      map_config_to_image[model+quantization+wrong_corr_norm+shot+batch_size+task+artifact]=DIR+file
  return map_config_to_image

def get_map_config_to_image_batch1(artifact,DIR):
  map_config_to_image={}
  for file in (os.listdir(DIR)):
    if fnmatch.fnmatch(file,'*.png'):
      if file.find(artifact)==-1:
        continue
      
      model,quantization,wrong_corr_norm,shot,batch_size,task,_artifacts=getinfo(file)
      if batch_size=='batch=1':
        map_config_to_image[model+quantization+wrong_corr_norm+shot+batch_size+task+artifact]=DIR+file
  return map_config_to_image

def dump_batch_size_variation(artifact,DIR,worksheet):
  print_headings_batch(worksheet)
  worksheet.write_string(3,0,'Model')
  worksheet.write_string(3,1,'#shot')
  worksheet.write_string(3,2,'qtz')
  
  map_config_to_image=get_map_config_to_image(artifact,DIR)
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
                worksheet.set_row(row , 160)
                worksheet.set_column(col,col,36)
                #worksheet.write_string(row,col,'haha')
                worksheet.insert_image(row,col, map_config_to_image[model+quantization+'normal'+shot+batch_size+task+artifact],{'x_scale': 0.4, 'y_scale': 0.4}) 
                col=col+2
            col=col+1
        row=row+1
      row=row+1
    row=row+1
  row=row+1
      




def dump_qtz_variation(artifact,DIR,worksheet):
  print_headings_qtz(worksheet)

  worksheet.write_string(3,0,'Model')
  worksheet.write_string(3,1,'#shot')
  map_config_to_image_batch1=get_map_config_to_image_batch1(artifact,DIR)
  row=7
  for model in POSSIBLE_MODELS:
    for shot in POSSIBLE_SHOT:
      worksheet.write_string(row,0,model)
      worksheet.write_string(row,1,shot)
      col=5
      for task in POSSIBLE_TASKS:
        for quantization in POSSIBLE_QTZ:
            worksheet.set_row(row , 160)
            worksheet.set_column(col,col,36)
            #worksheet.write_string(row,col,'haha')
            worksheet.insert_image(row,col, map_config_to_image_batch1[model+quantization+'normal'+shot+'batch=1'+task+artifact],{'x_scale': 0.4, 'y_scale': 0.4})
            col=col+2
        col=col+1    
      row=row+1
    row=row+1
  row=row+1


def dump_batch_size_variation_multicategory(artifact,DIR,worksheet):
  print_headings_batch(worksheet)

  worksheet.write_string(3,0,'Model')
  worksheet.write_string(3,1,'#shot')
  worksheet.write_string(3,2,'qtz')
  worksheet.write_string(3,3,'Category')
  map_config_to_image=get_map_config_to_image(artifact,DIR)
  row=7
  for model in POSSIBLE_MODELS:
    for shot in POSSIBLE_SHOT:
      for quantization in POSSIBLE_QTZ:
        for category in POSSIBLE_CAT:
            
            worksheet.write_string(row,0,model)
            worksheet.write_string(row,1,shot)
            worksheet.write_string(row,2,quantization)
            worksheet.write_string(row,3,category)
            col=5
            for task in POSSIBLE_TASKS:
              for  batch_size in POSSIBLE_BATCH:
                #worksheet.write_string(row,col,'haha')
                worksheet.set_row(row , 160)
                worksheet.set_column(col,col,36)
                worksheet.insert_image(row,col, map_config_to_image[model+quantization+category+shot+batch_size+task+artifact],{'x_scale': 0.4, 'y_scale': 0.4})
                col=col+2
              col=col+1  
            row=row+1
        row=row+1
      row=row+1
    row=row+1
  row=row+1




def dump_qtz_variation_multicategory(artifact,DIR,worksheet):
  print_headings_qtz(worksheet)

  worksheet.write_string(3,0,'Model')
  worksheet.write_string(3,1,'#shot')
  worksheet.write_string(3,2,'Category')

  map_config_to_image=get_map_config_to_image_batch1(artifact,DIR)
  row=7
  for model in POSSIBLE_MODELS:
    for shot in POSSIBLE_SHOT:
      for category in POSSIBLE_CAT:
        worksheet.write_string(row,0,model)
        worksheet.write_string(row,1,shot)
        worksheet.write_string(row,2,category)
        col=5
        for task in POSSIBLE_TASKS:
            for quantization in POSSIBLE_QTZ:
                #worksheet.write_string(row,col,'haha')
                worksheet.set_row(row , 160)
                worksheet.set_column(col,col,36)
                worksheet.insert_image(row,col, map_config_to_image[model+quantization+category+shot+'batch=1'+task+artifact],{'x_scale': 0.4, 'y_scale': 0.4})
                col=col+2
            col=col+1
        row=row+1
      row=row+1
    row=row+1
  row=row+1



  


for artifact in tqdm(POSSIBLE_ARTIFACTS):
  worksheet_batch_lin = workbook_batch.add_worksheet(artifact)
  worksheet_batch_hist = workbook_batch.add_worksheet(artifact+'hist')
  worksheet_qtz_lin = workbook_qtz.add_worksheet(artifact)
  worksheet_qtz_hist = workbook_qtz.add_worksheet(artifact+'hist')

  if artifact=='best_likelihoods' or artifact=='top_margin':
    dump_batch_size_variation_multicategory(artifact,'./hostvm/artifacts/batch/',worksheet_batch_lin)
    dump_batch_size_variation_multicategory(artifact,'./hostvm/artifacts/histogram/',worksheet_batch_hist)
    dump_qtz_variation_multicategory(artifact,'./hostvm/artifacts/quantization/',worksheet_qtz_lin)
    dump_qtz_variation_multicategory(artifact,'./hostvm/artifacts/histogram/',worksheet_qtz_hist)
  else:
    dump_batch_size_variation(artifact,'./hostvm/artifacts/batch/',worksheet_batch_lin)
    dump_batch_size_variation(artifact,'./hostvm/artifacts/histogram/',worksheet_batch_hist)
    dump_qtz_variation(artifact,'./hostvm/artifacts/quantization/',worksheet_qtz_lin)
    dump_qtz_variation(artifact,'./hostvm/artifacts/histogram/',worksheet_qtz_hist)



workbook_batch.close()
workbook_qtz.close()

#llama2-7b16bitnormal0shotbatch=1abstract algebracorrect_likelihoods