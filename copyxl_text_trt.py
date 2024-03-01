import xlsxwriter
import fnmatch
import os
from tqdm import tqdm
import numpy as np
import collections
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import hist
from functools import cmp_to_key
import copyxl_utils
import json
import pandas as pd

workbook_qtz = xlsxwriter.Workbook("empty.xlsx")
cell_format = workbook_qtz.add_format()
cell_format.set_font_size(8)

POSSIBLE_AGG_ARTIFACTS= ['accuracy','clearance','% of answers changed','% flipped','% from correct to wrong','% from wrong to correct','number of answers same', 'maximum variation','top_margin','best_likelihoods', 'correct_likelihoods']

POSSIBLE_ARTIFACTS=['best likelihood','ground truth likelihood','margin between best 2','A option likelihood','B option likelihood','C option likelihood','D option likelihood','clearance','mask','accuracy']
ARTIFACT_TO_ENGLISH= {'best_likelihoods': 'best likelihood',
                    'correct_likelihoods':'ground truth likelihood',
                    'top_margin':'margin between best 2',
                    'a_likelihoods':'A option likelihood',
                    'b_likelihoods':'B option likelihood',
                    'c_likelihoods':'C option likelihood',
                    'd_likelihoods':'D option likelihood',
                    'clearance': 'clearance',
                    '_mask': 'mask',
                    'accuracy':'mask',
                    }


#POSSIBLE_TASKS=['abstract algebra','machine learning', 'philosophy', 'govt politics']
#POSSIBLE_TASKS=['abstract algebra', 'machine learning', 'philosophy', 'govt politics']
POSSIBLE_TASKS=['hendrycksTest-abstract_algebra', 'hendrycksTest-anatomy', 'hendrycksTest-astronomy', 'hendrycksTest-business_ethics', 'hendrycksTest-clinical_knowledge', 'hendrycksTest-college_biology', 'hendrycksTest-college_chemistry', 'hendrycksTest-college_computer_science', 'hendrycksTest-college_mathematics', 'hendrycksTest-college_medicine', 'hendrycksTest-college_physics', 'hendrycksTest-computer_security', 'hendrycksTest-conceptual_physics', 'hendrycksTest-econometrics', 'hendrycksTest-electrical_engineering', 'hendrycksTest-elementary_mathematics', 'hendrycksTest-formal_logic', 'hendrycksTest-global_facts', 'hendrycksTest-high_school_biology', 'hendrycksTest-high_school_chemistry', 'hendrycksTest-high_school_computer_science', 'hendrycksTest-high_school_european_history', 'hendrycksTest-high_school_geography', 'hendrycksTest-high_school_government_and_politics', 'hendrycksTest-high_school_macroeconomics', 'hendrycksTest-high_school_mathematics', 'hendrycksTest-high_school_microeconomics', 'hendrycksTest-high_school_physics', 'hendrycksTest-high_school_psychology',	'hendrycksTest-high_school_statistics', 'hendrycksTest-high_school_us_history', 'hendrycksTest-high_school_world_history', 'hendrycksTest-human_aging', 'hendrycksTest-human_sexuality', 'hendrycksTest-international_law', 'hendrycksTest-jurisprudence', 'hendrycksTest-logical_fallacies', 'hendrycksTest-machine_learning', 'hendrycksTest-management', 'hendrycksTest-marketing', 'hendrycksTest-medical_genetics', 'hendrycksTest-miscellaneous', 'hendrycksTest-moral_disputes', 'hendrycksTest-moral_scenarios', 'hendrycksTest-nutrition', 'hendrycksTest-philosophy', 'hendrycksTest-prehistory', 'hendrycksTest-professional_accounting', 'hendrycksTest-professional_law', 'hendrycksTest-professional_medicine', 'hendrycksTest-professional_psychology', 'hendrycksTest-public_relations', 'hendrycksTest-security_studies', 'hendrycksTest-sociology', 'hendrycksTest-us_foreign_policy', 'hendrycksTest-virology', 'hendrycksTest-world_religions']
POSSIBLE_MODELS=['Llama2-7bchat','Llama2-13bchat','Llama2-70bchat','Yi-34bchat','Yi-6bchat']
POSSIBLE_CAT=['normal', 'wrong', 'correct']
POSSIBLE_SHOT=['5shot']


ARTIFACT_TO_FUNCTION={'% of answers changed':copyxl_utils.percent_of_answers_changed,
                      '% flipped':copyxl_utils.percent_of_answers_flipped,
                      '% from correct to wrong':copyxl_utils.percent_of_answers_changed_from_correct_to_wrong,
                      '% from wrong to correct':copyxl_utils.percent_of_answers_changed_from_wrong_to_correct,
                      'maximum variation':copyxl_utils.maximum_variation,
                      'number of answers same':copyxl_utils.number_of_answers_same,
                      'kl div':copyxl_utils.kl_div,
}

#rootazurestoragequantenginestrtllama70bchatsmoothquantsq081gpuhendrycksTest-high_school_government_and_politics_batch_size_None__best_option_length.txt


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
    mask2=map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['_mask']+agg_artifact]


    if a1.size==0 or a2.size==0:
      #print('print_info_qtz',model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['a_likelihoods']+agg_artifact)
      
      worksheet.write_string(row,col,'NA')
      worksheet.write_string(row,col+1,'NA')
    else:
        #worksheet.write_number(row,col,ARTIFACT_TO_FUNCTION[agg_artifact](a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1)[0])
        #worksheet.write_number(row,col+1,ARTIFACT_TO_FUNCTION[agg_artifact](a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1)[1])
        if quantization != '16bit':
          #analyzeflips2(a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1,mask2=mask2,label=model+'AWQ'+shot+task)
          #analyzeflips(a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1,mask2=mask2)
          #analyzeflips3(a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1,mask2=mask2,label=model+'AWQ'+shot+task)
          #number_of_changes_to_level(a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1,mask2=mask2,task=task,label=model+'AWQ'+shot+task)
          #validating_hypothesis(map_config_to_data=map_config_to_data)

          pass
  
def print_info_qtz_general(agg_artifact,map_config_to_data,worksheet,row,col,model,quantization,wrong_corr_norm,shot,task):
  if agg_artifact=='accuracy' or agg_artifact=='clearance' or agg_artifact=='top_margin' or agg_artifact=='best_likelihoods' or agg_artifact=='correct_likelihoods':
    if model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH[agg_artifact] in map_config_to_data:
        worksheet.write_number(row,col,np.mean(map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH[agg_artifact]]))

        f=open(agg_artifact+'.jsonl','a')
        df = pd.DataFrame(columns=['model','quantization','task','agg_artifact'])
        
        df.loc[-1] = [model,quantization,task,np.mean(map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH[agg_artifact]])]  # adding a row
        df.index = df.index + 1  # shifting index
        df = df.sort_index()  # sorting by index
        f.write(df.to_json(orient="records")[1:-1]+'\n')

    else:
        #print('f ',model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH[agg_artifact])
        worksheet.write_string(row,col,'NA')

  elif agg_artifact=='% of answers changed' or agg_artifact=='% flipped'  or agg_artifact=='% from correct to wrong' or agg_artifact=='% from wrong to correct' or agg_artifact=='maximum variation' or agg_artifact=='number of answers same' or agg_artifact=='kl div':
    a1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['a_likelihoods']]
    b1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['b_likelihoods']]
    c1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['c_likelihoods']]
    d1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['d_likelihoods']]

    a2=map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['a_likelihoods']]
    b2=map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['b_likelihoods']]
    c2=map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['c_likelihoods']]
    d2=map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['d_likelihoods']]

    mask1=map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['_mask']]
    mask2=map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['_mask']]


    if a1.size==0 or a2.size==0:
      #print('f ',model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['a_likelihoods'])
      
      worksheet.write_string(row,col,'NA')
      worksheet.write_string(row,col+1,'NA')
    else:
        #plot_usefullness(a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1,mask2=mask2,task=task,model=model,quantization=quantization)
        worksheet.write_number(row,col,ARTIFACT_TO_FUNCTION[agg_artifact](a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1,mask2=mask2)[0])
        worksheet.write_number(row,col+1,ARTIFACT_TO_FUNCTION[agg_artifact](a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1,mask2=mask2)[1],cell_format)

        f=open(agg_artifact+'.jsonl','a')
        df = pd.DataFrame(columns=['model','quantization','task','agg_artifact'])
        
        df.loc[-1] = [model,quantization,task,ARTIFACT_TO_FUNCTION[agg_artifact](a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1,mask2=mask2)[0]]  # adding a row
        df.index = df.index + 1  # shifting index
        df = df.sort_index()  # sorting by index
        f.write(df.to_json(orient="records")[1:-1]+'\n')
        
          
      
    



def print_headings_qtz(worksheet):
  col=3
  for task in POSSIBLE_TASKS:
    worksheet.write_string(0,col+2,task)
    worksheet.write_string(1,col+2,'QUANTIZATION')
    worksheet.write_string(2,col,'16bit')
    worksheet.write_string(2,col+2,'BnB 8bit')
    worksheet.write_string(2,col+4,'GPTQ 8bit')
    worksheet.write_string(2,col+6,'SQ 8bit')
    worksheet.write_string(2,col+8,'GPTQ 4bit')
    worksheet.write_string(2,col+10,'AWQ 4bit')
    worksheet.write_string(2,col+12,'BnB 4bit')

    col=col+14
    col=col+1

def print_headings_qtz_fullmmlu(worksheet):
  col=3
  worksheet.write_string(0,col+2,"FULL MMLU")
  worksheet.write_string(1,col+2,'QUANTIZATION')
  worksheet.write_string(2,col,'16bit')
  worksheet.write_string(2,col+2,'BnB 8bit')
  worksheet.write_string(2,col+4,'GPTQ 8bit')
  worksheet.write_string(2,col+6,'SQ 8bit')
  worksheet.write_string(2,col+8,'GPTQ 4bit')
  worksheet.write_string(2,col+10,'AWQ 4bit')
  worksheet.write_string(2,col+12,'BnB 4bit')


def dump_qtz_variation(agg_artifact,worksheet,map_config_to_data):
  print_headings_qtz(worksheet)
  worksheet.write_string(3,0,'Model')
  row=7
  for model in POSSIBLE_MODELS:
    worksheet.write_string(row,0,model)
    col=3
    for task in POSSIBLE_TASKS:
      for quantization in ['16bit','BnB8bit','GPTQ8bit','SQ8bit','GPTQ4bit','AWQ4bit','BnB4bit']:
        print_info_qtz_general(agg_artifact,map_config_to_data,worksheet,row,col,model,quantization,'normal','fewshot5',task)
        col=col+2
      col=col+1     
    row=row+1

def dump_qtz_variation_fullmmlu(agg_artifact,worksheet,map_config_to_data_fullmmlu):
  print_headings_qtz_fullmmlu(worksheet)
  worksheet.write_string(3,0,'Model')
  row=7
  for model in POSSIBLE_MODELS:
    worksheet.write_string(row,0,model)
    col=3
    for quantization in ['16bit','BnB8bit','GPTQ8bit','SQ8bit','GPTQ4bit','AWQ4bit','BnB4bit']:
      print_info_qtz_general(agg_artifact,map_config_to_data_fullmmlu,worksheet,row,col,model,quantization,'normal','fewshot5','')
      col=col+2
    col=col+1     
    row=row+1




DIR='./hostvm/logs/'
map_config_to_data=copyxl_utils.get_map_config_to_data_general(DIR)
map_config_to_data_fullmmlu=copyxl_utils.get_map_config_to_data_general_fullmmlu(map_config_to_data)
#print(sorted(map_config_to_data_fullmmlu.keys()))

def print_artifacts():
  for agg_artifact in (['accuracy']):
    #print(agg_artifact)
    worksheet_qtz = workbook_qtz.add_worksheet(agg_artifact)
    #print(sorted(map_config_to_data.keys()))
    dump_qtz_variation(agg_artifact,worksheet_qtz,map_config_to_data)


def fullmmlu_print_artifacts():
  for agg_artifact in (['kl div','% of answers changed']):
    worksheet_qtz = workbook_qtz.add_worksheet(agg_artifact)
    dump_qtz_variation_fullmmlu(agg_artifact,worksheet_qtz,map_config_to_data_fullmmlu)


#fullmmlu_print_artifacts()
print_artifacts()
#copyxl_utils.analyze_variation_wrt_noise()
workbook_qtz.close()
