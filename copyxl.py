import xlsxwriter
import fnmatch
import os
workbook = xlsxwriter.Workbook("small_results7.xlsx")
ARTIFACT_PATTERN='top_margin'
TYPE=True
MAX_ROW=95 if TYPE else 35
WORKSHEET_NAME=ARTIFACT_PATTERN
DIR='./artifacts/'

worksheet = workbook.add_worksheet(WORKSHEET_NAME)
worksheet.set_column(4, 4, 90)
worksheet.set_column(6, 6, 90)
worksheet.set_column(8, 8, 90)
worksheet.write_string(8,4,'16bit')
worksheet.write_string(8,6,'8bit')
worksheet.write_string(8,8,'4bit')


for i in range(9,MAX_ROW):
    worksheet.set_row(i , 409)
row_table={
  'mpt7b0shotnormal': 10,
  'mpt7b0shotwrong': 11,
  'mpt7b0shotcorrect': 12,
  'mpt7b1shotnormal': 14,
  'mpt7b1shotwrong': 15,
  'mpt7b1shotcorrect': 16,
  'mpt7b3shotnormal': 18,
  'mpt7b3shotwrong': 19,
  'mpt7b3shotcorrect': 20,
  'mpt7b5shotnormal': 22,
  'mpt7b5shotwrong': 23,
  'mpt7b5shotcorrect': 24,
  'mpt30b0shotnormal': 26,
  'mpt30b0shotwrong': 27,
  'mpt30b0shotcorrect': 28,
  'mpt30b1shotnormal': 30,
  'mpt30b1shotwrong': 31,
  'mpt30b1shotcorrect': 32,
  'mpt30b3shotnormal': 34,
  'mpt30b3shotwrong': 35,
  'mpt30b3shotcorrect': 36,
  'mpt30b5shotnormal': 38,
  'mpt30b5shotwrong': 39,
  'mpt30b5shotcorrect': 40,
  'llama7b0shotnormal': 42,
  'llama7b0shotwrong': 43,
  'llama7b0shotcorrect': 44,
  'llama7b1shotnormal': 46,
  'llama7b1shotwrong': 47,
  'llama7b1shotcorrect': 48,
  'llama7b3shotnormal': 50,
  'llama7b3shotwrong': 51,
  'llama7b3shotcorrect': 52,
  'llama7b5shotnormal': 54,
  'llama7b5shotwrong': 55,
  'llama7b5shotcorrect': 56,
  'llama13b0shotnormal': 58,
  'llama13b0shotwrong': 59,
  'llama13b0shotcorrect': 60,
  'llama13b1shotnormal': 62,
  'llama13b1shotwrong': 63,
  'llama13b1shotcorrect': 64,
  'llama13b3shotnormal': 66,
  'llama13b3shotwrong': 67,
  'llama13b3shotcorrect': 68,
  'llama13b5shotnormal': 70,
  'llama13b5shotwrong': 71,
  'llama13b5shotcorrect': 72,
}

row_table_unitary={
  'mpt7b0shotnormal': 10,
  'mpt7b1shotnormal': 11,
  'mpt7b3shotnormal': 12,
  'mpt7b5shotnormal': 13,
  'mpt30b0shotnormal': 16,
  'mpt30b1shotnormal': 17,
  'mpt30b3shotnormal': 18,
  'mpt30b5shotnormal': 19,
  'llama7b0shotnormal': 22,
  'llama7b1shotnormal': 23,
  'llama7b3shotnormal': 24,
  'llama7b5shotnormal': 25,
  'llama13b0shotnormal': 28,
  'llama13b1shotnormal': 29,
  'llama13b3shotnormal': 30,
  'llama13b5shotnormal': 31,

}

col_table={
    '16':4,
    '8':6,
    '4':8,
}
# Write the byte stream image to a cell. The filename must  be specified.
for file in os.listdir(DIR):
  if fnmatch.fnmatch(file,'*.png'):
    if file.find(ARTIFACT_PATTERN)==-1:
      continue

    #pretrainedmetallamaLlama27bchathffewshot0_best_likelihoods.txt.png
    #pretrainedmosaicmlmpt7binstructloadin4bitTruefewshot3_best_likelihoods.txt.png
    #pretrainedmosaicmlmpt7binstructloadin4bitTruefewshot0_correct_exp_loglikelihoods.txt
    model=''

    if file[11]=='e':
      if file[25]=='7':
        model='llama7b'
      elif file[25]=='1':
        model='llama13b'
    elif file[11]=='o':
      if file[21]=='3':
        model='mpt30b'
      elif file[21]=='7':
        model='mpt7b'

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

    if model=='' or (TYPE and type=='')  or shot=='' or quantization=='':
      continue

    rowname = model+shot+type if TYPE else model+shot+'normal'
    colname = quantization

    row_used = row_table if TYPE else row_table_unitary
    row=row_used[rowname]
    col=col_table[colname]
    if model=='llama7b' or model=='llama13b':
      model=model+'-chat'
    if model=='mpt7b':
      model=model+'-instruct'
    #30b was not instruct
    worksheet.write_string(row,0,model)
    worksheet.write_string(row,1,shot)
    if TYPE:
        worksheet.write_string(row,2,type)
    print(file)
    worksheet.insert_image(row,col, DIR+file)


workbook.close()