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



workbook_qtz = xlsxwriter.Workbook("empty.xlsx")
POSSIBLE_AGG_ARTIFACTS= ['accuracy','clearance','number of answers changed','number of answers same','#answers changed from correct','#answers changed from wrong', 'maximum variation']
POSSIBLE_QTZ=['16bit','4bit']



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


def silhouette_score_(list1,list2):
    s=0
    for pt in list1:
        intra_cluster_distance=0
        inter_cluster_distance=0
        for pt2 in list1:
            intra_cluster_distance+= abs(pt-pt2)
        for pt2 in list2:
            inter_cluster_distance+= abs(pt-pt2)
        inter_cluster_distance = inter_cluster_distance/len(list2)
        intra_cluster_distance = 0 if len(list1)-1==0 else intra_cluster_distance/(len(list1)-1)
        #print(inter_cluster_distance,intra_cluster_distance)
        s+= (inter_cluster_distance - intra_cluster_distance)/max(inter_cluster_distance,intra_cluster_distance) if max(inter_cluster_distance,intra_cluster_distance)>0 else 0
        #print((inter_cluster_distance - intra_cluster_distance)/max(inter_cluster_distance,intra_cluster_distance))
    for pt in list2:
        intra_cluster_distance=0
        inter_cluster_distance=0
        for pt2 in list2:
            intra_cluster_distance+= abs(pt-pt2)
        for pt2 in list1:
            inter_cluster_distance+= abs(pt-pt2)
        inter_cluster_distance = inter_cluster_distance/len(list1)
        intra_cluster_distance = 0 if len(list2)-1==0 else intra_cluster_distance/(len(list2)-1)
        s+= (inter_cluster_distance - intra_cluster_distance)/max(inter_cluster_distance,intra_cluster_distance) if max(inter_cluster_distance,intra_cluster_distance)>0 else 0
        #print((inter_cluster_distance - intra_cluster_distance)/max(inter_cluster_distance,intra_cluster_distance))
    
    return s/(len(list1)+len(list2))


def cluster(list):
    list = sorted(list,reverse=True)
    sil = []
    for r in range(0,len(list)-1):
        list1 = list[0:r+1]
        list2 = list[r+1:]
        
        sil.append(silhouette_score_(list1,list2))
        print(r,list1,list2,silhouette_score_(list1,list2))
    print(sil)
    return (np.argmax(sil)),sil
 
def entropy_(list):
  entr=0
  for e in list:
    entr+= -e*np.log(e)/np.log(4)
  return entr


def plot(fname,options,a1,b1,c1,d1,a2,b2,c2,d2,type):
  
  probs = {
    'original': (a1, b1, c1, d1),
    'quantized': (a2, b2, c2, d2),
  }

  x = np.arange(len(options))  # the label locations
  width = 0.25  # the width of the bars
  multiplier = 0

  fig, ax = plt.subplots(layout='constrained')
  #plt.title('radius = '+str(cluster([a1,b1,c1,d1])))

  for option, prob in probs.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, prob, width, label=option)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Probability')
  ax.set_title(type+' radius = '+str(cluster([a1,b1,c1,d1])[0]))
  ax.set_xticks(x + width, options)
  ax.set_xlabel(str(cluster([a1,b1,c1,d1])[1]))
  ax.legend(loc='upper right', ncols=2)
  
  plt.savefig(fname)
  print(fname)
  plt.close()
  #exit()

def plot_hist(c2c,w2wdiff,w2wsame,c2w,w2c,name_,fname):
  
  fig, ax = plt.subplots( nrows=2, ncols=3,layout='constrained' )  # create figure & 1 axis
  if len(c2c)>3:
    hist(c2c, bins='freedman', ax=ax[0][0], histtype='stepfilled',alpha=1.0, density=False)
  if len(w2wdiff)>3:
    hist(w2wdiff, bins='freedman', ax=ax[0][1], histtype='stepfilled',alpha=1.0, density=False)
  if len(w2wsame)>3:
    hist(w2wsame, bins='freedman', ax=ax[0][2], histtype='stepfilled',alpha=1.0, density=False)
  if len(c2w)>3:
    hist(c2w, bins='freedman', ax=ax[1][0], histtype='stepfilled',alpha=1.0, density=False)
  if len(w2c)>3:
    hist(w2c, bins='freedman', ax=ax[1][1], histtype='stepfilled',alpha=1.0, density=False)


  label='c2c\n total= '+str(len(c2c))+str(' mean= %.5f'%np.mean(c2c))#+str(' std dev= %.5f '%np.std(c2c))
  ax[0][0].set_xlabel(label,wrap=True,labelpad=10)    

  label='w2w diff\n total= '+str(len(w2wdiff))+str(' mean= %.5f'%np.mean(w2wdiff))#+str(' std dev= %.5f '%np.std(w2wdiff))
  ax[0][1].set_xlabel(label,wrap=True,labelpad=10)      

  label='w2w same\n total= '+str(len(w2wsame))+str(' mean= %.5f'%np.mean(w2wsame))#+str(' std dev= %.5f '%np.std(w2wsame))
  ax[0][2].set_xlabel(label,wrap=True, labelpad=10)      

  label='c2w\n total= '+str(len(c2w))+str(' mean= %.5f'%np.mean(c2w))#+str(' std dev= %.5f '%np.std(c2w))
  ax[1][0].set_xlabel(label,wrap=True,labelpad=10)      

  label='w2c\n total= '+str(len(w2c))+str(' mean= %.5f'%np.mean(w2c))#+str(' std dev= %.5f '%np.std(w2c))
  ax[1][1].set_xlabel(label,wrap=True,labelpad=10)   

  fig.suptitle('TOP MARGIN' +name_)
        
  fig.savefig(fname)   # save the figure to file
  plt.close(fig)    # close the figure window
  print(name_)

def plot_hist2(same,diff,name_,fname):
  same = np.sort(same)
  diff = np.sort(diff)
  fig, ax = plt.subplots( nrows=1, ncols=4,layout='constrained', figsize=(40,10) )  # create figure & 1 axis
  if len(same)>3:
    y1=hist(same, bins='freedman', ax=ax[0], histtype='stepfilled',alpha=1.0, density=False)
  if len(diff)>3:
    y2=hist(diff, bins='freedman', ax=ax[1], histtype='stepfilled',alpha=1.0, density=False)

  print(y1[1])
  print(y2[1])
  combined_width = np.sort(np.append(y2[1],y1[1]))
  print(combined_width)  
  combined_height = [] # %flips wwrt top_margin
  combined_height2 = [] #%same wrt top_margin
  for i in range(0,len(combined_width)-1):
    left=combined_width[i]
    right=combined_width[i+1]
    diff_cnt = 0
    same_cnt = 0
    percentflip = 0
    for x in same:
      if x>=left and x<right:
        same_cnt+=1
      if x>right:
        break
    for x in diff:
      if x>=left and x<=right:
        diff_cnt+=1
      if x>right:
        break
    percentflip = diff_cnt/(diff_cnt+same_cnt) if diff_cnt+same_cnt>0 else 0
    percentsame = same_cnt/(diff_cnt+same_cnt) if diff_cnt+same_cnt>0 else 0
    combined_height.append(percentflip)
    combined_height2.append(percentsame)

  print(combined_height)
  print(combined_height2)
  assert(len(combined_height)==len(combined_width)-1)
  assert(len(combined_height2)==len(combined_width)-1)
  label='same\n total= '+str(len(same))+str(' mean= %.5f'%np.mean(same))#+str(' std dev= %.5f '%np.std(c2c))
  ax[0].set_xlabel(label,wrap=True,labelpad=10)    

  label='diff\n total= '+str(len(diff))+str(' mean= %.5f'%np.mean(diff))#+str(' std dev= %.5f '%np.std(w2wdiff))
  ax[1].set_xlabel(label,wrap=True,labelpad=10)      

  label = 'percent flips'
  ax[2].bar(x=combined_width[:-1], height=combined_height, width=np.diff(combined_width), align='edge', fc='MediumOrchid', ec='black')
  ax[2].set_xlabel(label,wrap=True, labelpad=10)

  label = 'percent same'
  ax[3].bar(x=combined_width[:-1], height=combined_height2, width=np.diff(combined_width), align='edge', fc='MediumOrchid', ec='black')
  ax[3].set_xlabel(label,wrap=True, labelpad=10)
  ax[3].set_xlabel(label,wrap=True, labelpad=10)

  ax[2].tick_params(axis="x", labelsize=20,rotation=45)
  ax[3].tick_params(axis="x", labelsize=20,rotation=45)

  ax[0].set_xticks(y1[1])
  ax[1].set_xticks(y2[1])

  ax[2].set_xticks(combined_width)
  ax[3].set_xticks(combined_width)
  #plt.tight_layout()
  fig.suptitle('TOP MARGIN ')      

  fig.savefig(fname)   # save the figure to file
  plt.close(fig)    # close the figure window
  print(name_)



def analyzeflips3(**kwargs):
  resp2=[]
  resp1=[]
  for a,b,c,d in zip(kwargs['a1'],kwargs['b1'],kwargs['c1'],kwargs['d1']):  
    resp1.append((a,b,c,d))
  for a,b,c,d in zip(kwargs['a2'],kwargs['b2'],kwargs['c2'],kwargs['d2']):
    resp2.append((a,b,c,d))

  same=[]
  diff=[]
  for x,y,m1,m2 in zip(resp1,resp2,kwargs['mask1'],kwargs['mask2']):
    if np.argmax(x)==np.argmax(y):
      same.append(sorted(x)[-1]-sorted(x)[-2])
      #same.append(entropy_(x))
    else:
      diff.append(sorted(x)[-1]-sorted(x)[-2])
      #diff.append(entropy_(x))

  plot_hist2(same=same,diff=diff,name_=kwargs['label'],fname='/code/tensorrt_llm/lm-evaluation-harness/hostvm/images/flips_analysis/others1/'+kwargs['label']+'.png')



def analyzeflips2(**kwargs):
  resp2=[]
  resp1=[]
  for a,b,c,d in zip(kwargs['a1'],kwargs['b1'],kwargs['c1'],kwargs['d1']):  
    resp1.append((a,b,c,d))
  for a,b,c,d in zip(kwargs['a2'],kwargs['b2'],kwargs['c2'],kwargs['d2']):
    resp2.append((a,b,c,d))

  it =0
  r2w=[]
  w2wdiff=[]
  w2wsame=[]
  w2r=[]
  c2c=[]
  for x,y,m1,m2 in zip(resp1,resp2,kwargs['mask1'],kwargs['mask2']):
    it = it + 1
    if np.argmax(x)!=np.argmax(y):
      #change
      #right to wrong
      if m1==1 and m2==0:
        r2w.append(sorted(x)[-1]-sorted(x)[-2])
      #wrong to right
      if m1==0 and m2==1:
        w2r.append(sorted(x)[-1]-sorted(x)[-2])
      if m1==0 and m2==0:
        w2wdiff.append(sorted(x)[-1]-sorted(x)[-2])
    else:
      #same
      if m1==1 and m2==1:
        c2c.append(sorted(x)[-1]-sorted(x)[-2])

      if m1==0 and m2==0:
        w2wsame.append(sorted(x)[-1]-sorted(x)[-2])

  plot_hist(c2c=c2c,w2wdiff=w2wdiff,w2wsame=w2wsame, c2w=r2w, w2c=w2r, name_=kwargs['label'],fname='/code/tensorrt_llm/lm-evaluation-harness/hostvm/images/flips_analysis/'+kwargs['label']+'.png')
  
def analyzeflips(**kwargs):
  resp2=[]
  resp1=[]
  for a,b,c,d in zip(kwargs['a1'],kwargs['b1'],kwargs['c1'],kwargs['d1']):  
    resp1.append((a,b,c,d))
  for a,b,c,d in zip(kwargs['a2'],kwargs['b2'],kwargs['c2'],kwargs['d2']):
    resp2.append((a,b,c,d))

  it =0
  for x,y,m1,m2 in zip(resp1,resp2,kwargs['mask1'],kwargs['mask2']):
    it = it + 1
    if np.argmax(x)!=np.argmax(y):
      #change
      #right to wrong
      if m1==1 and m2==0:
        plot('/code/tensorrt_llm/lm-evaluation-harness/hostvm/images/flips_analysis/correct_to_wrong/'+str(it)+'.png',('A','B','C','D'),x[0],x[1],x[2],x[3],y[0],y[1],y[2],y[3],'right to wrong')
      #wrong to right
      if m1==0 and m2==1:
        plot('/code/tensorrt_llm/lm-evaluation-harness/hostvm/images/flips_analysis/wrong_to_correct/'+str(it)+'.png',('A','B','C','D'),x[0],x[1],x[2],x[3],y[0],y[1],y[2],y[3],'wrong to right')
      if m1==0 and m2==0:
        plot('/code/tensorrt_llm/lm-evaluation-harness/hostvm/images/flips_analysis/wrong_to_wrong_diff/'+str(it)+'.png',('A','B','C','D'),x[0],x[1],x[2],x[3],y[0],y[1],y[2],y[3],'wrong to wrong diff')
    else:
      #same
      if m1==1 and m2==1:
        plot('/code/tensorrt_llm/lm-evaluation-harness/hostvm/images/flips_analysis/correct_to_correct/'+str(it)+'.png',('A','B','C','D'),x[0],x[1],x[2],x[3],y[0],y[1],y[2],y[3],'correct to correct')

      if m1==0 and m2==0:
        plot('/code/tensorrt_llm/lm-evaluation-harness/hostvm/images/flips_analysis/wrong_to_wrong_same/'+str(it)+'.png',('A','B','C','D'),x[0],x[1],x[2],x[3],y[0],y[1],y[2],y[3],'wrong to wrong same')


def number_of_changes_to_level(**kwargs):
  resp2=[]
  resp1=[]
  
  for a,b,c,d in zip(kwargs['a1'],kwargs['b1'],kwargs['c1'],kwargs['d1']):  
    resp1.append((a,b,c,d))
  for a,b,c,d in zip(kwargs['a2'],kwargs['b2'],kwargs['c2'],kwargs['d2']):
    resp2.append((a,b,c,d))

  total = len(resp1)
  number_same = 0
  number_changed_to_second = 0
  number_changed_to_third = 0
  number_changed_to_fourth = 0

  number_correct = 0
  number_correct_second_best = 0
  number_correct_third_best = 0
  number_correct_fourth_best = 0

  correct_to_correct = 0
  correct_to_wrong = 0
  wrong_to_correct = 0
  wrong_to_wrong_diff = 0
  wrong_to_wrong_same = 0

  number_changed_to_second_given_correct=0
  number_changed_to_third_given_correct=0
  number_changed_to_fourth_given_correct=0
  number_changed_to_second_given_wrong=0 
  number_changed_to_third_given_wrong=0
  number_changed_to_fourth_given_wrong=0

  task_to_file = {'abstract algebra':'hendrycksTest-abstract_algebra.txt',
                  'machine learning':'hendrycksTest-machine_learning.txt',
                  'philosophy':'hendrycksTest-philosophy.txt',
                  'govt politics':'hendrycksTest-high_school_government_and_politics.txt',
  }
  f = open('/code/tensorrt_llm/lm-evaluation-harness/hostvm/correct_answers/'+task_to_file[kwargs['task']],'r')
  correct_answers_char = f.read().split()
  correct_answers_int = [ord(x)-ord('A') for x in correct_answers_char]

  for x,y,correct in zip(resp1,resp2,correct_answers_int):
    if np.argmax(x)==np.argmax(y):
      number_same+=1
    elif np.argsort(x)[-2]==np.argmax(y):
      number_changed_to_second+=1
    elif np.argsort(x)[-3]==np.argmax(y):
      number_changed_to_third+=1
    elif np.argsort(x)[-4]==np.argmax(y):
      number_changed_to_fourth+=1

    if np.argmax(x)==correct:
      number_correct+=1
    elif np.argsort(x)[-2]==correct:
      number_correct_second_best+=1
    elif np.argsort(x)[-3]==correct:
      number_correct_third_best+=1
    elif np.argsort(x)[-4]==correct:
      number_correct_fourth_best+=1

    if np.argmax(x)==correct and np.argmax(y)==correct:
      correct_to_correct+=1
    elif np.argmax(x)==correct and np.argmax(y)!=correct:
      correct_to_wrong+=1
    elif np.argmax(x)!=correct and np.argmax(y)==correct:
      wrong_to_correct+=1
    elif np.argmax(x)!=correct and np.argmax(y)!=correct and np.argmax(x)==np.argmax(y):
      wrong_to_wrong_same+=1
    elif np.argmax(x)!=correct and np.argmax(y)!=correct and np.argmax(x)!=np.argmax(y):
      wrong_to_wrong_diff+=1
    else:
      assert(False)
    
    if np.argmax(x)==correct and np.argsort(x)[-2]==np.argmax(y):
      number_changed_to_second_given_correct+=1
    elif np.argmax(x)==correct and np.argsort(x)[-3]==np.argmax(y):
      number_changed_to_third_given_correct+=1
    elif np.argmax(x)==correct and np.argsort(x)[-4]==np.argmax(y):
      number_changed_to_fourth_given_correct+=1

    if np.argmax(x)!=correct and np.argsort(x)[-2]==np.argmax(y):
      number_changed_to_second_given_wrong+=1
    elif np.argmax(x)!=correct and np.argsort(x)[-3]==np.argmax(y):
      number_changed_to_third_given_wrong+=1
    elif np.argmax(x)!=correct and np.argsort(x)[-4]==np.argmax(y):
      number_changed_to_fourth_given_wrong+=1


  f = open('/code/tensorrt_llm/lm-evaluation-harness/hostvm/images/flips_analysis/others2/'+kwargs['label']+'.txt','w')
  f.write(kwargs['label']+'\n')  
  f.write('total number of questions = '+str(total)+'\n')
  f.write('number of questions same = '+str(number_same)+'\n')
  f.write('number of questions changed to second best = '+str(number_changed_to_second)+'\n')
  f.write('number of questions changed to third best = '+str(number_changed_to_third)+'\n')
  f.write('number of questions changed to fourth best = '+str(number_changed_to_fourth)+'\n')
  assert(abs(number_same+number_changed_to_second+number_changed_to_third+number_changed_to_fourth-total)<5)

  #print('\n')

  f.write('number of questions answered correctly = '+str(number_correct)+'\n')
  f.write('number of questions where correct option is the second best option = '+str(number_correct_second_best)+'\n')
  f.write('number of questions where correct option is the third best option = '+str(number_correct_third_best)+'\n')
  f.write('number of questions where correct option is the fourth best option = '+str(number_correct_fourth_best)+'\n')
  #print(number_correct+number_correct_second_best+number_correct_third_best+number_correct_fourth_best,total)
  #print('\n')

  f.write('number of correct-> wrong = '+str(correct_to_wrong)+'\n')
  f.write('number of wrong->correct = '+str(wrong_to_correct)+'\n')
  f.write('number of wrong->wrong diff = '+str(wrong_to_wrong_diff)+'\n')
  f.write('number of wrong->wrong same = '+str(wrong_to_wrong_same)+'\n')
  f.write('number of correct->correct = '+str(correct_to_correct)+'\n')
  #print('\n\n')
  #print(correct_to_wrong,number_changed_to_second_given_correct+number_changed_to_third_given_correct+number_changed_to_fourth_given_correct)
  assert(abs(number_correct+number_correct_second_best+number_correct_third_best+number_correct_fourth_best-total)<5)

  f.close()
  return total,number_same,number_changed_to_second,number_changed_to_third,number_changed_to_fourth,number_correct,number_correct_second_best,number_correct_third_best,number_correct_fourth_best,correct_to_correct,correct_to_wrong,wrong_to_correct,wrong_to_wrong_diff,wrong_to_wrong_same,number_changed_to_second_given_correct,number_changed_to_third_given_correct,number_changed_to_fourth_given_correct,number_changed_to_second_given_wrong,number_changed_to_third_given_wrong,number_changed_to_fourth_given_wrong

def validating_hypothesis(**kwargs):
  map_config_to_data = kwargs['map_config_to_data']
  #print(map_config_to_data)
  wrong_to_correct_list = []
  correct_to_wrong_list=[]
  for model in POSSIBLE_MODELS:
    for shot in POSSIBLE_SHOT:
      for task in POSSIBLE_TASKS:
        for quantization in ['4bit']:
          a1=map_config_to_data[model+'16bit'+'normal'+shot+task+ARTIFACT_TO_ENGLISH['a_likelihoods']]
          #print(model+'16bit'+'normal'+shot+task+ARTIFACT_TO_ENGLISH['a_likelihoods'])
          b1=map_config_to_data[model+'16bit'+'normal'+shot+task+ARTIFACT_TO_ENGLISH['b_likelihoods']]
          c1=map_config_to_data[model+'16bit'+'normal'+shot+task+ARTIFACT_TO_ENGLISH['c_likelihoods']]
          d1=map_config_to_data[model+'16bit'+'normal'+shot+task+ARTIFACT_TO_ENGLISH['d_likelihoods']]

          a2=map_config_to_data[model+quantization+'normal'+shot+task+ARTIFACT_TO_ENGLISH['a_likelihoods']]
          b2=map_config_to_data[model+quantization+'normal'+shot+task+ARTIFACT_TO_ENGLISH['b_likelihoods']]
          c2=map_config_to_data[model+quantization+'normal'+shot+task+ARTIFACT_TO_ENGLISH['c_likelihoods']]
          d2=map_config_to_data[model+quantization+'normal'+shot+task+ARTIFACT_TO_ENGLISH['d_likelihoods']]

          mask1=map_config_to_data[model+'16bit'+'normal'+shot+task+ARTIFACT_TO_ENGLISH['_mask']]
          mask2=map_config_to_data[model+quantization+'normal'+shot+task+ARTIFACT_TO_ENGLISH['_mask']]

          total,number_same,number_changed_to_second,number_changed_to_third,number_changed_to_fourth,number_correct,number_correct_second_best,number_correct_third_best,number_correct_fourth_best,correct_to_correct,correct_to_wrong,wrong_to_correct,wrong_to_wrong_diff,wrong_to_wrong_same,number_changed_to_second_given_correct,number_changed_to_third_given_correct,number_changed_to_fourth_given_correct,number_changed_to_second_given_wrong, number_changed_to_third_given_wrong, number_changed_to_fourth_given_wrong =number_of_changes_to_level(a1=a1,b1=b1,c1=c1,d1=d1,a2=a2,b2=b2,c2=c2,d2=d2,mask1=mask1,mask2=mask2,label=model+quantization+'normal'+shot+task,task=task)

          #theoretical_wrong_to_correct = (total-number_correct)*(number_changed_to_second*1.0/total)*(number_correct_second_best*1.0/(number_correct_second_best+number_correct_third_best+number_correct_fourth_best))

          theoretical_correct_to_wrong = (number_correct)*(number_changed_to_second*1.0/total)

          theoretical_wrong_to_correct = (number_changed_to_second_given_wrong*1.0)*(number_correct_second_best*1.0/(number_correct_second_best+number_correct_third_best+number_correct_fourth_best)) + (number_changed_to_third_given_wrong*1.0)*(number_correct_third_best*1.0/(number_correct_second_best+number_correct_third_best+number_correct_fourth_best)) + (number_changed_to_fourth_given_wrong*1.0)*(number_correct_fourth_best*1.0/(number_correct_second_best+number_correct_third_best+number_correct_fourth_best))

          #theoretical_correct_to_wrong = (number_changed_to_second_given_correct+number_changed_to_third_given_correct+number_changed_to_fourth_given_correct)

          wrong_to_correct_list.append((theoretical_wrong_to_correct,wrong_to_correct))
          correct_to_wrong_list.append((theoretical_correct_to_wrong,correct_to_wrong))
  
  wrong_to_correct_list = sorted(wrong_to_correct_list,key=lambda x:x[1])
  correct_to_wrong_list = sorted(correct_to_wrong_list,key=lambda x:x[1])
  fig, ax = plt.subplots(2,layout='constrained')
  x = np.arange(len(wrong_to_correct_list))
  ax[0].plot(x,[y[0] for y in wrong_to_correct_list], 'go-', label='Theoretical', linewidth=2)
  ax[0].plot(x,[y[1] for y in wrong_to_correct_list], 'bo-', label='Real', linewidth=2)
  ax[0].set_title('Wrong to Correct')

  ax[1].plot(x,[y[0] for y in correct_to_wrong_list], 'go-', label='Theoretical', linewidth=2)
  ax[1].plot(x,[y[1] for y in correct_to_wrong_list], 'bo-', label='Real', linewidth=2)
  ax[1].set_title('Correct to Wrong')
  ax[0].legend()
  ax[1].legend()
  fig.suptitle('Validation for AWQ')
  fig.savefig('/code/tensorrt_llm/lm-evaluation-harness/hostvm/images/flips_analysis/validation_for_AWQ_new.png')   # save the figure to file
  plt.close(fig)   







def number_of_answers_changed(**kwargs):
  resp2=[]
  resp1=[]
  print(kwargs['a1'],kwargs['b1'],kwargs['c1'],kwargs['d1'])
  print(kwargs['a2'],kwargs['b2'],kwargs['c2'],kwargs['d2'])
  print()
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
    if fnmatch.fnmatch(file,'*chatawq*') and fnmatch.fnmatch(file,'*.txt') and fnmatch.fnmatch(file,'rootazurestorage*'):
      #print(file)
      model,wrong_corr_norm,shot,task,_artifacts=getinfo(file)
      #print(model,shot,wrong_corr_norm,task,_artifacts)
      if agg_artifact=='accuracy':
        if file.find('mask')!=-1:
          arr=np.loadtxt(DIR+file)
          map_config_to_data[model+'4bit'+wrong_corr_norm+shot+task+'accuracy']=np.mean(arr)*100.0
      elif agg_artifact=='clearance':
        if file.find('clearance')!=-1:
          arr=np.loadtxt(DIR+file)
          map_config_to_data[model+'4bit'+wrong_corr_norm+shot+task+'clearance']=np.mean(arr)*100.0
      elif agg_artifact=='number of answers changed' or agg_artifact=='number of answers same' or file.find('_mask')!=-1 or agg_artifact=='#answers changed from correct' or agg_artifact=='#answers changed from wrong' or agg_artifact=='maximum variation':
        if file.find('a_likelihoods')!=-1 or file.find('b_likelihoods')!=-1 or file.find('c_likelihoods')!=-1 or file.find('d_likelihoods')!=-1 or file.find('_mask')!=-1:
          arr=np.loadtxt(DIR+file)
          map_config_to_data[model+'4bit'+wrong_corr_norm+shot+task+_artifacts+agg_artifact]=arr
    
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


  #print(map_config_to_data)
  return map_config_to_data


def get_map_config_to_data_new(DIR):
  map_config_to_data=collections.defaultdict(lambda:np.array([]))
  #map_config_to_data={}
  for file in sorted(os.listdir(DIR)):
    if fnmatch.fnmatch(file,'*chatawq*') and fnmatch.fnmatch(file,'*.txt') and fnmatch.fnmatch(file,'rootazurestorage*'):
      model,wrong_corr_norm,shot,task,_artifacts=getinfo(file)
      #print(model,shot,wrong_corr_norm,task,_artifacts)
      arr=np.loadtxt(DIR+file)
      map_config_to_data[model+'4bit'+wrong_corr_norm+shot+task+_artifacts]=arr

    if fnmatch.fnmatch(file,'*.txt') and fnmatch.fnmatch(file,'pretrained*'):
      model,wrong_corr_norm,shot,task,_artifacts=getinfo(file)
      arr=np.loadtxt(DIR+file)
      map_config_to_data[model+'16bit'+wrong_corr_norm+shot+task+_artifacts]=arr
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
    mask2=map_config_to_data[model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['_mask']+agg_artifact]


    if a1.size==0 or a2.size==0:
      print(model+quantization+wrong_corr_norm+shot+task+ARTIFACT_TO_ENGLISH['a_likelihoods']+agg_artifact)
      exit()
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
  
    
    



def print_headings_qtz(worksheet):
  col=5
  for task in POSSIBLE_TASKS:
    worksheet.write_string(0,col+2,task)
    worksheet.write_string(1,col+2,'QUANTIZATION')
    worksheet.write_string(2,col,'16bit')
    worksheet.write_string(2,col+2,'4bit')
    col=col+4
    col=col+1



def dump_qtz_variation(agg_artifact,DIR,worksheet):
  print_headings_qtz(worksheet)

  worksheet.write_string(3,0,'Model')
  worksheet.write_string(3,1,'#shot')
  map_config_to_data=get_map_config_to_data(agg_artifact,DIR)
  #print(map_config_to_data.keys())
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

def print_artifacts():
  for agg_artifact in (POSSIBLE_AGG_ARTIFACTS):
    #print(agg_artifact)
    worksheet_qtz = workbook_qtz.add_worksheet(agg_artifact)

    dump_qtz_variation(agg_artifact,'./hostvm/logs/',worksheet_qtz)

    workbook_qtz.close()

#print_artifacts()

def validate():
  map_config_to_data=get_map_config_to_data_new('./hostvm/logs/')
  #print(map_config_to_data)
  validating_hypothesis(map_config_to_data=map_config_to_data)

validate()