import subprocess
import sys
import tqdm
import time
import sys
fh = open('output.txt', 'w')
original_stderr = sys.stderr
sys.stderr = fh

#models=['/root/azure-storage/huggingface/Llama-2-13b-chat-hf-gptq-4bit', '/root/azure-storage/huggingface/Llama-2-13b-chat-hf-gptq-8bit', '/root/azure-storage/huggingface/Llama-2-70b-chat-hf-gptq-4bit', '/root/azure-storage/huggingface/Llama-2-70b-chat-hf-gptq-8bit', '/root/azure-storage/huggingface/Llama-2-70b-chat-hf-llmint8-4bit', '/root/azure-storage/huggingface/Llama-2-7b-chat-hf-llmint8-4bit', '/root/azure-storage/huggingface/Llama-2-7b-chat-hf-gptq-4bit', '/root/azure-storage/huggingface/Llama-2-7b-chat-hf-gptq-8bit', '/root/azure-storage/huggingface/yi-34b-chat', '/root/azure-storage/huggingface/yi-34b-gptq-4bit', '/root/azure-storage/huggingface/yi-34b-gptq-8bit', '/root/azure-storage/huggingface/yi-6b-chat', '/root/azure-storage/huggingface/yi-6b-chat-gptq-4bit', '/root/azure-storage/huggingface/yi-6b-gptq-8bit', '/root/azure-storage/llama-weights-abhinav/Llama-2-13b-chat-hf', '/root/azure-storage/llama-weights-abhinav/Llama-2-70b-chat-hf', '/root/azure-storage/llama-weights-abhinav/Llama-2-7b-chat-hf', '/root/azure-storage/huggingface/yi34b-llmint8-4bit', '/root/azure-storage/huggingface/yi34b-llmint8-8bit', '/root/azure-storage/huggingface/yi6b-llmint8-4bit', '/root/azure-storage/huggingface/yi6b-llmint8-8bit', '/root/azure-storage/huggingface/Llama-2-13b-chat-hf-llmint8-8bit', '/root/azure-storage/huggingface/Llama-2-13b-chat-hf-llmint8-4bit', '/root/azure-storage/huggingface/Llama-2-7b-chat-hf-llmint8-8bit', '/root/azure-storage/huggingface/Llama-2-70b-chat-hf-llmint8-8bit' ]

mmlu_tasks=['hendrycksTest-abstract_algebra', 'hendrycksTest-anatomy', 'hendrycksTest-astronomy', 'hendrycksTest-business_ethics', 'hendrycksTest-clinical_knowledge', 'hendrycksTest-college_biology', 'hendrycksTest-college_chemistry', 'hendrycksTest-college_computer_science', 'hendrycksTest-college_mathematics', 'hendrycksTest-college_medicine', 'hendrycksTest-college_physics', 'hendrycksTest-computer_security', 'hendrycksTest-conceptual_physics', 'hendrycksTest-econometrics', 'hendrycksTest-electrical_engineering', 'hendrycksTest-elementary_mathematics', 'hendrycksTest-formal_logic', 'hendrycksTest-global_facts', 'hendrycksTest-high_school_biology', 'hendrycksTest-high_school_chemistry', 'hendrycksTest-high_school_computer_science', 'hendrycksTest-high_school_european_history', 'hendrycksTest-high_school_geography', 'hendrycksTest-high_school_government_and_politics', 'hendrycksTest-high_school_macroeconomics', 'hendrycksTest-high_school_mathematics', 'hendrycksTest-high_school_microeconomics', 'hendrycksTest-high_school_physics', 'hendrycksTest-high_school_psychology',	'hendrycksTest-high_school_statistics', 'hendrycksTest-high_school_us_history', 'hendrycksTest-high_school_world_history', 'hendrycksTest-human_aging', 'hendrycksTest-human_sexuality', 'hendrycksTest-international_law', 'hendrycksTest-jurisprudence', 'hendrycksTest-logical_fallacies', 'hendrycksTest-machine_learning', 'hendrycksTest-management', 'hendrycksTest-marketing', 'hendrycksTest-medical_genetics', 'hendrycksTest-miscellaneous', 'hendrycksTest-moral_disputes', 'hendrycksTest-moral_scenarios', 'hendrycksTest-nutrition', 'hendrycksTest-philosophy', 'hendrycksTest-prehistory', 'hendrycksTest-professional_accounting', 'hendrycksTest-professional_law', 'hendrycksTest-professional_medicine', 'hendrycksTest-professional_psychology', 'hendrycksTest-public_relations', 'hendrycksTest-security_studies', 'hendrycksTest-sociology', 'hendrycksTest-us_foreign_policy', 'hendrycksTest-virology', 'hendrycksTest-world_religions']

'''
tasks = ['bigbench_causal_judgement', 'anli_r1', 'arithmetic_1dc', 'blimp_adjunct_island', 'cola', 'copa', 'coqa', 'gsm8k', 'headqa', 'hellaswag', 'lambada_openai', 'logiqa',  'math_geometry', 'mnli pile_arxiv triviaqa', 'truthfulqa_gen', 'squad2', 
'scrolls_summscreenfd', 'winogrande boolq']
commands=[]
for task in tasks:
    commands.append(f"python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/opt-125m,use_accelerate=True --tasks {task} --batch_size 64 --no_cache --write_out --output_base_path ./hostvm/jsonlogs/")



# Run each command concurrently
processes = []
for cmd in commands:
    processes.append(subprocess.Popen(cmd, shell=True))

# Wait for all processes to finish
for process in processes:
    process.wait()

print("All commands completed.")
'''

OUTPUT_JSONLOGS  = '/root/azure-storage/eval_outputs/jsonlogs/'
OUTPUT_TEXTLOGS  = '/root/azure-storage/eval_outputs/hostvm/logs/'

OUTPUT_MMLU_JSONLOGS  = '/root/azure-storage/eval_outputs/mmlu/jsonlogs/5-shot/'
OUTPUT_MMLU_TEXTLOGS  = '/root/azure-storage/eval_outputs/mmlu/hostvm/logs/'

OUTPUT_MMLU_JSONLOGS_PRETRAINED  = '/root/azure-storage/eval_outputs/mmlu/jsonlogs/pretrained/'


OUTPUT_MMLU_TEXTLOGS_ALTBS= '/root/azure-storage/eval_outputs/mmlu/hostvm/alt_batchsize/'
OUTPUT_MMLU_JSONLOGS_ATLBS = '/root/azure-storage/eval_outputs/mmlu/jsonlogs/alt_batchsize/'

MMLU_DUMP_PATH='/root/azure-storage/eval_outputs/logprobs/mmlu/'
DUMP_PATH = '/root/azure-storage/eval_outputs/logprobs/'

models_gpu3 = ['/root/azure-storage/huggingface/yi-6b-chat-awq4bit-gs128',
               '/root/azure-storage/huggingface/yi-34b-chat-awq4bit-gs128',
               '/root/azure-storage/huggingface/Llama-2-13b-chat-hf-awq4bit-gs128',
               '/root/azure-storage/huggingface/Llama-2-70b-chat-hf-awq4bit-gs128',
               '/root/azure-storage/huggingface/Llama-2-7b-chat-hf-awq4bit-gs128',
]



models_gpu1 = ['/root/azure-storage/llama-weights-abhinav/Llama-2-7b-chat-hf']


models_gpu2 = ['/root/azure-storage/huggingface/pretrained/Llama-2-7b-hf-awq4bit-gs128',
               '/root/azure-storage/llama-weights-abhinav/pretrained/Llama-2-70b-hf',
               '/root/azure-storage/huggingface/pretrained/Llama-2-70b-hf-awq4bit-gs128',
               '/root/azure-storage/huggingface/pretrained/Llama-2-13b-hf-awq4bit-gs128',
]

models_gpu2 = ['/root/azure-storage/huggingface/yi-6b-chat-gptq-4bit',
               '/root/azure-storage/huggingface/yi-34b-gptq-4bit',
               '/root/azure-storage/huggingface/Llama-2-13b-chat-hf-gptq-4bit',
               '/root/azure-storage/huggingface/Llama-2-70b-chat-hf-gptq-4bit',
               '/root/azure-storage/huggingface/Llama-2-7b-chat-hf-gptq-4bit',
]


#engines_gpu0 = [('/root/azure-storage/quant/engines/trt_llama_13b_chat_newsmoothquant_sq08_1gpu_per_token_channel','/root/azure-storage/llama-weights-abhinav/Llama-2-13b-chat-hf'),('/root/azure-storage/quant/engines/trt_llama_7b_chat_newsmoothquant_sq08_1gpu_per_token_channel','/root/azure-storage/llama-weights-abhinav/Llama-2-7b-chat-hf')]

engines_gpu0 = [('/root/azure-storage/quant/engines/trt_yi_34b_chat_smoothquant_sq08_1gpu_per_token_channel','/root/azure-storage/huggingface/yi-34b-chat'),
                ('/root/azure-storage/quant/engines/trt_yi_6b_chat_smoothquant_sq08_1gpu_per_token_channel','/root/azure-storage/huggingface/yi-6b-chat'),
                ('/root/azure-storage/quant/engines/trt_llama_13b_chat_newsmoothquant_sq08_1gpu_per_token_channel','/root/azure-storage/llama-weights-abhinav/Llama-2-13b-chat-hf'),
                ('/root/azure-storage/quant/engines/trt_llama_70b_chat_newsmoothquant_sq08_1gpu_per_token_channel','/root/azure-storage/llama-weights-abhinav/Llama-2-70b-chat-hf'),
                ('/root/azure-storage/quant/engines/trt_llama_7b_chat_newsmoothquant_sq08_1gpu_per_token_channel','/root/azure-storage/llama-weights-abhinav/Llama-2-7b-chat-hf')

]


engines_gpu1 = [('/root/azure-storage/quant/engines/trt_yi_6b_chat_smoothquant_sq05_1gpu_per_token_channel','/root/azure-storage/huggingface/yi-6b-chat')
]
engines_gpu2 = [('/root/azure-storage/quant/engines/trt_yi_34b_chat_smoothquant_sq05_1gpu_per_token_channel','/root/azure-storage/huggingface/yi-34b-chat')]
engines_gpu3 = [ ('/root/azure-storage/quant/engines/pretrained/trt_Llama-2-13b-hf_smoothquant_sq05_1gpu_per_token_channel','/root/azure-storage/llama-weights-abhinav/pretrained/Llama-2-13b-hf'),('/root/azure-storage/quant/engines/pretrained/trt_Llama-2-7b-hf_smoothquant_sq05_1gpu_per_token_channel','/root/azure-storage/llama-weights-abhinav/pretrained/Llama-2-7b-hf'),
('/root/azure-storage/quant/engines/pretrained/trt_Llama-2-70b-hf_smoothquant_sq05_1gpu_per_token_channel','/root/azure-storage/llama-weights-abhinav/pretrained/Llama-2-70b-hf'),
]

models_gpuall = ['/root/azure-storage/llama-weights-abhinav/Llama-2-70b-chat-hf']

tasks = ['hellaswag', 'piqa', 'winogrande','arc_easy', 'arc_challenge','lambada_standard']
commands =[]
'''
for model in models_gpu2:
    for task in tasks:
        if task == 'piqa' or task=='winogrande':
            commands.append(f"CUDA_VISIBLE_DEVICES=2 python main.py --model hf-causal-experimental --model_args pretrained={model},use_accelerate=True --tasks {task} --batch_size 1 --no_cache --write_out --output_base_path {OUTPUT_JSONLOGS} --artifacts_2options --logdir {OUTPUT_TEXTLOGS}")
        if task == 'arc_easy' or task=='arc_challenge' or task=='lambada_standard':
            commands.append(f"CUDA_VISIBLE_DEVICES=2 python main.py --model hf-causal-experimental --model_args pretrained={model},use_accelerate=True --tasks {task} --batch_size 1 --no_cache --write_out --output_base_path {OUTPUT_JSONLOGS} ")
        if task == 'hellaswag':
            commands.append(f"CUDA_VISIBLE_DEVICES=2 python main.py --model hf-causal-experimental --model_args pretrained={model},use_accelerate=True --tasks {task} --batch_size 1 --no_cache --write_out --output_base_path {OUTPUT_JSONLOGS} --artifacts --logdir {OUTPUT_TEXTLOGS}")
'''
#for trt engines
'''
for engine,tokenizer in engines_gpu1:
    cuda_id=1
    for task in tasks:
        if task == 'piqa' or task=='winogrande':
            commands.append(f"CUDA_VISIBLE_DEVICES={cuda_id} python main.py --engine_dir {engine} --tokenizer_dir {tokenizer}   --tasks {task} --batch_size 1 --no_cache --write_out --output_base_path {OUTPUT_JSONLOGS} --artifacts_2options --logdir {OUTPUT_TEXTLOGS}")
        if task == 'arc_easy' or task=='arc_challenge' or task=='lambada_standard':
            commands.append(f"CUDA_VISIBLE_DEVICES={cuda_id} python main.py --engine_dir {engine} --tokenizer_dir {tokenizer}  --tasks {task} --batch_size 1 --no_cache --write_out --output_base_path {OUTPUT_JSONLOGS} ")
        if task == 'hellaswag':
            commands.append(f"CUDA_VISIBLE_DEVICES={cuda_id} python main.py --engine_dir {engine} --tokenizer_dir {tokenizer}  --tasks {task} --batch_size 1 --no_cache --write_out --output_base_path {OUTPUT_JSONLOGS} --artifacts --logdir {OUTPUT_TEXTLOGS}")
'''

'''
for engine,tokenizer in engines_gpu0:
    cuda_id=0
    for task in ['arc_easy','arc_challenge']:
        commands.append(f"CUDA_VISIBLE_DEVICES={cuda_id} python main.py --engine_dir {engine} --tokenizer_dir {tokenizer}  --tasks {task} --batch_size 1 --num_fewshot 0 --no_cache --logprob_dump_path {DUMP_PATH}")

        
for engine,tokenizer in engines_gpu0:
    cuda_id=0
    for task in mmlu_tasks:
        commands.append(f"CUDA_VISIBLE_DEVICES={cuda_id} python main.py --engine_dir {engine} --tokenizer_dir {tokenizer}  --tasks {task} --batch_size 1 --num_fewshot 5 --no_cache --logprob_dump_path {MMLU_DUMP_PATH}")
'''



for model in models_gpu2:
    cuda_id=2
    for task in ['arc_easy','arc_challenge']:
        commands.append(f"CUDA_VISIBLE_DEVICES={cuda_id} python main.py --model hf-causal-experimental --model_args pretrained={model},use_accelerate=True  --tasks {task} --batch_size 1 --num_fewshot 0 --no_cache --logprob_dump_path {DUMP_PATH}")

for model in models_gpu2:
    cuda_id=2
    for task in mmlu_tasks:
        commands.append(f"CUDA_VISIBLE_DEVICES={cuda_id} python main.py --model hf-causal-experimental --model_args pretrained={model},use_accelerate=True  --tasks {task} --batch_size 1 --num_fewshot 5 --no_cache --logprob_dump_path {MMLU_DUMP_PATH}")

'''
for model in models_gpu0:
    cuda_id='0,1'
    for task in mmlu_tasks:
        commands.append(f"CUDA_VISIBLE_DEVICES={cuda_id} python main.py --model hf-causal-experimental --model_args pretrained={model},use_accelerate=True  --tasks {task} --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")
'''

'''
commands.append(f"CUDA_VISIBLE_DEVICES=0 python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/yi-34b-gptq-8bit,use_accelerate=True  --tasks hendrycksTest-high_school_european_history --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")

commands.append(f"CUDA_VISIBLE_DEVICES=0 python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/yi34b-llmint8-8bit,use_accelerate=True  --tasks hendrycksTest-college_medicine --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")

commands.append(f"CUDA_VISIBLE_DEVICES=0 python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/yi34b-llmint8-8bit,use_accelerate=True  --tasks hendrycksTest-high_school_european_history --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")

commands.append(f"CUDA_VISIBLE_DEVICES=0 python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/yi34b-llmint8-8bit,use_accelerate=True  --tasks hendrycksTest-high_school_us_history --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")

commands.append(f"CUDA_VISIBLE_DEVICES=0 python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/yi34b-llmint8-8bit,use_accelerate=True  --tasks hendrycksTest-high_school_world_history --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")

commands.append(f"CUDA_VISIBLE_DEVICES=0 python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/yi34b-llmint8-8bit,use_accelerate=True  --tasks hendrycksTest-professional_law --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")

commands.append(f"CUDA_VISIBLE_DEVICES=0 python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/yi34b-llmint8-8bit,use_accelerate=True  --tasks hendrycksTest-professional_medicine --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")
'''
'''
commands.append(f"CUDA_VISIBLE_DEVICES=1 python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/Llama-2-70b-chat-hf-awq4bit-gs128,use_accelerate=True  --tasks hendrycksTest-high_school_european_history --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")
'''
'''
commands.append(f"CUDA_VISIBLE_DEVICES=1 python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/Llama-2-70b-chat-hf-gptq-4bit,use_accelerate=True  --tasks hendrycksTest-high_school_european_history --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")

commands.append(f"CUDA_VISIBLE_DEVICES=1 python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/Llama-2-70b-chat-hf-llmint8-4bit,use_accelerate=True  --tasks hendrycksTest-high_school_european_history --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")

commands.append(f"CUDA_VISIBLE_DEVICES=1 python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/Llama-2-70b-chat-hf-llmint8-4bit,use_accelerate=True  --tasks hendrycksTest-high_school_us_history --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")

commands.append(f"CUDA_VISIBLE_DEVICES=1 python main.py --model hf-causal-experimental --model_args pretrained=/root/azure-storage/huggingface/Llama-2-70b-chat-hf-llmint8-4bit,use_accelerate=True  --tasks hendrycksTest-professional_law --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS} --artifacts --logdir {OUTPUT_MMLU_TEXTLOGS}")
'''

'''
for engine,tokenizer in engines_gpu3:
    cuda_id=3
    for task in mmlu_tasks:
        commands.append(f"CUDA_VISIBLE_DEVICES={cuda_id} python main.py --engine_dir {engine} --tokenizer_dir {tokenizer}  --tasks {task} --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS_PRETRAINED} ")
'''
'''
for model in models_gpu2:
    cuda_id='1,2'
    for task in mmlu_tasks:
        commands.append(f"CUDA_VISIBLE_DEVICES={cuda_id} python main.py --model hf-causal-experimental --model_args pretrained={model},use_accelerate=True  --tasks {task} --batch_size 1 --num_fewshot 5 --no_cache --write_out --output_base_path {OUTPUT_MMLU_JSONLOGS_PRETRAINED} ")
'''

concat_commands = "; ".join(commands)
print(concat_commands)
process = subprocess.Popen(concat_commands, shell=True)
process.wait()