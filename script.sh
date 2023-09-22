#models=(meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-13b-chat-hf)
fewshot=(0 1 3 5)

for model in "${models[@]}"; do
	for shot in "${fewshot[@]}"; do
    echo 'python main.py --model hf-causal-experimental --model_args pretrained=$model --num_fewshot $shot --tasks hendrycksTest-philosophy --no_cache --batch_size 1 > hostvm/logs/$model-$shot-shot.txt '
    python main.py --model hf-causal-experimental --model_args pretrained=$model --num_fewshot $shot --tasks hendrycksTest-philosophy --no_cache --batch_size 1 > hostvm/logs/$model-$shot-shot-16bit.txt 

    echo 'python main.py --model hf-causal-experimental --model_args pretrained=$model,load_in_8bit=True --num_fewshot $shot --tasks hendrycksTest-philosophy --no_cache --batch_size 32 > hostvm/logs/$model-$shot-shot.txt '
    python main.py --model hf-causal-experimental --model_args pretrained=$model,load_in_8bit=True --num_fewshot $shot --tasks hendrycksTest-philosophy --no_cache --batch_size 32 > hostvm/logs/$model-$shot-shot-8bit.txt 

    echo 'python main.py --model hf-causal-experimental --model_args pretrained=$model,load_in_4bit=True --num_fewshot $shot --tasks hendrycksTest-philosophy --no_cache --batch_size 32 > hostvm/logs/$model-$shot-shot.txt '
    python main.py --model hf-causal-experimental --model_args pretrained=$model,load_in_4bit=True --num_fewshot $shot --tasks hendrycksTest-philosophy --no_cache --batch_size 32 > hostvm/logs/$model-$shot-shot-4bit.txt 
    done
done 