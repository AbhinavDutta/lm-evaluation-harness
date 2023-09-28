models=(meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-13b-chat-hf)
#models=(meta-llama/Llama-2-70b-hf meta-llama/Llama-2-70b-chat-hf mosaicml/mpt-30b-chat mosaicml/mpt-30b)
fewshot=(0 1 3 5)
batch_sizes=(1 5 32)
tasks=(hendrycksTest-abstract_algebra  hendrycksTest-machine_learning hendrycksTest-philosophy hendrycksTest-high_school_government_and_politics)
for model in "${models[@]}"; do
	for shot in "${fewshot[@]}"; do
        for task in "${tasks[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                model_name=$(echo $model | tr '/' '-')
                echo $model_name
                python main.py --model hf-causal-experimental --model_args pretrained=$model --num_fewshot $shot --tasks $task --no_cache --artifacts --batch_size $batch_size > hostvm/logs/$model_name-$shot-shot-16bit-$task-batch_$batch_size.txt || echo " FAILED python main.py --model hf-causal-experimental --model_args pretrained=$model --num_fewshot $shot --tasks $task --no_cache --artifacts --batch_size $batch_size > hostvm/logs/$model_name-$shot-shot-16bit-$task-batch_$batch_size.txt  "
                
                python main.py --model hf-causal-experimental --model_args pretrained=$model,load_in_8bit=True --num_fewshot $shot --tasks $task --no_cache --artifacts --batch_size $batch_size > hostvm/logs/$model_name-$shot-shot-8bit-$task-batch_$batch_size.txt || echo "FAILED python main.py --model hf-causal-experimental --model_args pretrained=$model,load_in_8bit=True --num_fewshot $shot --tasks $task --no_cache --artifacts --batch_size $batch_size > hostvm/logs/$model_name-$shot-shot-8bit-$task-batch_$batch_size.txt "

                python main.py --model hf-causal-experimental --model_args pretrained=$model,load_in_4bit=True --num_fewshot $shot --tasks $task --no_cache --artifacts --batch_size $batch_size > hostvm/logs/$model_name-$shot-shot-4bit-$task-batch_$batch_size.txt || echo "FAILED python main.py --model hf-causal-experimental --model_args pretrained=$model,load_in_4bit=True --num_fewshot $shot --tasks $task --no_cache --artifacts --batch_size $batch_size > hostvm/logs/$model_name-$shot-shot-4bit-$task-batch_$batch_size.txt" 
            
            done
        done
    done
done 