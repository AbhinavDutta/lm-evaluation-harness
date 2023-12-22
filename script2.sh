#models=(meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-13b-chat-hf)
engines=(/root/azure-storage/quant/smoothquant/smooth_llama_7b-chat_sq0.8_1gpu_per_token_channel /root/azure-storage/quant/smoothquant/smooth_llama_13b-chat_sq0.8_1gpu_per_token_channel  /root/azure-storage/quant/smoothquant/smooth_llama_70b-chat_sq0.8_1gpu_per_token_channel)
tokenizer=/root/azure-storage/llama-weights-abhinav/Llama-2-70b-chat-hf
fewshot=(0 1 3 5)
tasks=(hendrycksTest-abstract_algebra  hendrycksTest-machine_learning hendrycksTest-philosophy hendrycksTest-high_school_government_and_politics)
for engine in "${engines[@]}"; do
	for shot in "${fewshot[@]}"; do
        for task in "${tasks[@]}"; do
            engine_name=$(echo $engine | tr '/' '_')
            echo $engine_name,$shot,$task
            python main.py  --num_fewshot $shot --tasks $task --no_cache --artifacts --engine_dir $engine --tokenizer_dir $tokenizer  > hostvm/logs/$engine_name-$shot-shot-$task.txt || echo " FAILED python main.py  --num_fewshot $shot --tasks $task --no_cache --artifacts  > hostvm/logs/$engine_name-$shot-shot-$task.txt  "
        done
    done
done 