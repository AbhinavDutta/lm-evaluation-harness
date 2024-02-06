#models=(meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-13b-chat-hf meta-llama/Llama-2-70b-chat-hf)
: <<'skip'
engines=(/root/azure-storage/quant/smoothquant/smooth_llama_7b-chat_sq0.8_1gpu_per_token_channel /root/azure-storage/quant/smoothquant/smooth_llama_13b-chat_sq0.8_1gpu_per_token_channel  /root/azure-storage/quant/smoothquant/smooth_llama_70b-chat_sq0.8_1gpu_per_token_channel)
engines=(/root/azure-storage/quant/engines/trt_llama_7b_chat_awq_4bit_gs128_1gpu /root/azure-storage/quant/engines/trt_llama_13b_chat_awq_4bit_gs128_1gpu /root/azure-storage/quant/engines/trt_llama_70b_chat_awq_4bit_gs128_1gpu)
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
skip

models=(/root/azure-storage/huggingface/Llama-2-13b-chat-hf-gptq-4bit /root/azure-storage/huggingface/Llama-2-13b-chat-hf-gptq-8bit /root/azure-storage/huggingface/Llama-2-70b-chat-hf-gptq-4bit /root/azure-storage/huggingface/Llama-2-7b-chat-hf-llmint8-4bit)
fewshot=(5)
tasks=(hendrycksTest-abstract_algebra hendrycksTest-anatomy hendrycksTest-astronomy hendrycksTest-business_ethics hendrycksTest-clinical_knowledge hendrycksTest-college_biology hendrycksTest-college_chemistry hendrycksTest-college_computer_science hendrycksTest-college_mathematics hendrycksTest-college_medicine hendrycksTest-college_physics hendrycksTest-computer_security hendrycksTest-conceptual_physics hendrycksTest-econometrics hendrycksTest-electrical_engineering hendrycksTest-elementary_mathematics hendrycksTest-formal_logic hendrycksTest-global_facts hendrycksTest-high_school_biology hendrycksTest-high_school_chemistry hendrycksTest-high_school_computer_science hendrycksTest-high_school_european_history hendrycksTest-high_school_geography hendrycksTest-high_school_government_and_politics hendrycksTest-high_school_macroeconomics hendrycksTest-high_school_mathematics hendrycksTest-high_school_microeconomics hendrycksTest-high_school_physics hendrycksTest-high_school_psychology	hendrycksTest-high_school_statistics hendrycksTest-high_school_us_history hendrycksTest-high_school_world_history hendrycksTest-human_aging hendrycksTest-human_sexuality hendrycksTest-international_law hendrycksTest-jurisprudence hendrycksTest-logical_fallacies hendrycksTest-machine_learning hendrycksTest-management hendrycksTest-marketing hendrycksTest-medical_genetics hendrycksTest-miscellaneous hendrycksTest-moral_disputes hendrycksTest-moral_scenarios hendrycksTest-nutrition hendrycksTest-philosophy hendrycksTest-prehistory hendrycksTest-professional_accounting hendrycksTest-professional_law hendrycksTest-professional_medicine hendrycksTest-professional_psychology hendrycksTest-public_relations hendrycksTest-security_studies hendrycksTest-sociology hendrycksTest-us_foreign_policy hendrycksTest-virology hendrycksTest-world_religions)
batch_sizes=(1)
for model in "${models[@]}"; do
	for shot in "${fewshot[@]}"; do
        for task in "${tasks[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                model_name=$(echo $model | tr '/' '_')
                echo $model_name,$shot,$task,$batch_size

                if [ -e "hostvm/logs/$model_name-$shot-shot-$task-batch_$batch_size.txt" ]; then
                    echo "Task already done"
                else
                    python main.py --model hf-causal-experimental --model_args pretrained=$model,use_accelerate=True --num_fewshot $shot --tasks $task --no_cache --artifacts --batch_size $batch_size > hostvm/logs/$model_name-$shot-shot-$task-batch_$batch_size.txt || echo " FAILED python main.py --model hf-causal-experimental --model_args pretrained=$model,use_accelerate=True --num_fewshot $shot --tasks $task --no_cache --artifacts --batch_size $batch_size  "
                fi
            done
        done
    done
done 
