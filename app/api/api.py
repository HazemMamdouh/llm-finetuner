import torch
from app.modules.data_model import InputData, OutputData
import jsonlines
import subprocess
import json
from typing import List
from fastapi.encoders import jsonable_encoder


def check_available_GPUs():
    """
    This function is to select the non utilized gpu
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        for gpu_id in range(torch.cuda.device_count()):
            utilization = round(torch.cuda.memory_allocated(gpu_id)/1024**3,1)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory/1e9
            percentage = utilization / total_memory
            if percentage <= 0.1:
                return gpu_id
    return -1
        
    

def finetune_model(training_data: InputData, model_id: str):
    """
    This function determines the gpu that could be utilized and start the process of training as a background task
    """
    gpu_id = check_available_GPUs()
    response = dict()
    if gpu_id < 0 :
        response["gpu_id"] = -1
        response["message"] = "Training could not take place since there is no gpu vacany"
        response["gpu_type"] = ""
        response["model_id"] = model_id
    else:
        response["gpu_id"] = gpu_id
        response["message"] = "Model Finetunning is starting ..."
        response["gpu_type"] = torch.cuda.get_device_name(gpu_id)
        response["model_id"] = model_id
        with open("data/alpaca_data_en_52k.json", "w") as outfile:
            outfile.write("[")
            for json_object in training_data:
                json.dump(jsonable_encoder(json_object), outfile,indent = 4)
                outfile.write(",\n")
            outfile.write("]")
        
        process = subprocess.Popen(["CUDA_VISIBLE_DEVICES="+str(gpu_id), "python","src/train_bash.py --model_name_or_path 'openlm-research/open_llama_3b_v2' --dataset alpaca_en \
        --template default \
        --stage sft \
        --do_train \
        --finetuning_type lora \
        --lora_target q_proj,v_proj \
        --output_dir "+model_id+" \
        --overwrite_cache \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --save_steps 1000 \
        --learning_rate 5e-5 \
        --num_train_epochs 3.0 \
        --plot_loss \
        --fp16"])
        
    return response

