################## GPT2  ###############
#evaluate gpt2
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_eval_batch_size 8 \
    --do_eval \
    --output_dir /home/vmagent/app/data/LLM/gpt2/evaluate \
    --overwrite_output_dir \
    --report_to none

#finetune gpt2
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /home/vmagent/app/data/LLM/gpt2/finetune \
    --overwrite_output_dir \
    --report_to none

#finetune gpt2 with no trainer script
python run_clm_no_trainer.py \
    --model_name_or_path /home/vmagent/app/data/LLM/gpt2/data \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --output_dir /home/vmagent/app/data/LLM/gpt2/finetune-no-trainer\
    --report_to none

#evaluate gpt2 with no trainer script
python run_clm_no_trainer.py \
    --model_name_or_path /home/vmagent/app/data/LLM/gpt2/finetune \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_eval_batch_size 8 \
    --output_dir /home/vmagent/app/data/LLM/gpt2/test \
    --report_to none \
    --eval

######################### GPT2 with DeNas################
#finetune gpt2 with DeNas
python run_clm_no_trainer.py \
    --model_name_or_path /home/vmagent/app/data/LLM/gpt2/data \
    --is_denas \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --output_dir /home/vmagent/app/data/LLM/gpt2/gpt2-denas/finetune\
    --report_to none


######################### GPT2 with DeNas and KD ################
#finetune gpt2 with no trainer with DeNas and KD
python run_clm_no_trainer.py \
    --model_name_or_path /home/vmagent/app/data/LLM/gpt2/data \
    --teacher_model_name_or_path /home/vmagent/app/data/LLM/gpt2/finetune \
    --is_denas \
    --is_transferrable \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --output_dir /home/vmagent/app/data/LLM/gpt2/gpt2-denas/finetune-kd \
    --report_to none

########################GPT-XL################################
#finetune gpt2-xl
python run_clm.py \
    --model_name_or_path /home/vmagent/app/data/LLM/gpt2-xl/gpt2-xl \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir /home/vmagent/app/data/LLM/gpt2-xl/finetune \
    --overwrite_output_dir \
    --report_to none

########################GPT-J################################
#evaluate gpt-j-6b
python run_clm.py \
    --model_name_or_path EleutherAI/gpt-j-6B \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_eval_batch_size 8 \
    --do_eval \
    --output_dir /home/vmagent/app/data/LLM/gpt-j-6B/evaluate \
    --overwrite_output_dir \
    --report_to none

#finetune gpt-j-6b
python run_clm.py \
    --model_name_or_path EleutherAI/gpt-j-6B \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /home/vmagent/app/data/LLM/gpt-j-6B/finetune \
    --overwrite_output_dir \
    --report_to none