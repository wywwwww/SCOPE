# bash experiments/imagenet-r.sh
# experiment settings
DATASET=ImageNet_R

# save directory
OUTDIR_S=outputs/${DATASET}/5-task
OUTDIR=outputs/${DATASET}/10-task
OUTDIR_L=outputs/${DATASET}/20-task

# training settings
GPUID='3'
REPEAT=5
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR_S
mkdir -p $OUTDIR
mkdir -p $OUTDIR_L

# SCOPE
#
# prompt parameter args:
#    arg 1 = prompt component pool size, default equal to task number
#    arg 2 = prompt length, default 8
#    arg 3 = temperature

# --- 5-task ------
python -u run.py --config configs/imnet-r_prompt_short_scope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name SCOPE \
        --prompt_param 10 8 1 --seeds 0 1 2 3 4\
        --log_dir ${OUTDIR_S}/SCOPE
sleep 10

# --- 10-task ------  
python -u run.py --config configs/imnet-r_prompt_scope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name SCOPE \
        --prompt_param 10 8 1 --seeds 0 1 2 3 4\
        --log_dir ${OUTDIR}/SCOPE
sleep 10

# --- 20-task ------
python -u run.py --config configs/imnet-r_prompt_long_scope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name SCOPE \
        --prompt_param 10 8 1 --seeds 0 1 2 3 4\
        --log_dir ${OUTDIR_L}/SCOPE