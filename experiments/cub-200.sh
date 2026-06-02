# bash experiments/cub-200.sh
# experiment settings
DATASET=cub-200

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='0'
CONFIG=configs/cub-200_prompt_scope.yaml
REPEAT=5
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR



# SCOPE
#
# prompt parameter args:
#    arg 1 = prompt component pool size, default equal to task number
#    arg 2 = prompt length, default 8
#    arg 3 = temperature

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name SCOPE \
    --prompt_param 10 8 5 --ca_batch_size_ratio 2 --seeds 0 12 24 36 48 \
    --pretrained_weigh sup21k \
    --log_dir ${OUTDIR}/SCOPE
