DATA_ROOT=../data
BATCH=24
DATASET=voc
NAME=MiB
TASK=15-1-split2
STEP=1
LR=0.001
EPOCH=2
METHOD=MiB
NUMSHOT=5
NUMRUN=5


python -m torch.distributed.launch --nproc_per_node=1 run.py --data_root ${DATA_ROOT}  \
       --batch_size ${BATCH} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step ${STEP} \
       --lr ${LR} --epochs ${EPOCH} --method ${METHOD} --num_shot ${NUMSHOT} --num_runs ${NUMRUN} \
       --few_shot True --all_step True