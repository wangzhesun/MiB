# base learning step
DATA_ROOT=../data
BATCH=1 #24
DATASET=coco #voc
NAME=MiB
TASK=split3 #15-1-split0
STEP=0
LR=0.01
EPOCH=30 #30
METHOD=MiB
NUMSHOT=5
NUMRUN=5 #5


python -m torch.distributed.launch --nproc_per_node=1 run.py --data_root ${DATA_ROOT}  \
       --batch_size ${BATCH} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step ${STEP} \
       --lr ${LR} --epochs ${EPOCH} --method ${METHOD}



## continual learning step
#DATA_ROOT=../data
#BATCH=24
#DATASET=voc
#NAME=MiB
#TASK=15-1-split2
#STEP=1
#LR=0.001
#EPOCH=30 #30
#METHOD=MiB
#NUMSHOT=1
#NUMRUN=2 #5
#
#
#python -m torch.distributed.launch --nproc_per_node=1 run.py --data_root ${DATA_ROOT}  \
#       --batch_size ${BATCH} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step ${STEP} \
#       --lr ${LR} --epochs ${EPOCH} --method ${METHOD} --num_shot ${NUMSHOT} --num_runs ${NUMRUN} \
#       --few_shot True --all_step True
#       # comment two entries in the last line if do not want few shot or wamt to run each step separately