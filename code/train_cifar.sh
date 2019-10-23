#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

case $HOST in
"alpha")
    PYTHON="/home/elliot/anaconda3/envs/pytorch041/bin/python" # python environment
    TENSORBOARD='/home/elliot/anaconda3/envs/pytorch041/bin/tensorboard'
    ;;
esac

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=noise_resnet20
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_channelwise_1e-5decay

#dataset path
data_path='/home/elliot/data/pytorch/cifar10'
#tensorboard log path
tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info}/tb_log  

############### main function ############################
{
$PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.00001 --momentum 0.9 \
    --adv_eval --epoch_delay 5 \
    --adv_train
} &
############## Tensorboard logging ##########################
{
if [ "$enable_tb_display" = true ]; then 
    sleep 30 
    wait
    $TENSORBOARD --logdir $tb_path  --port=6006
fi
} &
{
if [ "$enable_tb_display" = true ]; then
    sleep 45
    wait
    case $HOST in
    "alpha")
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &
wait