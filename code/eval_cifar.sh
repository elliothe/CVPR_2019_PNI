#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Remember to configure the conda environment to pytorch 0.4.1
case $HOST in
"alpha")
    PYTHON="/home/elliot/anaconda3/envs/pytorch_041/bin/python" # python environment
    TENSORBOARD='/home/elliot/anaconda3/envs/pytorch_041/bin/tensorboard'
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
label_info=eval


data_path='/home/elliot/data/pytorch/cifar10' #dataset path
tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info}/tb_log  #tensorboard log path

# set the pretrained model path
pretrained_model=./save/2019-04-19/cifar10_noise_resnet20_160_SGD_PNI_W_channelwise/checkpoint.pth.tar
############### Neural network ############################
{
$PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 1 \
    --print_freq 100 --decay 0.0002 --momentum 0.9 \
    --evaluate --resume ${pretrained_model} \
    --adv_eval --epoch_delay 5 
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