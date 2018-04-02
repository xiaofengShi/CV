#!/bin/bash
# This is the eval script.

DATASET_DIR=/home/sxf/MyProject_Python/TFtest/SSD-Tensorflow/datasets/normal_data_to_tf_records/
#/home/doctorimage/kindlehe/common/dataset/VOCdevkit/
#../../../../common/dataset/VOC2007/VOCtest_06-Nov-2007/VOCdevkit/
EVAL_DIR=../log_files/log_eval/    # Directory where the results are saved to
CHECKPOINT_PATH=/home/sxf/MyProject_Python/TFtest/SSD-Tensorflow/checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt
#../../../../common/models/tfmodlels/SSD/VGG_VOC0712_SSD_300x300_ft_iter_120000/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt


#dataset_name这个参数在代码里面写死了
python3 ../eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=1

#python ../eval_ssd_network.py \
#    --eval_dir=${EVAL_DIR} \        # Directory where the results are saved to
#    --dataset_dir=${DATASET_DIR} \  # The directory where the dataset files are stored
#    --dataset_name=voc2007_test \ # The name of the dataset to load
#    --dataset_split_name=test \     # The name of the train/test split
#    --model_name=ssd_300_vgg \      # The name of the architecture to evaluate
#    --checkpoint_path=${CHECKPOINT_PATH} \  #The directory where the model was written to or an absolute path to a
                                            #checkpoint file
#    --batch_size=1                  # The number of samples in each batch
