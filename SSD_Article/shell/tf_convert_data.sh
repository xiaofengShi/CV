#filename = tf_convert_data.sh
#!/bin/bash
#This is a shell script to convert Pascal VOC datasets(2007 and 2012) into TF-Records only.

#Directory where the original dataset is stored
DATASET_DIR=/home/sxf/MyProject_Python/TFtest/SSD-Tensorflow/datasets/normal_data_set_voc/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/

#Output directory where to store TFRecords files
OUTPUT_DIR=/home/sxf/MyProject_Python/TFtest/SSD-Tensorflow/datasets/normal_data_to_tf_records/

python3 ../tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2007_train \
    --output_dir=${OUTPUT_DIR}
