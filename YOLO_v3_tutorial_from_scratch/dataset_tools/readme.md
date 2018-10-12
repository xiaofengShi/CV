# Dataset_tools
Create the annotation to train model for different dataset format.
- coco_annotation.py  
    - create dataset for the uniform dataset MScoco
- voc_annotation.py
    - create dataset for the uniform dataset VOC

## annotation output 
- One row for one image;
- Row format: image_file_path box1 box2 ... boxN;
- Box format: x_min,y_min,x_max,y_max,class_id (no space).
For VOC dataset, try python voc_annotation.py
