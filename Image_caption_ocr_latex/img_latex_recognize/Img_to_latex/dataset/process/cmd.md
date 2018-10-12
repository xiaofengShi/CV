cmd_orders
==========
# cmd_orders local
### First of all-file encoding
```
 In the first, you should see the encoding of the file 'im2latex_formulas.lst'. eapically process the labels
 1.Open this file in vim and in the 'esc' model
 2.Type the cmd ':set fileencoding' then you can see  the encoding of this file.
 3.Make this file encoding 'utf-8': type the cmd ': set fileencoding = utf-8'
 4.Check the change: again type the cmd:':set fileencoding' and see the file encoding.

```

### images
```
cd process
python3 scripts/preprocessing/preprocess_images.py --input-dir /Users/xiaofeng/Code/Github/dataset/formula/generate/data_formula_ori --output-dir /Users/xiaofeng/Code/Github/dataset/formula/generate/images_processed
```

### labels

```
python3 scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file /Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/im2latex_dataset/generate/im2latex_formulas.lst --output-file /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/formulas.norm.lst
```

### train.filter
```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /Users/xiaofeng/Code/Github/dataset/formula/generate/images_processed --label-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/formulas.norm.lst --data-path /Users/xiaofeng/Code/Github/im2latex-tensorflow/im2latex-dataset/generate/train.list --output-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/train_filter.lst
```

### validate.filter
```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /Users/xiaofeng/Code/Github/dataset/formula/generate/images_processed --label-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/formulas.norm.lst --data-path /Users/xiaofeng/Code/Github/im2latex-tensorflow/im2latex-dataset/generate/validate.list --output-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/validate_filter.lst
```
### vocabulary
```
python3 scripts/preprocessing/generate_latex_vocab.py --data-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/train_filter.lst --label-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/formulas.norm.lst --output-file /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/latex_vocab.txt
```

# cmd_in_remote_enhance

### **images**
```
cd im2markup
python3 scripts/preprocessing/preprocess_images.py --input-dir /home/xiaofeng/data/formula/generate_enhance/ori/img_ori --output-dir /home/xiaofeng/data/formula/generate_enhance/prepared/images_processed
```

### **labels**

```
python3 scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file /home/xiaofeng/data/formula/generate_enhance/ori/im2latex_formulas_enhance.lst --output-file /home/xiaofeng/data/formula/generate_enhance/prepared/formulas.norm.lst

move from: /home/xiaofeng/data/formula/generate_enhance/ori/im2latex_formulas_enhance.lst
       to:  /home/xiaofeng/data/formula/generate_enhance/prepared/formulas.norm.lst
and rename.
```

### **train.filter**
```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /home/xiaofeng/data/formula/generate_enhance/prepared/images_processed --label-path /home/xiaofeng/data/formula/generate_enhance/prepared/formulas.norm.lst --data-path /home/xiaofeng/data/formula/generate_enhance/ori/train_enhance.list --output-path /home/xiaofeng/data/formula/generate_enhance/prepared/train_filter.lst
```

### **validate.filter**
```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /home/xiaofeng/data/formula/generate_enhance/prepared/images_processed --label-path /home/xiaofeng/data/formula/generate_enhance/prepared/formulas.norm.lst --data-path /home/xiaofeng/data/formula/generate_enhance/ori/validate_enhance.list --output-path /home/xiaofeng/data/formula/generate_enhance/prepared/validate_filter.lst
```
### **vocabulary**
```
python3 scripts/preprocessing/generate_latex_vocab.py --data-path /home/xiaofeng/data/formula/generate_enhance/prepared/train_filter.lst --label-path /home/xiaofeng/data/formula/generate_enhance/prepared/formulas.norm.lst --output-file /home/xiaofeng/data/formula/generate_enhance/prepared/latex_vocab.txt
```





