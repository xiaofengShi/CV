 从数据库获得数据并转化成公式图片
===========================
## Great reference
* [Dataset ori repo is here](https://github.com/Miffyli/im2latex-dataset)
* 参考---[rf1--将pdf转成图片](https://www.jianshu.com/p/fd46db1d1fee)
* pdf_to_img.py 将pdf文件转成png文件，保留源文件的色彩信息,
* required:PyPDF2==1.26.0,Wand==0.4.4

* 直接运行**data_generate.py** 生成--图像，formula列表

    生成图像之后，使用process内程序进行图像和公式的标准化处理,具体的操作方法，请查询**cmd.md**文件
## Prerequsites--需要安装的依赖软件

`Most of the code is written in tensorflow, with Python for preprocessing.`

#### Preprocess
The proprocessing for this dataset is exactly reproduced as the original torch implementation by the HarvardNLP group

Python

* Pillow
* numpy

Optional: We use Node.js and KaTeX for preprocessing [Installation](https://nodejs.org/en/)

##### pdflatex [Installaton](https://www.tug.org/texlive/)

Pdflatex is used for rendering LaTex during evaluation.

##### ImageMagick convert [Installation](http://www.imagemagick.org/script/index.php)

Convert is used for rending LaTex during evaluation.

- linux `sudo apt install imagemagick`
- Mac `brew install imagemagick`

##### Webkit2png [Installation](http://www.paulhammond.org/webkit2png/)

Webkit2png is used for rendering HTML during evaluation.

### Preprocessing Instructions

The images in the dataset contain a LaTeX formula rendered on a full page. To accelerate training, we need to preprocess the images.

```
cd dataset/process
```
```
python scripts/preprocessing/preprocess_images.py --input-dir ../formula_images --output-dir ../images_processed
```

The above command will crop the formula area, and group images of similar sizes to facilitate batching.

Next, the LaTeX formulas need to be tokenized or normalized.

```
python scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file ../im2latex_formulas.lst --output-file formulas.norm.lst
```

The above command will normalize the formulas. Note that this command will produce some error messages since some formulas cannot be parsed by the KaTeX parser.

Then we need to prepare train, validation and test files. We will exclude large images from training and validation set, and we also ignore formulas with too many tokens or formulas with grammar errors.

```
python scripts/preprocessing/preprocess_filter.py --filter --image-dir ../images_processed --label-path formulas.norm.lst --data-path ../im2latex_train.lst --output-path train.lst
```

```
python scripts/preprocessing/preprocess_filter.py --filter --image-dir ../images_processed --label-path formulas.norm.lst --data-path ../im2latex_validate.lst --output-path validate.lst
```

```
python scripts/preprocessing/preprocess_filter.py --no-filter --image-dir ../images_processed --label-path formulas.norm.lst --data-path ../im2latex_test.lst --output-path test.lst
```

Finally, we generate the vocabulary from training set. All tokens occuring less than (including) 1 time will be excluded from the vocabulary.

```
python scripts/preprocessing/generate_latex_vocab.py --data-path train.lst --label-path formulas.norm.lst --output-file latex_vocab.txt
```

Train, Test and Valid images need to be segmented into buckets based on image size (height, width) to facilitate batch processing.
