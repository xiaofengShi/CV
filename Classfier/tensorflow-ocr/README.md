# tensorflow-ocr
适用于接口的版本，放在[这里](https://github.com/siriusdemon/hackaway/tree/master/projects/ocr)


### 中文汉字印刷体识别
修改自这个[仓库](https://github.com/soloice/Chinese-Character-Recognition)

主要是修改了数据输入部分的代码，复用了原作者的网络结构和程序结构。

### config.py
定义了识别时的图片大小，训练字符集，模型位置等

### gen.py
用于产生某种字体的字图

### preprocess.py
用于对输入图片进行预处理

### tensorflow-ocr.py
主执行文件

### TODO
+ 去掉tensorflow-ocr.py与config.py的重复定义

### 用法
python tensorflow-ocr.py --mode=train --max_steps=200000 --eval_steps=1000 --save_steps=10000
