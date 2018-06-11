## YOLO_tensorflow

Tensorflow implementation of [YOLO](https://arxiv.org/pdf/1506.02640.pdf), including training and test phase.

### Installation

1. Clone yolo_tensorflow repository
	```Shell
	$ git clone https://github.com/hizhangp/yolo_tensorflow.git
    $ cd yolo_tensorflow
	```

2. Download Pascal VOC dataset, and create correct directories
	```Shell
	$ ./download_data.sh
	```

3. Download [YOLO_small](https://drive.google.com/file/d/0B5aC8pI-akZUNVFZMmhmcVRpbTA/view?usp=sharing)
weight file and put it in `data/weight`

4. Modify configuration in `yolo/config.py`

5. Training
	```Shell
	$ python train.py
	```

6. Test
	```Shell
	$ python test.py
	```

### THRORY

- Yolo的CNN网络将输入的图片分割成 ![S\times S](https://www.zhihu.com/equation?tex=S%5Ctimes+S) 网格，然后每个单元格负责去检测那些中心点落在该格子内的目标，如图6所示，可以看到狗这个目标的中心落在左下角一个单元格内，那么该单元格负责预测这个狗。每个单元格会预测 ![B](https://www.zhihu.com/equation?tex=B) 个边界框（bounding box）以及边界框的置信度（confidence score）。所谓置信度其实包含两个方面，一是这个边界框含有目标的可能性大小，二是这个边界框的准确度。前者记为 ![Pr(object)](https://www.zhihu.com/equation?tex=Pr%28object%29) ，当该边界框是背景时（即不包含目标），此时 ![Pr(object)=0](https://www.zhihu.com/equation?tex=Pr%28object%29%3D0) 。而当该边界框包含目标时， ![Pr(object)=1](https://www.zhihu.com/equation?tex=Pr%28object%29%3D1) 。边界框的准确度可以用预测框与实际框（ground truth）的IOU（intersection over union，交并比）来表征，记为 ![\text{IOU}^{truth}_{pred}](https://www.zhihu.com/equation?tex=%5Ctext%7BIOU%7D%5E%7Btruth%7D_%7Bpred%7D)。因此置信度可以定义为 ![Pr(object)*\text{IOU}^{truth}_{pred}](https://www.zhihu.com/equation?tex=Pr%28object%29%2A%5Ctext%7BIOU%7D%5E%7Btruth%7D_%7Bpred%7D) 。很多人可能将Yolo的置信度看成边界框是否含有目标的概率，但是其实它是两个因子的乘积，预测框的准确度也反映在里面。边界框的大小与位置可以用4个值来表征： ![(x, y,w,h)](https://www.zhihu.com/equation?tex=%28x%2C+y%2Cw%2Ch%29) ，其中 ![(x,y)](https://www.zhihu.com/equation?tex=%28x%2Cy%29) 是边界框的中心坐标，而 ![w](https://www.zhihu.com/equation?tex=w) 和 ![h](https://www.zhihu.com/equation?tex=h) 是边界框的宽与高。还有一点要注意，中心坐标的预测值 ![(x,y)](https://www.zhihu.com/equation?tex=%28x%2Cy%29) 是相对于每个单元格左上角坐标点的偏移值，并且单位是相对于单元格大小的，单元格的坐标定义如图6所示。而边界框的 ![w](https://www.zhihu.com/equation?tex=w) 和 ![h](https://www.zhihu.com/equation?tex=h) 预测值是相对于整个图片的宽与高的比例，这样理论上4个元素的大小应该在 ![[0,1]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D) 范围。这样，每个边界框的预测值实际上包含5个元素： ![(x,y,w,h,c)](https://www.zhihu.com/equation?tex=%28x%2Cy%2Cw%2Ch%2Cc%29) ，其中前4个表征边界框的大小与位置，而最后一个值是置信度。

  ![img](https://pic2.zhimg.com/80/v2-fdfea5fcb4ff3ecc327758878e4ad6e1_hd.jpg)

- 还有分类问题，对于每一个单元格其还要给出预测出 ![C](https://www.zhihu.com/equation?tex=C) 个类别概率值，其表征的是由该单元格负责预测的边界框其目标属于各个类别的概率。但是这些概率值其实是在各个边界框置信度下的条件概率，即 ![Pr(class_{i}|object)](https://www.zhihu.com/equation?tex=Pr%28class_%7Bi%7D%7Cobject%29) 。值得注意的是，不管一个单元格预测多少个边界框，其只预测一组类别概率值，这是Yolo算法的一个缺点，在后来的改进版本中，Yolo9000是把类别概率预测值与边界框是绑定在一起的。同时，我们可以计算出各个边界框类别置信度（class-specific confidence scores):![Pr(class_{i}|object)*Pr(object)*\text{IOU}^{truth}_{pred}=Pr(class_{i})*\text{IOU}^{truth}_{pred}](https://www.zhihu.com/equation?tex=Pr%28class_%7Bi%7D%7Cobject%29%2APr%28object%29%2A%5Ctext%7BIOU%7D%5E%7Btruth%7D_%7Bpred%7D%3DPr%28class_%7Bi%7D%29%2A%5Ctext%7BIOU%7D%5E%7Btruth%7D_%7Bpred%7D) 。

- 边界框类别置信度表征的是该边界框中目标属于各个类别的可能性大小以及边界框匹配目标的好坏。后面会说，一般会根据类别置信度来过滤网络的预测框。

- 总结一下，每个单元格需要预测 ![(B*5+C)](https://www.zhihu.com/equation?tex=%28B%2A5%2BC%29) 个值。如果将输入图片划分为 ![S\times S](https://www.zhihu.com/equation?tex=S%5Ctimes+S) 网格，那么最终预测值为 ![S\times S\times (B*5+C)](https://www.zhihu.com/equation?tex=S%5Ctimes+S%5Ctimes+%28B%2A5%2BC%29) 大小的张量。整个模型的预测值结构如下图所示。对于PASCAL VOC数据，其共有20个类别，如果使用 ![S=7,B=2](https://www.zhihu.com/equation?tex=S%3D7%2CB%3D2) ，那么最终的预测结果就是 ![7\times 7\times 30](https://www.zhihu.com/equation?tex=7%5Ctimes+7%5Ctimes+30) 大小的张量。在下面的网络结构中我们会详细讲述每个单元格的预测值的分布位置。

  ![img](https://pic2.zhimg.com/80/v2-258df167ee37b5594c72562b4ae61d1a_hd.jpg)

- Yolo采用卷积网络来提取特征，然后使用全连接层来得到预测值。网络结构参考GooLeNet模型，包含24个卷积层和2个全连接层，如图8所示。对于卷积层，主要使用1x1卷积来做channle reduction，然后紧跟3x3卷积。对于卷积层和全连接层，采用Leaky ReLU激活函数:![max(x, 0.1x)](https://www.zhihu.com/equation?tex=max%28x%2C+0.1x%29) 。但是最后一层却采用线性激活函数。

  ![img](https://pic1.zhimg.com/80/v2-5d099287b1237fa975b1c19bacdfc07f_hd.jpg)

- 可以看到网络的最后输出为 ![7\times 7\times 30](https://www.zhihu.com/equation?tex=7%5Ctimes+7%5Ctimes+30) 大小的张量。这和前面的讨论是一致的。这个张量所代表的具体含义如图9所示。对于每一个单元格，前20个元素是类别概率值，然后2个元素是边界框置信度，两者相乘可以得到类别置信度，最后8个元素是边界框的 ![(x, y,w,h)](https://www.zhihu.com/equation?tex=%28x%2C+y%2Cw%2Ch%29) 。大家可能会感到奇怪，对于边界框为什么把置信度 ![c](https://www.zhihu.com/equation?tex=c) 和 ![(x, y,w,h)](https://www.zhihu.com/equation?tex=%28x%2C+y%2Cw%2Ch%29) 都分开排列，而不是按照 ![(x, y,w,h,c)](https://www.zhihu.com/equation?tex=%28x%2C+y%2Cw%2Ch%2Cc%29) 这样排列，其实纯粹是为了计算方便，因为实际上这30个元素都是对应一个单元格，其排列是可以任意的。但是分离排布，可以方便地提取每一个部分。这里来解释一下，首先网络的预测值是一个二维张量 ![P](https://www.zhihu.com/equation?tex=P) ，其shape为 ![[batch, 7\times 7\times 30]](https://www.zhihu.com/equation?tex=%5Bbatch%2C+7%5Ctimes+7%5Ctimes+30%5D) 。采用切片，那么 ![P_{[:,0:7*7*20]}](https://www.zhihu.com/equation?tex=P_%7B%5B%3A%2C0%3A7%2A7%2A20%5D%7D) 就是类别概率部分，而 ![P_{[:,7*7*20:7*7*(20+2)]}](https://www.zhihu.com/equation?tex=P_%7B%5B%3A%2C7%2A7%2A20%3A7%2A7%2A%2820%2B2%29%5D%7D) 是置信度部分，最后剩余部分 ![P_{[:,7*7*(20+2):]}](https://www.zhihu.com/equation?tex=P_%7B%5B%3A%2C7%2A7%2A%2820%2B2%29%3A%5D%7D) 是边界框的预测结果。这样，提取每个部分是非常方便的，这会方面后面的训练及预测时的计算。

- 

  

  

 

### Requirements

1. Tensorflow

2. OpenCV
