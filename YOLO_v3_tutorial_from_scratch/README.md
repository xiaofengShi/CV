# YOLO_v3_tutorial_from_scratch
Accompanying code for Paperspace tutorial series ["How to Implement YOLO v3 Object Detector from Scratch"](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

Here's what a typical output of the detector will look like ;)

![Detection Example](https://i.imgur.com/m2jwnen.png)

## About the training Code

This code is only mean't as a companion to the tutorial series and won't be updated. If you want to have a look at the ever updating YOLO v3 code, go to my other repo at https://github.com/ayooshkathuria/pytorch-yolo-v3

The pretrained weights can get use the cmd 

```shell
wget https://pjreddie.com/media/files/yolov3.weights 
```

Also, the other repo offers a lot of customisation options, which are not present in this repo for making tutorial easier to follow. (Don't wanna confuse the shit out readers, do we?)

![]()

Minimal implementation of YOLOv3 in PyTorch. By the way, I think Pytorch is a better framwork compared with tensorflow and keras, so I will complete this repo by pytorch.

## 1. Paper

### YOLOv3: An Incremental Improvement

_Joseph Redmon, Ali Farhadi_ 

**Abstract** 
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Original Implementation]](

## Pipeline

### 2.1 FCN

YOLO仅是用卷积层，所以它是全卷积网络（FCN）。**==它具有75个卷积层，具有跳过连接(densenet)和上采样层。不使用任何形式的池化，使用具有步幅为2的卷积层来下采样特征图。这有助于防止由于池化导致低级特征的丢失。==**作为FCN，YOLO的输入图像的大小是任意的。然而，在实践中，我们可能想要保持输入大小不变。

Conv2d 的输入输出计算公式（pytorch）：

input: (N,C_in,H_in,W_in) 
output: (N,C_out,H_out,W_out)
$$
\left\{
\begin{array}{lr}
H_{out}=floor((H_{in}+2padding[0]-dilation[0](kernerl_{size[0]}-1)-1)/stride[0]+1) &(1)\\
W_{out}=floor((W_{in}+2padding[1]-dilation[1](kernerl_{size[1]}-1)-1)/stride[1]+1) &(2)
\end{array}
\right.
$$
其中的一个重要问题是，如果我们想要批量处理图像（批量图像可以由GPU并行处理，从而提高速度），我们需要固定所有图像的高度和宽度。这是为了将多个图像级联成一个大批量（将多个PyTorch张量连接成一个）

网络通过称为网络步幅的因子对图像进行下采样。例如，如果网络的步幅为32，则尺寸为$416*416$的输入图像将产生尺寸为$13*13$的输出。一般而言，网络中任何层的步幅等于该层的输出的尺寸比网络的输入图像的尺寸小的倍数。

### 2.2 Interpreting the output

In YOLO3, the prediction is done by using a convolutional layer which uses 1 x 1 convolutions.

Now, the first thing to notice is our **output is a feature map**. Since we have used 1 x 1 convolutions, the size of the prediction map is exactly the size of the feature map before it. In YOLO v3 (and it's descendants), the way you interpret this prediction map is that each cell can predict a fixed number of bounding boxes.

**Depth-wise, we have (B x (5 + C)) entries in the feature map.** 

- B represents the number of bounding boxes each cell can predict.According to the paper, each of these B bounding boxes may specialize in detecting a certain kind of object. <u>Each of the bounding boxes have *5 + C* attributes,</u> which describe the center coordinates, the dimensions, the objectness score and *C* class confidences for each bounding box. <u>YOLO v3 predicts 3 bounding boxes for every cell.</u>

**We expect each cell of the feature map to predict an object through one of it's bounding boxes if the center of the object falls in the receptive field of that cell.** (Receptive field is the region of the input image visible to the cell. Refer to the link on convolutional neural networks for further clarification).

We predict the box in three differdent feature size: **[$13*13,26*26,52*52$]**, Firstly, in the latest down convolution, and the feature map is 13x13, and each cell has three anchors to predict the grounding truth; Then contact the up convuluntion featuremap and the down convolution in the same size wihic is 26x26 and 52x52, and samely each cell has three anchors. so total  proposal box number is
$$
(13*13+26*26+52*52)*3=10647
$$
each box has such prediction: four box shift distance, one object score and C class prediction scores which C is the classes num.
$$
[center_{x},center_{y},width,height,object_{score},cls_{1},cls_{2},...cls_{C}]
$$
Look the picture bellow, each anchor box on a cell in  the feature map has ==$B*(5+c)$==  prediction output.





## RAEFERENCE

- [PYTORCH_YOLO_V3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
  - 原理讲解博客[here](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/)
  - blog对应repop,只能检测无法训练[here](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch)
- [Keras_yolo_v3](https://github.com/qqwweee/keras-yolo3)