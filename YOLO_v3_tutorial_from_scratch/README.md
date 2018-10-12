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

## Pipeline

### 全卷积网络

> YOLO仅是用卷积层，所以它是全卷积网络（FCN）。**==它具有75个卷积层，具有跳过连接和上采样层。不使用任何形式的池化，使用具有步幅为2的卷积层来下采样特征图。这有助于防止由于池化导致低级特征的丢失。==**作为FCN，YOLO的输入图像的大小是任意的。然而，在实践中，我们可能想要保持输入大小不变，因为各种问题只有在我们实现算法时才会显示出来。
>
> Conv2d 的输入输出计算公式：
>
> input: (N,C_in,H_in,W_in) 
> output: (N,C_out,H_out,W_out)
> $$
> H_{out}=floor((H_{in}+2padding[0]-dilation[0](kernerl_size[0]-1)-1)/stride[0]+1)
> $$
>
> $$
> W_{out}=floor((W_{in}+2padding[1]-dilation[1](kernerl_size[1]-1)-1)/stride[1]+1)
> $$
>
> 其中的一个重要问题是，如果我们想要批量处理图像（批量图像可以由GPU并行处理，从而提高速度），我们需要固定所有图像的高度和宽度。这是为了将多个图像级联成一个大批量（将多个PyTorch张量连接成一个）
>
> 网络通过称为网络步幅的因子对图像进行下采样。例如，如果网络的步幅为32，则尺寸为416 x 416的输入图像将产生尺寸为13 x 13的输出。一般而言，网络中任何层的步幅等于该层的输出的尺寸比网络的输入图像的尺寸小的倍数。

#### Interpreting the output

> In YOLO, the prediction is done by using a convolutional layer which uses 1 x 1 convolutions
>
> Now, the first thing to notice is our **output is a feature map**. Since we have used 1 x 1 convolutions, the size of the prediction map is exactly the size of the feature map before it. In YOLO v3 (and it's descendants), the way you interpret this prediction map is that each cell can predict a fixed number of bounding boxes.
>
>



