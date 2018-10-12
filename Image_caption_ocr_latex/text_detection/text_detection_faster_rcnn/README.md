# REPO INTRODUCTION

## 1. 目的

- 进行场景不定长文字区域的识别和提取，输出该区域box

## 2. pipeline：

### 2.1 Pipeline ctpn

- ctpn的网络设计和faster_rcnn非常相似，只是将faster_rcnn的rpn网络区域提议部分加入了bilstm的设计，并使用了固定宽度，不同高度方向的anchor
- ctpn只设计文本框的回归，不涉及文本和背景之间的分类计算，也就是指包含faster_rcnn的rpn损失，不包含roi_pooling之后的rcnn的损失计算

### 2.2 网络修改

- 保留原来的ctpn的完整网络结构，将双向LSTM变成多层双向LSTM；
- 在ctpn之后加入了roipooling和rcnn的分类损失计算，使网络理论上编程一种目标识别的框架
  - **TODO**
    - 测试网络识别结果，编写测试模块
    - 如果测试结果👌，进行整体网络数据流的搭建
    - 如果测试结果不好，尝试其他方法

## 3. 环境部署实现

### 3.1 在GPU版本要进行环境设置

```
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
#### 3.1.1 reference

- [error: roi_pooling_op.cu.o: No such file or directory](https://github.com/CharlesShang/TFFRCNN/issues/34)

- [ identifier "**__builtin_ia32_mwaitx" is undefined**]-[./lib/make.sh](./lib/make.sh) 

  ```
  TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
  TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
  
  CUDA_PATH=/usr/local/cuda/
  CXXFLAGS=''
  
  if [[ "$OSTYPE" =~ ^darwin ]]; then
          CXXFLAGS+='-undefined dynamic_lookup'
  fi
  
  cd roi_pooling_layer
  
  if [ -d "$CUDA_PATH" ]; then
          nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
                  -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS --expt-relaxed-constexpr\
                  -arch=sm_37
  
          g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc  -D_GLIBCXX_USE_CXX11_ABI=0  \
                  roi_pooling_op.cu.o -I $TF_INC -I $TF_INC/external/nsync/public  -L $TF_LIB -D GOOGLE_CUDA=1  -ltensorflow_framework -fPIC $CXXFLAGS \
                  -lcudart -L $CUDA_PATH/lib64
  else
          g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
                  -I $TF_INC -fPIC $CXXFLAGS
  fi
  
  cd ..
  
  #cd feature_extrapolating_layer
  
  #nvcc -std=c++11 -c -o feature_extrapolating_op.cu.o feature_extrapolating_op_gpu.cu.cc \
  #       -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50
  
  #g++ -std=c++11 -shared -o feature_extrapolating.so feature_extrapolating_op.cc \
  #       feature_extrapolating_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64
  #cd .
  ```

### 3.2 Cython modules 

按照要运行的tensorflow的CPU、GPU版本进行Cython modules的制作

- **如果使用的是CPU版本，运行[setup-python3](./setup-python3.sh)进行Cython modules的制作**
- **如果使用的GPU版本，运行[setup_remote.sh](./setup_gpu.sh)进行Cython modules的制作**



















