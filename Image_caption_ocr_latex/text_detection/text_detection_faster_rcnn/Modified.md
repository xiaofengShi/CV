1. 修改anchor尺寸
   - 2018年07月09日17:34:11 
   - 修改文件：[./lib/rpn_msr/generate_anchors.py](./lib/rpn_msr/generate_anchors.py)   Line:35,36
     - 修改文件：[VGGtrain.py](./lib/networks/VGGnet_train.py) 参数‘anchor_sacles’,'n_classes'
   - 修改原因：目前公式和文字之间边界存在误判，将anchor的宽度减小，由原来的16修改为8
2. 