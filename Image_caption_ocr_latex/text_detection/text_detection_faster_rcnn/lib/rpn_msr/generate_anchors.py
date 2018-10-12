import numpy as np


def generate_basic_anchors(sizes, base_size=16):
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    anchors = np.zeros((len(sizes), 4), np.int32)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors


def scale_anchor(anchor, h, w):
    x_ctr = (anchor[0] + anchor[2]) * 0.5
    y_ctr = (anchor[1] + anchor[3]) * 0.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    return scaled_anchor


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6, 1)):
    # 只进行竖直方向的区分，对于水平方向不区分
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    """ 对anchor进行修正，将anchor变小，期望检测精度更高-当个字符的宽度约为30，原来的16有点大 """
    # widths = [16]
    widths = [8]
    sizes = []
    for h in heights:
        for w in widths:
            sizes.append((h, w))
    # return generate_basic_anchors(size)
    return generate_basic_anchors(sizes, 8)


if __name__ == '__main__':
    import time

    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed

    embed()
