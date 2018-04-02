# import util
#
#
# def test_is_gpu_available():
#     #	for i in range(4):
#     if (util.tf.is_gpu_available()):
#         print("GPU is available, %s CUDA installed" % ('with' if util.tf.is_gpu_available(True) else 'without'))
#
#
# def test_get_available_gpus():
#     devices = util.tf.get_available_gpus()
#     for d in devices:
#         print(d)
#
#
# if util.mod.is_main(__name__):
#     test_is_gpu_available()
#     test_get_available_gpus()


def is_gpu_available(cuda_only=True):
    """
    code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/test.py
    Returns whether TensorFlow can access a GPU.
    Args:
      cuda_only: limit the search to CUDA gpus.
    Returns:
      True iff a gpu device of the requested kind is available.
    """
    from tensorflow.python.client import device_lib as _device_lib

    if cuda_only:
        return any((x.device_type == 'GPU')
                   for x in _device_lib.list_local_devices())
    else:
        return any((x.device_type == 'GPU' or x.device_type == 'SYCL')
                   for x in _device_lib.list_local_devices())


#
def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


x = is_gpu_available(cuda_only=True)
y = get_available_gpus()
print('is_gpu_available:', x)
print('get_available_gpus:', y)

#
if __name__ == '__main__':
    # test_is_gpu_available()
    # test_get_available_gpus()
    is_gpu_available()
    get_available_gpus()
