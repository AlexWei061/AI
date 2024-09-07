# coding = utf-8
from mxnet import nd

w = nd.arange(4).reshape((1, 1, 2, 2))
b = nd.array([1])
data = nd.arange(9).reshape((1, 1, 3, 3))
out = nd.Convolution(data, w, b, kernel = w.shape[2:], num_filter = w.shape[0])                 # convolution:卷积, kernal:weight的2*2, num_filter: 
print("input :", data, "\n\nweight :", w, "\n\nbias :", b, "\n\noutput :", out)
out = nd.Convolution(data, w, b, kernel = w.shape[2:], num_filter = w.shape[0], 
                     stride = (2, 2), pad = (1, 1))                                             # stride:每次移动的距离 pad:矩阵向外扩展的距离
print("input :", data, "\n\nweight :", w, "\n\nbias :", b, "\n\noutput :", out)

w = nd.arange(8).reshape((1, 2, 2, 2))                                                          # 输入有多个矩阵
b = nd.array([1])
data = nd.arange(18).reshape((1, 2, 3, 3))
out = nd.Convolution(data, w, b, kernel = w.shape[2:], num_filter = w.shape[0])
print("input :", data, "\n\nweight :", w, "\n\nbias :", b, "\n\noutput :", out)

w = nd.arange(16).reshape((2, 2, 2, 2))                                                         # 输出有多个矩阵
b = nd.array([1, 2])
data = nd.arange(18).reshape((1, 2, 3, 3))
out = nd.Convolution(data, w, b, kernel = w.shape[2:], num_filter = w.shape[0])
print("input :", data, "\n\nweight :", w, "\n\nbias :", b, "\n\noutput :", out)

data = nd.arange(18).reshape((1, 2, 3, 3))                                                      # 关于pooling
max_pool = nd.Pooling(data = data, pool_type = 'max', kernel = (2, 2))
avg_pool = nd.Pooling(data = data, pool_type = 'avg', kernel = (2, 2))
print("data :", data, "\n\nmax pool :", max_pool, "\n\navg pool :", avg_pool)