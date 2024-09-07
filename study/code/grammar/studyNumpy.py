# coding = utf-8

from mxnet import np, npx
npx.set_np()

x = np.arange(12)                                            # 创建一个12维的向量 [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]
print("x : ", x, '\n')
X = x.reshape(3, 4)                                          # reshape 成 3 * 4 的矩阵
print("X : ", X, '\n')
a = np.zeros((1, 3, 4))                                      # 全0
print("a : ", a, '\n')
b = np.ones((1, 3, 4))                                       # 全1
print("b : ", b, '\n')
c = np.random.normal(0, 1, size = (3, 4))                    # 均值为0 标准差为1 正态分布
print("c : ", c, '\n')

x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
print("x + y : ", x + y, '\n')
print("x - y : ", x - y, '\n')
print("x * y : ", x * y, '\n')
print("x / y : ", x / y, '\n')
print("x ** y : ", x ** y, '\n')                             # 以上的计算全都是按元素对应做运算
print("exp(x) : ", np.exp(x), '\n')                          # 这个也是

X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("X : ", X, '\n')
print("Y : ", Y, '\n')
xx = np.concatenate([X, Y], axis = 0)                        # 把两个矩阵竖直拼在一起
print("X Y 竖直拼在一起 :\n", xx, '\n')
yy = np.concatenate([X, Y], axis = 1)                        # 把两个矩阵横向拼在一起
print("X Y 横向拼在一起 :\n", yy, '\n')

print("X 和 Y 每个位置是否对应相等\n", X == Y, '\n')           # 看每个位置是否对应相等
sumx = X.sum()
print("sumx = ", sumx, '\n')                                 # 对 x 求和

a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
print("a :\n", a)
print("b : ", b)
print("a + b :\n", a + b, '\n')                              # 相当于把 a 复制一列并且把 b 复制两行 都变成 3 * 2 的矩阵然后再相加

x = np.arange(12).reshape(3, 4)
print("x :\n", x)
print("x[-1] : ", x[-1])
print("x[1 : 3] :\n", x[1 : 3])
print("x[1, 2] : ", x[1, 2])
x[1, 2] = 9
print("x :\n", x)
x[0:2, :] = 12
print("x :\n", x)