'''
参考资料：
PyTorch官方文档 https://pytorch.org/docs/stable/index.html
孙玉林，余本国. PyTorch深度学习入门与实践 (案例视频精讲)
'''

'''
导入使用的包和模块
'''
import PIL  # 简单地导入PIL库
from PIL import Image  # 直接导入PIL库中的Image类


'''
读取图片
'''
image = Image.open("./Lena.jpg")
# 查看图像的格式和色彩模式
print(image.format)# 输出：JPEG
print(image.mode)# 输出：RGB


'''
可视化图片
'''
# 方式一
image.show()

# 方式二
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()


'''
保存图片到指定位置
'''
image.save("./test.jpg")


'''
三种图像转换
'''
# image为RGB图像
# 将彩色Lena图转换为灰度图像
img_gray = image.convert('L')
img_gray.show()

# 将彩色Lena图转换为二值图像
img_binary = image.convert('1')
img_binary.show()

# 三种图像的色彩模式
print(image.mode) # 输出：RGB
print(img_binary.mode) # 输出：1
print(img_gray.mode) # 输出：L


'''
图像的几何变换
'''
# 将图像缩放到原尺寸的一半大小
image_r = image.resize((128,128))
print(image_r.size) # 输出：(128, 128)

# 裁剪(60,60)至(200,200)框定的区域
image_c = image.crop((60,60,200,200))
image_c.show()

# 将图像旋转90度
image_ro = image.rotate(90)
image_ro.show()

# 上下翻转图片
image_t = image.transpose(Image.FLIP_TOP_BOTTOM)
image_t.show()


'''
图像增强过滤器
'''
from PIL import ImageFilter

# 模糊效果
img_filter = image.filter(ImageFilter.BLUR)
img_filter.show()


'''
图像增强效果
'''
from PIL import ImageEnhance

# 将图片亮度提高2倍
img_enhance = ImageEnhance.Brightness(image).enhance(2)
img_enhance.show()


######################################################################
# 张量及张量操作
######################################################################

'''
查看张量的属性
'''
# 简单地生成一个张量
import torch
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
print(x)

# 张量的元素值
print("x.data: \n", x.data)

# 张量的形状
print("x.shape: ", x.shape)

# reshape改变张量的形状，改变前后要保持元素个数相同
y = torch.arange(6)
print(y, y.shape)
y = y.reshape(2, 3)
print(y, y.shape)

# 张量的数据类型
print("x.dtype: ", x.dtype)

# 张量是否可以计算梯度
print("x.requires_grad: ", x.requires_grad)

# 改变张量的requires_grad属性
x.requires_grad = False
print("x.requires_grad: ", x.requires_grad)


'''
直接生成张量
'''
import torch
# torch.tensor()直接生成
tensor1 = torch.tensor([[1, 2], [3, 4]])
print(tensor1)

# 直接生成指定数据类型的张量
tensor2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
print(tensor2, tensor2.dtype)

# torch.Tensor()直接生成torch.float32类型的张量
tensor3 = torch.Tensor([[1, 2], [3, 4]])
print(tensor3, tensor3.dtype)

# 通过Numpy数组转化生成
import numpy as np

arr = np.array([[1, 2], [3, 4]])
print(arr, type(arr))
tensor4 = torch.from_numpy(arr)
print(tensor4, type(tensor4))

# 张量转化为Numpy数组
arr_t = tensor4.numpy()
print(arr_t, type(arr_t))


'''
特殊张量生成
'''
# 3×3大小的全为0的张量
tensor5 = torch.zeros(3, 3)
print(tensor5)

# 生成一个和tensor1形状相同，值全为0的张量
tensor6 = torch.zeros_like(tensor1)
print(tensor6)

# 在[0, 7)范围内以步长为1生成张量
tensor7 = torch.arange(start=0, end=7, step=1)
print(tensor7)

# 在[0, 10]范围内找到5个等间隔的点值，并使用这些值生成一维张量
tensor8 = torch.linspace(start=0, end=10, steps=5)
print(tensor8)

# 在[0, 2.0]范围内生成以对数为间距的张量
tensor9 = torch.logspace(start=0.0, end=2.0, steps=3)
print(tensor9)


'''
生成随机数张量
'''
# 使用(0, 1)的均匀分布中的数值随机生成一个3×3的张量
tensor10 = torch.rand(3,3)
print(tensor10)

# 随机生成一个3×3的张量，取值范围为[0, 6)内的整数
tensor11 = torch.randint(0, 6, (3,3))
print(tensor11)

# 通过指定的均值和标准差生成随机数
tensor12 = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(1.0))
print(tensor12)

# 生成服从分布均值全为0，分布标准差分别为1、2、3、4的随机数
stds = torch.arange(1.0, 5.0)
tensor13 = torch.normal(mean=torch.tensor(0.0), std=stds)
print(tensor13)

# 生成服从（0，1）正态分布的随机数
tensor14 = torch.randn(3,3)
print(tensor14)


'''
张量的维度
'''
# 构造0维张量并输出张量及其维度数
tensor_z = torch.tensor(0.1)
print(tensor_z, tensor_z.ndim)

# 构造1维张量
tensor_o = torch.rand(3)
print(tensor_o, tensor_o.ndim)

# 构造二维张量
tensor_t = torch.rand(2, 3)
print(tensor_t, tensor_t.ndim)

# 构造三维张量
tensor_s = torch.rand(2, 3, 4)
print(tensor_s, tensor_s.ndim)

# 构造二维形状为2×3大小的张量
t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print("t.ndim: ", t.ndim)
print("t.shape: ", t.shape)

# 使用torch.unsqueeze()对张量t在0维度添加维数为1的维度
t_uns = torch.unsqueeze(t, dim=0)
print(t_uns)
print("t_uns.ndim: ", t_uns.ndim)
print("t_uns.shape: ", t_uns.shape)

# 使用torch.squeeze()去除维数为1的维度
t_s1 = torch.squeeze(t_uns)
print("t_s1.ndim: ", t_s1.ndim)
print("t_s1.shape: ", t_s1.shape)

# 使用torch.cat()在指定维度上对输入的张量进行连接
t1 = torch.arange(6).reshape(2, 3)
t2 = torch.arange(6, 12).reshape(2, 3)
# 从第0维度连接
t3 = torch.cat((t1,t2), dim=0)
print(t3)
# 从第1维度连接
t4 = torch.cat((t1,t2), dim=1)
print(t4)


'''
张量索引
'''
tp = torch.arange(24).reshape(2, 3, 4)  # 构造一个新张量
print(tp)

# 使用索引获取张良指定位置的元素值

# 获取第0维度下第2个矩阵的第一行第一列的元素
print(tp[1, 0, 0])

# 输出第0维度下第一个矩阵的元素
print(tp[0])# 等同于tp[0, :, :]

# 输出第0维度下第一个矩阵前两行元素
print(tp[0, 0:2]) # 等同于tp[0, 0:2, :]

# 输出第0维度下第一个矩阵前两行前两列元素
print(tp[0, 0:2, 0:2])

# 通过索引操作张量的函数

# 沿指定维度0对输入进行切片，取torch.tensor(1)中指定的相应项
x1 = torch.index_select(tp, 0, torch.tensor(1))
print(x1)


'''
张量计算，简单的 加减乘除 逐元素计算
'''
# 张量与张量逐元素计算
tensor1 = torch.arange(6.0).reshape(2,3)
tensor2 = torch.arange(6.0, 12.0).reshape(2, 3)
print(tensor1)
print(tensor2)

tensor3 = tensor1 + tensor2 # 逐元素相加
print(tensor3)

tensor4 = tensor2 - tensor1 # 逐元素相减
print(tensor4)

tensor5 = tensor1 * tensor2 # 逐元素相乘
print(tensor5)

tensor6 = tensor2 / tensor1 # 逐元素相除
print(tensor6)

tensor7 = tensor2 // tensor1 # 逐元素整除
print(tensor7)

# 张量与常量逐元素计算
print(tensor1 + 2) # 逐元素加2

print(tensor1 - 2) # 逐元素减2

print(tensor1 / 2) # 逐元素除以2

print(tensor1 * 2) # 逐元素乘2

print(tensor1 // 2) # 逐元素整除2

print(tensor1 ** 2) # 计算张量的幂

tensor8 = torch.pow(tensor1, 2) # 计算张量的幂
print(tensor8)

tensor9 = torch.exp(tensor1) # 计算张量的指数

# 张量的矩阵运算
tensor10 = torch.t(tensor1) # 取矩阵的转置
print(tensor10)

x = torch.tensor([[1.,0.,0.],[2.,-1.,0.],[2.,1.,1.]])
tensor11 = torch.inverse(x) # 取矩阵的逆
print(x)

tensor12 = torch.mm(tensor1, x) # 矩阵乘法
print(tensor12)



