# pytorch笔记

## 软件使用

#### 打开jupyter(pytorch环境):

1、在Anaconda Prompt中用conda activate pytorch进入pytorch环境

2、输入jupyter notebook自动跳转到jupyter网页(想要打开笔记cd到文件夹中打开)



#### 启动TensorBoard

在终端中输入tensorboard --logdir=logs，打开网址，与SummaryWritter类共同使用

```python
(pytorch) PS D:\ai\project\pytorch_study\pythonProject1> tensorboard --logdir=logs
							#logs为日志所在的文件夹，取决于SummaryWriter("logs")中引号的内容
```

#### **命令行conda创建环境进入环境**

```python
conda create --name yourEnv python=2.7	# 创建新的conda环境
conda activate yolov8				# 进入名字为yolov8的环境
conda deactivate					# 退出当前环境
conda env list						# 查看conda环境包的列表
conda remove XXX					# 删除 XXX 包
conda env remove -n XXX				# 删除指定环境
conda info							# 查看各种信息
pip install -r .\requirements.txt	# 按照所需包
# 更改conda环境名字
conda create --name 新环境名 --clone 旧环境名
conda remove --name 旧环境名 --all
```

#### **永久更改为清华源**

```python
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

单词使用

```python
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其他源

```
豆瓣 https://pypi.doubanio.com/simple/
网易 https://mirrors.163.com/pypi/simple/
阿里云 https://mirrors.aliyun.com/pypi/simple/
腾讯云 https://mirrors.cloud.tencent.com/pypi/simple
清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/
```

取消所有源，替换为默认

```
pip config unset global.index-url
```

查看pytorch的安装是否支持cuda

```python
import torch
print(torch.__version__)		# eg:2.2.2+cpu则表示为cpu版本，2.2.2+cu121则是有gpu
print(torch.version.cuda)
print(torch.cuda.is_available())  #输出为True，则安装无误

help(torch.cuda.is_available) # 查看 torch.cuda.is_available 的用法
dir(torch)  # 查看torch包中有哪些区、有哪些工具
```

pip指令的一些操作

```python
pip install cv2
pip uninstall cv2
pip list
pip install --upgrade cv2
pip install cv2==4.5.2
pip show cv2		# 查看库的信息
```

终端中直接执行python文件

```python
python 文件名.py --weight yolov5m.py --img 1280
```

运行时默认赋予参数值

<img src="D:\tool\typora\image\image-20240425145900331.png" alt="image-20240425145900331" style="zoom:50%;" />

<img src="D:\tool\typora\image\image-20240425145843248.png" alt="image-20240425145843248" style="zoom: 50%;" />

**从GitHub克隆最新源代码**

```python
git clone https://github.com/tzutalin/labelImg.git
cd labelImg
```



## 特征提取

传统的一些特征提取算法

### HOG

​		HOG主要捕获**轮廓信息**。求取前先灰度化然后Gamma校正，降低图像局部的阴影和光照变化所造成的影响，同时可以抑制噪音的干扰。

<img src="D:\tool\typora\image\image-20240419095037949.png" alt="image-20240419095037949" style="zoom: 67%;" />

**图像归一化**

​		图像归一化是为了减少光照等的影响，降低图像局部的阴影，避免在图像的纹理强度中，局部表层曝光较大的情况。归一化的类别有：gamma空间归一化和颜色空间归一化。

**梯度幅值和梯度方向**

​		计算图像横坐标和纵坐标方向的梯度，并根据横坐标和纵坐标的梯度，计算梯度大小和梯度方向。在算法中，通常利用梯度算子（例如：[-1,0,1]）对原图像做卷积运算，得到水平方向（x）的梯度值，再利用转置的梯度算子对原图像做卷积运算，得到竖直方向（y）的梯度值。最后通过上述公式计算该像素点的梯度大小和方向。

**块内归一化操作**

​		由于局部光照的变化以及前景-背景对比度的变化，使得梯度强度的变化范围非常大。这就需要对梯度强度做归一化。归一化能够进一步地对光照、阴影和边缘进行压缩，使特征向量空间对光照，阴影和边缘变化具有鲁棒性。具体做法是把各个细胞单元组合成大的、空间上连通的区域（blocks）。这样，一个block内所有cell的特征向量串联起来便得到该block的HOG特征。这些区间是互有重叠的，这就意味着：每一个单元格的特征会以不同的结果多次出现在最后的特征向量中。归一化之后的块描述符（向量）就称之为HOG描述符。

​		将灰度图像像素点划分成一个或多个**窗口（win）**，一个窗口再分为多个**块（block）**，一个块再分为多个**细胞单元（cell）**。将梯度方向按一定角度分开，通常在0-180度（无向）或0-360度（有向）范围内。例如采用无向的梯度和9个直方图通道，则方向的范围划分为180/9=20度，方向即划分为9个**箱（bin）**。

<img src="D:\tool\typora\image\image-20240419095442329.png" alt="image-20240419095442329" style="zoom: 33%;" />

z1 ~ z9的9个bin的值代表该梯度方向上累加的梯度幅值，对每个cell区域内的所有像素按其梯度方向循环累加，得到该cell区域的梯度向量值。

一个窗口（win）的总HOG特征数为：block总数 × block中特征数，即：

​		(window_width / step -1)*(window_height / step -1)} * {num_of_bin * (block_width/cell_width)*(block_height/cell_height)}

**eg**:	win: 64x128;  block: 16x16;  cell: 8x8; 步长: 8;  bin个数: 9。每个块内36个特征。

则block总数为：(64/8 - 1)*(128/8 - 1)=105，那么一个win的总特征数为：105×(9×2×2) = 3780。



<img src="D:\tool\typora\image\image-20240419095738507.png" alt="image-20240419095738507" style="zoom:33%;" />

### SIFT

​		SIFT（Scale-invariant feature transform）特征提取算法，即尺度不变性特征变换。SIFT算法确立的特征点稳定性很好，不会因为尺度变化、旋转变化、光照的变化、噪音的干扰而受影响。SIFT算法的实现步骤总的来说可分为两步：特征点检出和特征点描述。SIFT提取的关键点是**角点**（个人理解：应该是角点+梯度直方图）。

**特征点检出**

​	SIFT算法的第一步是找到足够多的特征点。主要是用了DoG，就是把图像做不同程度的高斯模糊，平滑的区域一般变化不大，而纹理复杂的比如边缘、点、角之类区域一般变化很大，将相邻两张图片做差，这样变化很大的点就是特征点。当然为了找到足够的点，还需要把图像放大、缩小几倍来重复这个步骤找特征点。

**寻找极值点（特征点）**

​		一幅DoG图像，让目标点跟它周围的8邻域的8个点比较，并且跟它相邻尺度的DoG图像做三维的空间比较，因此一个目标点总共会同周边26个点比较，如下图。如果目标点经过如此比较后，确实是这26个点中的极大或极小值，就认为该点是图像在该尺寸下的极值点。

<img src="D:\tool\typora\image\image-20240419103433850.png" alt="image-20240419103433850" style="zoom: 33%;" />

​		按照上述比较法，找到多尺度DoG空间的所有极值点。但这些极值点都是离散的，在离散的空间中，局部极值点可能并不是真正意义上的极值点。因此，采用插值法进一步求得真正的极值点。如下图所示，利用已知的离散空间点插值得到连续空间极值点，通过对尺度空间DoG函数进行曲线拟合，进一步对函数方程求偏导，得到精确的极值点。

<img src="D:\tool\typora\image\image-20240419103524637.png" alt="image-20240419103524637" style="zoom:33%;" />

此外，还需要删除边缘效应的点。DoG的值会受到边缘的影响。

**获取特征点描述**

​		检出足够多的特征点后，我们就可以开始计算这些特征点的特征得到对应特征点的描述。这一步和HOG算法类似，即以检出的特征点为中心选16x16的区域作为local patch，这个区域又可以均分为4x4个子区域。对于每一个关键点，都拥有位置、尺度以及方向三个信息。为每个关键点建立一个描述符，用一组向量将这个关键点描述出来，使其不随各种变化而改变，比如光照变化、视角变化等等。

**选取特征点的方向**

​		采集特征点所在高斯金字塔图像3σ邻域窗口内像素的梯度和方向分布特征。在完成特征点的梯度计算后，使用直方图统计邻域内像素的梯度和方向。子区域中各个像素的梯度都可以分到8个bin（类似HOG算法）里面。梯度直方图将0~360度的方向范围分为36个柱，其中每柱10度。如图所示，直方图的峰值则代表了该特征点处邻域梯度的主方向，即作为该特征点的方向，其他的达到最大值80%的方向可作为辅助方向。

<img src="D:\tool\typora\image\image-20240419103745679.png" alt="image-20240419103745679" style="zoom:33%;" />

**选取特征点的方向**

​		SIFT描述子是特征点邻域高斯图像梯度统计结果的一种表示。通过对特征点周围图像区域分块，计算块内梯度直方图，生成具有独特性的向量。特征描述子与特征点所在的尺度有关，因此，对梯度的求取应在特征点对应的高斯图像上进行。将关键点附近的邻域划分为d*d(Lowe建议d=4)个子区域，每个子区域做为一个种子点，每个种子点有8个方向。

​		每一个小格都代表了特征点邻域所在的尺度空间的一个像素 ，箭头方向代表了像素梯度方向，箭头长度代表该像素的幅值。然后在4×4的窗口内计算8个方向的梯度方向直方图。绘制每个梯度方向的累加可形成一个种子点。这样就可以对每个特征点形成一个4*4*8=128维的描述子。

<img src="D:\tool\typora\image\image-20240419103919375.png" alt="image-20240419103919375" style="zoom: 50%;" />

HOG和SIFT两者特点对比如下：

| **步骤**   | **HOG**    | **SIFT**                         |
| ---------- | ---------- | -------------------------------- |
| 预处理1    | 图像灰度化 | 图像缩放（多个不同分辨率的图像） |
| 预处理2    | 图像归一化 | 高斯模糊、DoG特征图计算          |
| 描述子计算 | 分块计算   | 计算极值点、分块计算             |

| **对比项** | **HOG**                                                      | **SIFT**                                                     |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 特征特点   | 单一尺度                                                     | 多尺度                                                       |
| 应用领域   | 目标检测目标跟踪                                             | 图像匹配三维建模                                             |
| 优点       | 图像几何和光学形变都保持良好的不变性、刚性物体特征提取效果好 | 尺度不变性、旋转不变性、亮度不变性、噪点不敏感、特征维度小、抗遮挡 |
| 缺点       | 特征维度大、描述子生成过程冗长、无法处理遮挡、噪点相当敏感   | 对边缘不明显，难提取特征                                     |

数据集

COCO AP

VOC数据集

cifiar10

# 图像分类

计算机视觉中关于图像识别有四大类任务：

1. 分类-Classification
2. 定位-Location
3. 检测-Detection
4. 分割-Segmentation

<img src="D:\tool\typora\image\image-20240418105437563.png" alt="image-20240418105437563" style="zoom: 33%;" />

<img src="D:\tool\typora\image\image-20240418173843308.png" alt="image-20240418173843308" style="zoom:33%;" />

类或函数使用

SummaryWritter：是PyTorch中的一个类，用于将条目直接写入事件文件以供TensorBoard使用，将图片直观的展示

```python
from torch.utils.tensorboard import SummaryWriter		#导入

writer = SummaryWriter("logs")              # 将write写在logs(第一个参数)目录下

for i in range(100):						# 输出函数图像
    writer.add_scalar("y=x", i, i)		    # 参数分别为名称，y，x
writer.close()

writer.add_image("ToTensor", tensor_img)	# 输出图像
writer.close()
```

在pytorch环境中输入tensorboard --logdir=logs (--port=6007)查看。logs为文件所在目录



transforms的调用：

```python
tensor_trans = transforms.ToTensor()        # 转换为tensor类型的图片
tensor_img = tensor_trans(img)              # 输入图片，使用ToTensor()函数
```



_call\_:`__call__`方法使得一个类的实例能够像函数一样被调用。当我们将一个类的实例像函数一样调用时（例如，instance(arg1, arg2)）

```python
class Person:
    def __call__(self, name):
        print("hello" + name)
    def hello(self, name):						# 普通定义
        print("good" + name)

person = Person()
person("Wang")									# call调用
person.hello("Zhang")
```

argmax(1)找出每一行最大值所在索引，argmax(0)找出每一列最大值所在索引

网络模型、数据（输入/标注）、损失函数可以使用.cuda()

在使用前加上判断

```python
if torch.cuda.is_available():
```

nvidia-smi查看英伟达显卡信息(报错了前面加感叹号)



利用PIL查看图片的通道：

```python
from PIL import Image     
# 打开图片   
img = Image.open('path_to_your_image.jpg')     
# 获取图片的通道数   
channels = img.mode     
print('Number of channels:', channels)			
# 输出eg:RGB, RGBA
```

```python
import numpy as np  
from PIL import Image  
# 打开图片  
img = Image.open('path_to_your_image.jpg')   
# 将PIL图像转换为numpy数组  
np_img = np.array(img)  
# 查看数组的形状  
print('Shape of the array:', np_img.shape)
# 输出eg:(337, 585, 4) 4通道337×585
```

数据集

![image-20231112211203850](D:\tool\typora\image\image-20231112211203850.png)

## 卷积神经网络基础

​		神经网络的变种目前有，如误差反向传播（Back Propagation，BP）神经网络、概率神经网络、RNN-循环神经网络、DNN-深度神经网络CNN-卷积神经网络（适用于图像识别）、LSTM-时间递归神经网络（适用于语音识别）等。但最简单且原汁原味的神经网络则是多层感知器（Muti-Layer Perception, MLP）。

**MLP神经网络的结构和原理**
 最典型的MLP包括包括三层：输入层、隐层和输出层，MLP神经网络不同层之间是**全连接**的（全连接的意思就是：上一层的任何一个神经元与下一层的所有神经元都有连接）。

![image-20240401162718242](D:\tool\typora\image\image-20240401162718242.png)

由此可知，神经网络主要有三个基本要素：**权重、偏置和激活函数**

### 神经网络的分类

#### DNN(深度神经网络)

​		神经网络是基于感知机的扩展，而DNN可以理解为有很多隐藏层的神经网络。多层神经网络和深度神经网络DNN其实也是指的一个东西，DNN有时也叫做多层感知机（Multi-Layer perceptron,MLP）。

DNN存在的局限：
  参数数量膨胀。由于DNN采用的是全连接的形式，结构中的连接带来了数量级的权值参数，这不仅容易导致过拟   合，也容易造成陷入局部最优。

  局部最优。随着神经网络的加深，优化函数更容易陷入局部最优，且偏离真正的全局最优，对于有限的训练数据，性能甚至不如浅层网络。梯度消失。使用sigmoid激活函数（传递函数），在BP反向传播梯度时，梯度会衰减，随着神经网络层数的增加，衰减累积下，到底层时梯度基本为0。

  无法对时间序列上的变化进行建模。对于样本的时间顺序对于自然语言处理、语音识别、手写体识别等应用非常重要。

#### CNN(卷积神经网络)

​		主要针对DNN存在的参数数量膨胀问题，对于CNN，并不是所有的上下层神经元都能直接相连，而是通过“卷积核”作为中介（部分连接）。同一个卷积核在多有图像内是共享的，图像通过卷积操作仍能保留原先的位置关系。CNN之所以适合图像识别，正式因为CNN模型限制参数个数并挖掘局部结构的这个特点。

<img src="D:\tool\typora\image\image-20231115210655735.png" alt="image-20231115210655735" style="zoom: 50%;" />

<img src="D:\tool\typora\image\image-20240418211816990.png" alt="image-20240418211816990" style="zoom: 50%;" />

#### RNN(循环神经网络)

​		针对CNN中无法对时间序列上的变化进行建模的局限，为了适应对时序数据的处理，出现了RNN。在普通的全连接网络或者CNN中，每层神经元的信号只能向上一层传播，样本的处理在各个时刻独立（这种就是前馈神经网络）。而在RNN中，神经元的输出可以在下一个时间戳直接作用到自身。

​	（t+1）时刻网络的最终结果O(t+1)是该时刻输入和所有历史共同作用的结果，这就达到了对时间序列建模的目的。

​		存在的问题：RNN可以看成一个在时间上传递的神经网络，它的深度是时间的长度，而梯度消失的现象出现时间轴上。



BP——反向传播算法Back Propagation(不属于卷积神经网络)

![image-20240418212337927](D:\tool\typora\image\image-20240418212337927.png)





### 神经网络层

- pytorch 中 tensor（也就是输入输出层）的 通道排序为：`[batch, channel, height, width]`

![image-20240317151901283](D:\tool\typora\image\image-20240317151901283.png)

- model.py ——定义LeNet网络模型
- train.py ——加载数据集并训练，训练集计算loss，测试集计算accuracy，保存训练好的网络参数
- predict.py——得到训练好的网络参数后，用自己找的图像进行分类测试

#### 1.卷积层 (Convolutional Laver)

​		卷积层是深度学习中最重要的层之一，**主要用于提取输入数据的特征**。它通过使用一组可学习的滤波器(也称为卷积核)对输入数据进行卷积操作，从而实现特征的提取。卷积操作可以有效地捕捉到输入数据中的局部特征，并且具有平移不变性，即对于输入数据的不同位置，可以得到相同的特征响应。

​		卷积操作可以捕捉到图像中的局部特征而不受其位置的影响，能够实现平移不变性。

<img src="https://img-blog.csdnimg.cn/a1e1905ac9b04cd2ac539c417c1e42fd.gif" alt="img" style="zoom: 80%;" />

图像经过卷积之后的样子：

<img src="D:\tool\typora\image\image-20240418212144837.png" alt="image-20240418212144837" style="zoom: 50%;" />

经卷积后的输出层尺寸计算公式为：

<img src="D:\tool\typora\image\image-20231115202646479.png" alt="image-20231115202646479" style="zoom:80%;" />

- 输入图片大小 W×W（一般情况下Width=Height）

- 卷积核(Filter)大小 F×F

- 步长 S

- 计算出小数了默认向下取整

- padding的像素数 P             注：2P表示上下(左右)填充的像素之和，可以不相等，即为2P=p1+p2。

  （数据填充的主要目的是确保卷积核能够覆盖输入图像的边缘区域，同时保持输出特征图的大小。这对于在CNN中保留空间信息和有效处理图像边缘信息非常重要。）

#### 2.池化层(Pooling Laver)

​		**池化层主要用于减小特征图的尺寸，并且保留重要的特征信息。**它通过对输入数据的局部区域进行汇聚操作，例如最大池化或平均池化，从而减少特征图的大小。池化操作可以降低模型对输入数据的敏感性，并且减少模型的参数数量，从而降低计算复杂度。

<img src="D:\tool\typora\image\image-20231201170532471.png" alt="image-20231201170532471" style="zoom: 33%;" />

<img src="D:\tool\typora\image\image-20231201170628627.png" alt="image-20231201170628627" style="zoom: 33%;" />

#### 3.全连接层(Fuly Connected Layer)

​		全连接层是神经网络中最常见的层之一，它将前一层的所有神经元与当前层的所有神经元相连接。全连接层的作用是**将前面层提取到的特征进行组合和整合**，从而得到最终的输出结果。全连接层通常用于分类任务，例如图像分类、文本分类等。

​		最后，全连接层将提取的特征映射转化为网络的最终输出。这可以是一个分类标签、回归值或其他任务的结果。

#### 4.多层堆叠

CNN通常由多个卷积和池化层的堆叠组成，以逐渐提取更高级别的特征。深层次的特征可以表示更复杂的模式。

对一张图片进行卷积操作

```python
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2
'''
流程：
1.利用cv2读取灰度图
2.将图片转换成numpy.ndarray格式，且dtype='float32'
3.转成tensor格式，并且reshape成 (batch,channal,h,w)形式
4.卷积，激活函数，池化(注：sobel_kernel 要和img一个shape)
'''
im = cv2.imread("./rose.png", 0) # 读入一张灰度图的图片   466*619
im = np.array(im, dtype='float32')  # 转换为浮点型
# im = im.astype(np.float32)

im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))  # torch.Size([1, 1, 466, 619])

conv1 = nn.Conv2d(1, 1, 3, bias=False) # 定义卷积

sobel_kernel = np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]], dtype='float32') # 定义轮廓检测算子  3*3
sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))   # 适配卷积的输入输出   3*3=1*1*3*3
# conv1.weight.data = torch.from_numpy(sobel_kernel)  # 给卷积的 kernel 赋值,假如不赋值，会自动赋予一个kernel_size=3的卷积核

edge1 = conv1(im)   # 进行卷积
# edge1 = edge1.data.squeeze().numpy() # 将输出转换为图片的格式,(464, 617),<class 'tuple'>，为了方便展示图片
test1 = edge1.data.numpy()
x = F.relu(edge1)   # 加入一层激活函数
test2 = x.data.numpy()
# 最大池化层
# pool1 = nn.MaxPool2d(2, 2)
# x = pool1(x)       # 加入一层池化
# print(x.data)   # torch.Size([1, 1, 232, 308])  tensor

x = x.data.squeeze().numpy()  # (232, 308)  numpy
# print(x.shape)#输出大小

cv2.imshow('img', x)
cv2.waitKey(0)
```



### 误差计算

![image-20231201171030696](D:\tool\typora\image\image-20231201171030696.png)

![image-20231201172921845](D:\tool\typora\image\image-20231201172921845.png)
$$
将y_1写成y_1=w_{11}^{(2)}a_1+w_{21}^{(2)}a_2+w_{31}^{(2)}a_3+b_{1}^{(2)}便于计算
$$
![image-20231201174134901](D:\tool\typora\image\image-20231201174134901.png)

![image-20231201174559904](D:\tool\typora\image\image-20231201174559904.png)

![image-20231201174740775](D:\tool\typora\image\image-20231201174740775.png)
$$
新的w_{11}^{(2)}=旧的w_{11}^{(2)}-设置的学习率*w_{11}^{(2)}的损失梯度\ \ \ \ \ \ \ 	\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
$$


交叉熵损失

![image-20231201173118136](D:\tool\typora\image\image-20231201173118136.png)

权重的更新

![image-20231201175849784](D:\tool\typora\image\image-20231201175849784.png)

## 激活函数

![v2-5b6de25d9929e619784e016466aeb937_b](D:/tool/typora/image/v2-5b6de25d9929e619784e016466aeb937_b.webp)

#### Sigmoid函数：

<img src="D:\tool\typora\image\image-20231115211312376.png" alt="image-20231115211312376" style="zoom:67%;" />

其几何函数如下：

![image-20231115211350429](D:\tool\typora\image\image-20231115211350429.png)

#### Relu函数：

![image-20231115211444003](D:\tool\typora\image\image-20231115211444003.png)

其几何函数如下：

![image-20231115211525649](D:\tool\typora\image\image-20231115211525649.png)

##### Leaky ReLU函数

![image-20240423172337038](D:\tool\typora\image\image-20240423172337038.png)

特点：

- Leaky ReLU函数通过把x的非常小的线性分量给予负输入0.01x来调整负值的零梯度问题。

- Leaky有助于扩大ReLU函数的范围，通常$\alpha$的值为0.01左右。
- Leaky ReLU的函数范围是负无穷到正无穷。

其几何函数如下：

<img src="D:\tool\typora\image\image-20240423172402767.png" alt="image-20240423172402767" style="zoom:33%;" />

#### tanh函数/双曲正切激活函数：

![image-20231115211554491](D:\tool\typora\image\image-20231115211554491.png)

其几何函数如下：

![image-20231115211637145](D:\tool\typora\image\image-20231115211637145.png)

#### ELU函数

![image-20240506165318007](D:/tool/typora/image/image-20240506165318007.png)

α 常见的取值是在 0.1 到 0.3 之间

<img src="D:/tool/typora/image/image-20240506165352198.png" alt="image-20240506165352198" style="zoom: 33%;" />

#### SELU函数

![image-20240506165455600](D:/tool/typora/image/image-20240506165455600.png)

![image-20240506165534172](D:/tool/typora/image/image-20240506165534172.png)

<img src="D:/tool/typora/image/image-20240506165510475.png" alt="image-20240506165510475" style="zoom:33%;" />

#### GELU激活函数：

![image-20240403201909432](D:\tool\typora\image\image-20240403201909432.png)

![image-20240403201847785](D:\tool\typora\image\image-20240403201847785.png)

#### SiLU激活函数

![image-20240506165752768](D:/tool/typora/image/image-20240506165752768.png)

<img src="D:/tool/typora/image/image-20240506165837352.png" alt="image-20240506165837352" style="zoom:50%;" />

## 网络模型

### 1、AlexNet

​		使用两块GPU并行运算

![image-20231115213423403](D:\tool\typora\image\image-20231115213423403.png)

![image-20240115193941077](D:\tool\typora\image\image-20240115193941077.png)

![image-20231202110056282](D:\tool\typora\image\image-20231202110056282.png)

**局部响应归一化**

​		在神经网络中，我们用激活函数将神经元的输出做一个非线性映射，但是tanh和sigmoid这些传统的激活函数的值域都是有范围的，但是ReLU激活函数得到的值域没有一个区间，所以要对ReLU得到的结果进行归一化。也就是Local Response Normalization。局部响应归一化的方法如下面的公式：
$$
b^i_{(x, y)}=\frac{a^i_{(x,y)}}{k+\alpha \sum_{j=max(0,i-n/2)}^{min(N-1, i+n/2)}(a^j_{(x,y)})^2)^\beta }
$$
<img src="D:\tool\typora\image\image-20240418155243608.png" alt="image-20240418155243608" style="zoom:50%;" />

**过拟合**

<img src="D:\tool\typora\image\image-20231202110138753.png" alt="image-20231202110138753" style="zoom: 33%;" />



<img src="D:\tool\typora\image\image-20231115221545690.png" alt="image-20231115221545690" style="zoom:67%;" />

##### model.py

**卷积层提取特征** 和 **全连接层进行分类**

```python
import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 用nn.Sequential()将网络打包成一个模块，精简代码
        self.features = nn.Sequential(   # 卷积层提取图像特征
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True), 									# 直接修改覆盖原值，节省运算内存
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(   # 全连接层对图像分类
            nn.Dropout(p=0.5),			   # Dropout 随机失活神经元，默认比例为0.5
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()
            
	# 前向传播过程
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)	# 展平后再传入全连接层
        x = self.classifier(x)
        return x
        
	# 网络权重初始化，实际上 pytorch 在构建网络时会自动初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):                            # 若是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out',   # 用（何）kaiming_normal_法初始化权重
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)                    # 初始化偏重为0
            elif isinstance(m, nn.Linear):            # 若是全连接层
                nn.init.normal_(m.weight, 0, 0.01)    # 正态分布初始化
                nn.init.constant_(m.bias, 0)          # 初始化偏重为0
```

##### train.py

​		对训练集的预处理，多了随机裁剪和水平翻转这两个步骤。可以起到扩充数据集的作用，增强模型泛化能力。

```python
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),       # 随机裁剪，再缩放成 224×224
                                 transforms.RandomHorizontalFlip(p=0.5),  # 水平方向随机翻转，概率为 0.5, 即一半的概率翻转, 一半的概率不翻转
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
```

​		这次的花分类数据集 并不在 pytorch 的 torchvision.datasets. 中，因此需要用到datasets.ImageFolder()来导入。ImageFolder()返回的对象是一个包含数据集所有图像及对应标签构成的二维元组容器，支持索引和迭代，可作为torch.utils.data.DataLoader的输入。

```python
# 获取图像数据集的路径
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  		# get data root path 返回上上层目录
image_path = data_root + "/data_set/flower_data/"  				 		# flower data_set path

# 导入训练集并进行预处理
train_dataset = datasets.ImageFolder(root=image_path + "/train",		
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# 按batch_size分批次加载训练集
train_loader = torch.utils.data.DataLoader(train_dataset,	# 导入的训练集
                                           batch_size=32, 	# 每批训练的样本数
                                           shuffle=True,	# 是否打乱训练集
                                           num_workers=0)	# 使用线程数，在windows下设置为0
```

为了方便在 predict 时读取信息，将 索引：标签 存入到一个 `json` 文件中

```python
# 字典，类别：索引 {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
# 将 flower_list 中的 key 和 val 调换位置
cla_dict = dict((val, key) for key, val in flower_list.items())

# 将 cla_dict 写入 json 文件中
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)
```

训练过程中需要注意：

- `net.train()`：训练过程中开启 Dropout(或者作为判断是否在训练状态中，此时self.training为true)
- `net.eval()`： 验证过程关闭 Dropout

```python
net = AlexNet(num_classes=5, init_weights=True)  	  # 实例化网络（输出类型为5，初始化权重）
net.to(device)									 	  # 分配网络到指定的设备（GPU/CPU）训练
loss_function = nn.CrossEntropyLoss()			 	  # 交叉熵损失
optimizer = optim.Adam(net.parameters(), lr=0.0002)	  # 优化器（训练参数，学习率）

save_path = './AlexNet.pth'
best_acc = 0.0

for epoch in range(10):
    ##################################### train #####################################
    net.train()     					# 训练过程中开启 Dropout
    running_loss = 0.0					# 每个 epoch 都会对 running_loss  清零
    time_start = time.perf_counter()	# 对训练一个 epoch 计时
    
    for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
        images, labels = data   # 获取训练集的图像和标签
        optimizer.zero_grad()	# 清除历史梯度
        
        outputs = net(images.to(device))				 # 正向传播
        loss = loss_function(outputs, labels.to(device)) # 计算损失
        loss.backward()								     # 反向传播
        optimizer.step()								 # 优化器更新参数
        running_loss += loss.item()
        
        # 打印训练进度（使训练过程可视化）
        rate = (step + 1) / len(train_loader)           # 当前进度 = 当前step / 训练一轮epoch所需总step
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print('%f s' % (time.perf_counter()-time_start))

    #################################### validate ##################################
    net.eval()    # 验证过程中关闭 Dropout
    acc = 0.0  
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
            acc += (predict_y == val_labels.to(device)).sum().item()    
        val_accurate = acc / val_num
        
        # 保存准确率最高的那次网络参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')
```



##### predict.py

```python
import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

# 预处理
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
img = Image.open("蒲公英.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))

# 关闭 Dropout
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))     # 将输出压缩，即压缩掉 batch 这个维度
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.show()
```



### 2、VggNet

![image-20231202110713634](D:\tool\typora\image\image-20231202110713634.png)

通过多个小卷积核代替一个大卷积核可以减少训练所需参数

##### 感受野

​	在卷积神经网络中，决定某一层输出结果中一个元素所对应的输入层的区域大小，被称作感受野(receptive field)。输出特征矩阵(feature map)上的一个单元对应输入层上的区域大小。

![image-20231202111502430](D:\tool\typora\image\image-20231202111502430.png)

![image-20231202111941938](D:\tool\typora\image\image-20231202111941938.png)

##### model.py

<img src="D:\tool\typora\image\image-20240317160657322.png" alt="image-20240317160657322" style="zoom:50%;" />

VGG网络有 VGG-13、VGG-16等多种网络结构，能不能将这几种结构统一成一个模型呢？

```python
# vgg网络模型配置列表，数字表示卷积核个数，'M'表示最大池化层
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],											# 模型A
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],									# 模型B
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],					# 模型D
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], 	# 模型E
}

# 卷积层提取特征
def make_features(cfg: list): # 传入的是具体某个模型的参数列表
    layers = []
    in_channels = 3		# 输入的原始图像(rgb三通道)
    for v in cfg:
        # 最大池化层
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 卷积层
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)  # 单星号(*)将参数以元组(tuple)的形式导入


def vgg(model_name="vgg16", **kwargs):  # 双星号(**)将参数以字典的形式导入
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model
```



### 3、GoogleNet

![image-20231202112028915](D:\tool\typora\image\image-20231202112028915.png)

##### inception结构

GoogLeNet 提出了一种并联结构，下图是论文中提出的inception原始结构，将特征矩阵**同时输入到多个分支**进行处理，并将输出的特征矩阵**按深度进行拼接**(channel)，得到最终输出。

- inception的作用：增加网络深度和宽度的同时减少参数。

![image-20240317161103024](D:\tool\typora\image\image-20240317161103024.png)

inception + 降维

在 inception 的基础上，还可以加上降维功能的结构，如下图所示，在原始 inception 结构的基础上，在分支2，3，4上加入了**卷积核大小为1x1的卷积层**，目的是为了降维（减小深度），减少模型训练参数，减少计算量。

![image-20240317164330734](D:\tool\typora\image\image-20240317164330734.png)

网络模型：\![googlenet](D:\tool\typora\image\googlenet.png) (crtl+左键查看)

##### 辅助分类器

GooleNet有三个输出层（其中有两个辅助分类器）

辅助分类器的两个分支有什么用呢？

- 作用一：可以把他看做inception网络中的一个小细节，它确保了即便是隐藏单元和中间层也参与了特征计算，他们也能预测图片的类别，他在inception网络中起到一种调整的效果，并且能防止网络发生过拟合。
- 作用二：给定深度相对较大的网络，有效传播梯度反向通过所有层的能力是一个问题。通过将辅助分类器添加到这些中间层，可以期望较低阶段分类器的判别力。在训练期间，它们的损失以折扣权重（辅助分类器损失的权重是0.3）加到网络的整个损失上。

下面是原论文中给出的网络参数列表:

![image-20240317161432444](D:\tool\typora\image\image-20240317161432444.png)

##### train.py

GoogLeNet的网络输出 loss 有三个部分，分别是主干输出loss、两个辅助分类器输出loss（权重0.3）

```python
logits, aux_logits2, aux_logits1 = net(images.to(device))
loss0 = loss_function(logits, labels.to(device))
loss1 = loss_function(aux_logits1, labels.to(device))
loss2 = loss_function(aux_logits2, labels.to(device))
loss = loss0 + loss1 * 0.3 + loss2 * 0.3
```



### 4、ResNet

<img src="D:\tool\typora\image\image-20240313095230121.png" alt="image-20240313095230121" style="zoom: 50%;" />

![image-20240317161848317](D:\tool\typora\image\image-20240317161848317.png)

一般我们会觉得网络越深，特征信息越丰富，模型效果应该越好。但是实验证明，当网络堆叠到一定深度时，会出现两个问题：

1、梯度消失或梯度爆炸

关于梯度消失和梯度爆炸，其实看名字理解最好：

​		若每一层的误差梯度小于1，反向传播时，网络越深，梯度越趋近于0。反之，若每一层的误差梯度大于1，反向传播时，网路越深，梯度越来越大

2、退化问题(degradation problem)：在解决了梯度消失、爆炸问题后，仍然存在深层网络的效果可能比浅层网络差的现象

​		总结就是，**当网络堆叠到一定深度时，反而会出现深层网络比浅层网络效果差的情况。**对于退化问题，ResNet论文提出了 **residual结构**（**残差结构**）来减轻退化问题。为了解决深层网络中的退化问题，可以人为地让神经网络某些层跳过下一层神经元的连接，隔层相连，弱化每层之间的强联系。这种神经网络被称为 **残差网络** (**ResNets**)。

![image-20240318193107366](D:\tool\typora\image\image-20240318193107366.png)

##### residual（残差）结构

① 提出residual模块

![image-20240318184838519](D:\tool\typora\image\image-20240318184838519.png)

​		左边针对与网络层数较少的的网络所使用的的残差结构（34），右边主要针对与网络层数较多的网络所使用的结构（50/101/152）。**主分支与侧分支的输出矩阵特征shape必须相同（高宽通道数相同）。**右边在输入和输出都加上了1X1的卷积层，降维再升维（256→64→256），保证shape相同的情况下**减少参数。**使用的残差结构越多，减少的参数就越多。

实线残差结构和虚线残差结构区别：

​	实线残差结构：输入输出shape相同，可以直接相加。

​	虚线残差结构：输入输出矩阵shape不同，输入：【56,56,64】输出：【28,28,128】，只有将虚线残差结构的输出输入到实线残差结构，才能保证输入输出矩阵的shape相同。

![image-20240318184654726](D:\tool\typora\image\image-20240318184654726.png)

##### Batch Normalization处理

② 使用Batch Normalization加速训练（丢弃dropout）

1. 目的：使我们的一批（Batch）数据所对应的数据的feature map（所对应的特征矩阵）每一个通道所对应的维度满足均值为0，方差为1的分布规律。

​	**能够加速网络的收敛并提高准确率**

2. 在图像预处理的过程通常会对图像进行标准化处理。对于Conv1来说输入就是满足某一分布的特征矩阵，但对Conv2来说输入的feature map就不一定满足某一分布规律（整个训练样本集所对应的feature map的数据要满足分布规律）。BN的目的就是使Feature map满足均值为0，方差为1的分布规律。

   ![image-20240318185146830](D:\tool\typora\image\image-20240318185146830.png)

![image-20240318185401217](D:\tool\typora\image\image-20240318185401217.png)

γ，β是通过反向传播得到的学习率，初始值分别为1,0，使效果更好，ε防止分母为0。

假设我们输入的x是RGB三通道的彩色图像，那么这里的d就是输入图像的channels即d=3,x = (x^(1), x^(2), x^(3))，其中x^(1)就代表我们的R通道所对应的特征矩阵，依此类推。标准化处理也就是分别对我们的R通道，G通道，B通道进行处理。

**让feature map满足某一分布规律，理论上是指整个训练样本集所对应feature map的数据要满足分布规律**。也就是说要计算出整个训练集的feature map然后在进行标准化处理，对于一个大型的数据集明显是不可能的，所以论文中说的是Batch Normalization，也就是我们计算一个Batch数据的feature map，然后在进行标准化（batch越大越接近整个数据集的分布，效果越好）

![image-20240318185809712](D:\tool\typora\image\image-20240318185809712.png)

将特征矩阵按通道生成向量，分别计算均值，方差（正向传播中得到）。根据标准差计算公式计算每个通道的值。 

使用pytorch进行实验：

![image-20240327144608754](D:\tool\typora\image\image-20240327144608754.png)

3. 使用BN时需要注意的问题：

（1）训练时要将training参数设置为True，在验证时将training参数设置为False。在Pytorch中可通过创建模型的model.train()和model.eval()方法控制。

（2）Batch size尽可能设置大一点，设置小后表现可能很糟糕，设置越大求的均值和方差越接近整个训练集的均值和方差。

（3）建议将bn层放在卷积层和激活层之间，且卷积层不要使用偏置，使用偏置后求出的结果也是一样的。

##### 迁移学习

1. 优势：

① 能够快速的训练出一个理想的结果（减少训练时间）

② 当数据集较小时也能训练出理想的效果（使用别人之前预训练好的模型参数去训练比较小的数据集）

注：使用别人预训练模型参数时，要注意别人的预处理方式

2. 迁移学习简介

   ​		对于浅层卷积层所学习的一些角点信息/纹理信息等通用信息，在其他网络中仍然适用。所以将学习好的网络的浅层网络的一些参数迁移到新的网络当中。这样新的网络就也拥有了识别底层通用特征的能力。这样新的网络就能更快速的学习新的数据集的高维特征。

   ![image-20240318190417593](D:\tool\typora\image\image-20240318190417593.png)

3. 常见的迁移学习方式

①载入权重后训练所有参数（最后一层无法载入）→第三种方法

②载入权重之后只训练最后几层参数

③载入权重后在原网络基础上再添加一层全连接层，仅训练最后一个全连接层。

②③更快速，①效果更好

#### ResNetX网络

##### Group Convolution

​	ResNet网络的小幅提升，下图对比了卷积和组卷积，采用组卷积可以减少计算参数。

![image-20240318191345692](D:\tool\typora\image\image-20240318191345692.png)

![image-20240318191756589](D:\tool\typora\image\image-20240318191756589.png)

### 5、MobileNet

####  MobileNet v1网络详解

传统卷积神经网络，内存需求大、运算量大导致无法在移动设备以及嵌入式设备上运行；所以提出了MobileNet应用到实际生活中。2017年提出的轻量级CNN网络MobileNet专注于移动端或嵌入式设备，在准确率小幅降低的前提下减少了模型参数和运算量。

 MobileNet v1网络中主要有两个两点：

1. Depthwise Convolution(简称DW卷积，减少参数和运算量)
2. 增加超参数α(控制卷积层、核数量)、β(控制输入图像尺寸)，这两个超参数是人为设定的

![image-20240318203316977](D:\tool\typora\image\image-20240318203316977.png)

##### Depthwise Separable(DW)卷积

由两部分组成：DW卷积(Depthwise Conv)和PW卷积(Pointwise Conv)

传统卷积和DW卷积和PW卷积的区别：

![image-20240318203405331](D:\tool\typora\image\image-20240318203405331.png)

![image-20240318203623965](D:\tool\typora\image\image-20240318203623965.png)

使用DW卷积后输出特征矩阵的channel是与输入特征矩阵的channel相等的，如果想改变/自定义输出特征矩阵的channel，那只需要在DW卷积后接上一个PW卷积即可。PW卷积就是普通的卷积而已（只不过卷积核大小为1）

MobileNet v1网络

​	table6和table7对应的是MobileNet的两个超参数α(Width Multiplier)、β(Resolution Multiplier)：

1. α代表的是卷积核个数的倍率因子，控制卷积核个数。比如table6第一行当α取1.0时，准确率为70.6%
2. β代表的是分辨率的倍率因子，输入不同尺寸的图像会有不同的准确率。比如table7第一行输入图像为224×224，准确率为70.6%

![image-20240318204244552](D:\tool\typora\image\image-20240318204244552.png)

MobileNet v1网络存在的问题：depthwise部分的卷积核容易废掉，即卷积核参数大部分为0

#### MobileNet v2网络详解

2018年提出的MobileNet v2比v1版本更好。网络主要两个两点：

1. Inverted Residual(倒残差结构)
2. Linear Bottlenecks

##### Inverted Residual(倒残差)结构

​		残差结构就是两头大，中间小。先用1×1卷积降维，然后使用3×3卷积，再使用1×1卷积升维；(使用Relu激活函数)

​		倒残差结构就是两头小，中间大。先用1×1卷积升维，再使用3×3DW卷积，最后使用1×1降维；(使用Relu6激活函数)![image-20240318204716435](D:\tool\typora\image\image-20240318204716435.png)

![image-20240318204957458](D:\tool\typora\image\image-20240318204957458.png)

​		针对倒残差结构中最后1×1卷积层，使用了线性激活函数，而不是relu激活函数。因为relu激活函数对于低纬度特征信息会造成大量损失。Figure1给出了example，Input经过random matrix T、RuLU、T的逆矩阵，在低纬度的情况下还原的很差。而倒残差结构是“两头小，中间大”，所以在输出的时候就是高维到低维度的情况，所以不建议使用ReLU激活函数，这里使用了线性激活函数来替代。

![image-20240318205348041](D:\tool\typora\image\image-20240318205348041.png)

何时有shortcut（捷径）

![image-20240318205553184](D:\tool\typora\image\image-20240318205553184.png)

MobileNet v2网络框架

![image-20240318205724106](D:\tool\typora\image\image-20240318205724106.png)

1. s代表的是步距，有的bottleneck需要重复两次，两次中只有第一次s是为2的，这个根据Input的特征图大小也能判断出来
2. t代表是扩展因子。这个在上图中的表格也可以看到
3. 下图表格中最后一行k代表的就是分类类别个数，比如使用Imagenet数据集，对应就是该数据集的类别个数

MobileNet V3网络详解

1. 更新了Block(在原来v2的bottleneck基础上改成了bneck)
2. 使用NAS搜索参数(Neural Architecture Search)，不细说
3. 重新设计耗时层结构

block具体更新的点![image-20240318210732637](D:\tool\typora\image\image-20240318210732637.png)

上图中下半部分中的Mobilenet V3 block中的NL就代表了非线性激活函数。但不同层使用的是不同的激活函数

SE模块详解

​		如上图下半部分所示，就是将特征图进行池化，假设 7*7*32的特征图，池化成1*1*32的向量，然后经过第一个全连接层缩减得到1*1*8，然后经过第二个全连接层复原得到1*1*32。

​		如何复原成原来的特征图大小。下图解释是，得到了0.5、0.6，然后用原来的特征图和这个数据相乘，也就是把0.5、0.6当成了channel对应的权重（这里可以理解为对特征矩阵每一个channel分析出一个权重关系，比较重要的channel就赋予一个比较大的权重，不是很重要的channel就赋予一个比较小的权重）。

![image-20240318210944838](D:\tool\typora\image\image-20240318210944838.png)

重新设计耗时层结构

1. 减少第一个卷积层的卷积核个数
2. 精简last stage![image-20240318211414832](D:\tool\typora\image\image-20240318211414832.png)

重新设计激活函数

![image-20240318211619207](D:\tool\typora\image\image-20240318211619207.png)

MobileNet v3网络结构

![image-20240318211939938](D:\tool\typora\image\image-20240318211939938.png)

exp_size——代表升维的维度

#out——代表降维后的维度

SE——√ 代表使用注意力机制

### 6、ShuffleNet

#### ShuffleNetv1

ShuffleNet Unit中全是GConv和DWConv

![image-20240328205146086](D:\tool\typora\image\image-20240328205146086.png)

在ResNet中，1×1卷积占用了绝大多数计算量，在此换成了Group卷积

![image-20240328205434185](D:\tool\typora\image\image-20240328205434185.png)

中间对应stride=1，右边对应stride=2

![image-20240328205704079](D:\tool\typora\image\image-20240328205704079.png)

左下是一个残差结构；g=3时最典型

#### ShuffleNetv2

![image-20240328210454898](D:\tool\typora\image\image-20240328210454898.png)

影响计算速度的几个因素

![image-20240328212137246](D:\tool\typora\image\image-20240328212137246.png)

G4包括开启卷积操作，相加操作，偏置，激活函数等（对每一个元素进行操作）

v2结合上面四条准则对v1的block进行优化

![image-20240328212602954](D:\tool\typora\image\image-20240328212602954.png)

channel Spilit是按照一半一半进行划分的，DW卷积输入输出channel不变

![image-20240328212858423](D:\tool\typora\image\image-20240328212858423.png)

v2和v1的网络框架相似，就多了1×1的卷积



### 7、EfficientNet

#### EfficientNet v1

这篇论文主要是用NAS（Neural Architecture Search）技术来搜索网络的图像输入分辨率r，网络的深度depth以及channel的宽度width三个参数的合理化配置。

![image-20240328213836760](D:\tool\typora\image\image-20240328213836760.png)

![image-20240328214016447](D:\tool\typora\image\image-20240328214016447.png)

EfficientNet-B0：整个网络框架由一系列Stage组成，F_i表示对应Stage的运算操作，L_i表示在该Stage中重复F_i 的次数：

![image-20240326195807362](D:\tool\typora\image\image-20240326195807362.png)

接着作者又提出了一个混合缩放方法 ( compound scaling method) 在这个方法中使用了一个混合因子ϕ \phiϕ去统一的缩放width，depth，resolution参数:

![image-20240401103226918](D:\tool\typora\image\image-20240401103226918.png)

FLOPs(理论计算量)与depth的关系是:当depth翻倍，FLOPs也翻倍。

FLOPs与width的关系是:当width翻倍(即channal翻倍)，FLOPs会翻4倍，因为卷积层的FLOPs约等于featurew X
featurep x feature。x kernelx kernel}x kernemumber(假设输入输出特征矩阵的高宽不变)，当width翻倍，输入特征矩阵的channels (feature。）和输出特征矩阵的channels或卷积核的个数(kernemumber）都会翻倍，所以FLOPs会翻4倍。

FLOPs与resolution的关系是:当resolution翻倍，FLOPs也会翻4倍，和上面类似因为特征矩阵的宽度feature.和特征矩阵的高度featureh都会翻倍。

所以总的FLOPs倍率可以用近似用(a·β²·y²)^φ来表示，当限制α·β²·v²≈2时，对于任意一个b而言FLOPs相当增加了2^φ倍。

##### MBConv结构

`MBConv`其实就是MobileNetV3网络中的InvertedResidualBlock，但也有些许区别。一个是采用的激活函数不一样（EfficientNet的MBConv中使用的都是Swish激活函数），另一个是在每个MBConv中都加入了SE（Squeeze-and-Excitation）模块。

![image-20240329094502113](D:\tool\typora\image\image-20240329094502113.png)

BN、Swish为激活函数

##### SE模块

![image-20240329094752022](D:\tool\typora\image\image-20240329094752022.png)

![image-20240329094919710](D:\tool\typora\image\image-20240329094919710.png)

`drop_connect_rate:`	是在MBConv结构中dropout层使用的drop_rate，在官方keras模块的实现中MBConV结构的 drop_rate定从0递增到drop_connect_rate的(具体实现可以看下官方源码，注意，在源码实现中只有使用shortcut的的候才有Dropout层)。还需要注意的是，这里的Dropout层是Stochastic Depth，即会随机丢掉整个block的主分支(只剩捷径分支，相当于直接跳过了这个block)也可以理解为减少了网络的深度。具体可参考Deep Networks with Stochastic Depth这篇文章。
`dropout_rate:`	是最后一个全连接层前的dropout层(在stage9的Pooling与FC之间)的dropout_rate。

调整网络的宽度就是调整所采用卷积核的个数

EfficientNet v1中存在的问题：

1、训练图像的尺寸很大时，训练速度非常慢。

2、在网络浅层中使用Depthwise convolutions速度会很慢。

![image-20240401114216958](D:\tool\typora\image\image-20240401114216958.png)

3、同等的放大每个stage是次优的。

#### EfficientNet v2

这篇文章做出的三个贡献:

1. 引入新的网络(EfficientNet V2)，该网络在训练速度以及参数数量上都优于先前的一些网络。

2. 提出了改进的渐进学习方法，该方法会根据训练图像的尺寸动态调节正则方法(例如dropout、 data augmentation和mixup )。通过实验展示了该方法不仅能够提升训练速度，同时还能提升准确率。

3. 通过实验与先前的一些网络相比，训练速度提升11倍，参数数量减少为1/6.8。

NAS 搜索

​	这里采用的是trainning-aware NAS framework，搜索工作主要还是基于之前的Mnasnetl以及EficientNet。但是这次的优化目标联合了accuracy、 parameter efficiency以及trainning efficiency三个维度。

另外，作者通过以下方法来减小搜索空间的范围：

- 移除不需要的搜索选项，例如pooling skip操作（因为在EfficientNet中并没有使用到）
- 重用EfficientNet中搜索的channel sizes（需进一步补充）

**EfficientNetV2网络框架**

​		表4展示了作者使用NAS搜索得到的EfficientNetV2-S模型框架（注意，在源码中Stage6的输出Channels是等于256并不是表格中的272，Stage7的输出Channels是1280并不是表格中的1792，后续论文的版本会修正过来）。相比与EfficientNetV1，主要有以下不同：

- 第一个不同点在于EfficientNetV2中除了使用到NBConv模块外，还使用了Fused -NBConv模块（主要是在网络浅层中使用)。
- 第二个不同点是EfficientNetV2会使用较小的expansion ratio (MBConv中第一个expand convl×1或者Fused -MBConv中第一个expand conv3×3）比如4，在EfficientNetV1中基本都是6 。这样的好处是能够减少内存访问开销。
- 第三个不同点是EficientNetV2中更偏向使用更小(3×3 )的kernel_size，在EfficientNetV1中使用了很多5x5的kernel_size。通过下表可以看到使用的kernel_size全是3x3的，由于3×3的感受野是要比5×5小的，所以需要堆叠更多的层结构以增加感受野。
- 最后一个不同点是移除了EfficientNetV1中最后一个步距为1的stage(就是EficientNetV/1中的stage8，可能是因为它的参数数量过多并且内存访问开销过大（由于网络是通过NAS搜索出来的，所有这里也是作者的猜测)。![image-20240401111144454](D:\tool\typora\image\image-20240401111144454.png)

`Fused-MBConv`  模块上面再讲EfficientNet V1存在问题章节有讲到过，模块名称后跟的1，4表示expansion ratio，k3×3表示kenel size为 3×3。下图为结构图SE结构的（原论文图中有SE)。注意当stride=1且输入输出Channels相等时才有shortcut连接。还需要注意的是，当有shortcut连接时才有Dropout层，而且这里的Dropout层是Stochastic Depth，即会随机丢掉整个block的主分支（只剩捷径分支，相当于直接跳过了这个block)也可以理解为减少了网络的深度。具体可参考Deep Networks with Stochastic Depth这篇文章。

![image-20240401111437991](D:\tool\typora\image\image-20240401111437991.png)

`MBConv` 模块和EficientNet V1中是一样的，其中模块名称后跟的4，6表示expansion ratio，SE0.25表示使用了SE模块，0.25表示SE模块中第一个全连接层的节点个数是输入该NBConv模块特征矩阵channels的¼。注意当stride=1且输入输出Channels相等时才有shortcut连接。同样这里的Dropout层是Stochastic Depth 。

![image-20240401111948253](D:\tool\typora\image\image-20240401111948253.png)

注意每个Stage中会重复堆叠Operator模块多次，只有第一个Opertator模块的步距是按照表格中Stride来设置的，其他的默认都是1。 #Channels表示该Stage输出的特征矩阵的Channels，#Layers表示该Stage重复堆叠Operator的次数。

##### Stochastic Depth 

![image-20240401115104224](D:\tool\typora\image\image-20240401115104224.png)

##### Progressive Learning 渐进学习策略

![image-20240401115854484](D:\tool\typora\image\image-20240401115854484.png)

![image-20240401115910902](D:\tool\typora\image\image-20240401115910902.png)

GRU  LSTM

### Transformer基础

​		https://blog.csdn.net/qq_37541097/article/details/117691873

​		Transformer是2017年Google在Computation and Language上发表的，当时主要是针对自然语言处理领域提出的（之前的RNN模型记忆长度有限且无法并行化，只有计算完t_i 时刻后的数据才能计算$t_{(i+1)}$时刻的数据，但Transformer都可以做到）。在这篇文章中作者提出了Self-Attention的概念，然后在此基础上提出Multi-Head Attention。

![image-20240401182125090](D:\tool\typora\image\image-20240401182125090.png)

#### Self-Attention

​		假设输入的序列长度为2，输入就两个节点$x_1 ,x_2$，然后通过Input Embedding也就是图中的 $f(x)$ 将输入映射到$a_1 , a_2$。紧接着分别将$a_1 , a_2$分别通过三个变换矩阵$W_q,W_k,W_v$，(这三个参数是可训练的，是共享的)得到对应的$q^i ,k^i , v^i$ (这里在源码中是直接使用全连接层实现的，这里为了方便理解，忽略偏执)。

![image-20240401172415329](D:\tool\typora\image\image-20240401172415329.png)

其中

- **q**代表query查询，后续会去和每一个k进行匹配

- **k**代表key键，后续会被每个q匹配
- **v**代表value，从a中提取得到的信息
- 后续q和k匹配的过程可以理解成计算两者的相关性，相关性越大对应v的权重也就越大

$$
Q = (q^1, q^2)^T
$$

$$
K = (k^1, k^2)^T
$$

$$
V = (v^1, v^2)^T
$$

​		接着先拿$q^1$和每个$k$进行match，点乘操作，接着除以$√d$得到对应的$α$，其中**d**代表向量  $ k^i$的长度，在本示例中等于2，除以$√d$的原因在论文中的解释是“进行点乘后的数值很大，导致通过softmax后梯度变的很小”，所以通过除以$√d$来进行缩放。

![image-20240401173106059](D:\tool\typora\image\image-20240401173106059.png)

​		接着对每一行即($a_{(1,1)}$ , $a_{(1,2)}$ )和$(a_{(2,1)}$ , $a_{(2,2)}$ )分别进行softmax处理，然后分别乘以V对应的权重

![image-20240401173300152](D:\tool\typora\image\image-20240401173300152.png)

总结下来就是论文中的一个公式：

![image-20240401172521527](D:\tool\typora\image\image-20240401172521527.png)

#### Multi-Head Attention

​		中文名称多头注意力机制，每个头关注的侧重点不同。

​		首先还是和Self-Attention模块一样将$a_i$分别通过$W_q,W_k,W_v$得到对应的$q^i ,k^i , v^i$，然后再根据使用的head的数目h进一步把得到的$q^i ,k^i , v^i$均分成h份。比如下图中假设h=2然后$q^1$拆分成$q^{1,1}$和$q^{1,2}$，那么$q^{1,1}$就属于head1，$q^{1,2}$属于head2。

<img src="D:\tool\typora\image\image-20240401173503321.png" alt="image-20240401173503321" style="zoom: 67%;" />

一些代码就是简单的进行均分

<img src="D:\tool\typora\image\image-20240401174813936.png" alt="image-20240401174813936" style="zoom:67%;" />

得到的公式为

![image-20240401175206453](D:\tool\typora\image\image-20240401175206453.png)

![image-20240401175308441](D:\tool\typora\image\image-20240401175308441.png)

​	接着将每个head得到的结果进行concat拼接，比如下图中$b_{1,1}$ ($head_1$得到的$b_1$)和$b_{1,2}$ ($head_2$得到的$b_1$）拼接在一起， $b_{2,1}$ ($head_1$得到的$b_2$)和$b_{2,2}$ ($head_2$得到的$b_2$）

![image-20240401175537874](D:\tool\typora\image\image-20240401175537874.png)

接着将拼接后的结果通过$W^O$（可学习的参数）进行融合，如下图所示，融合后得到最终的结果$b_1, b_2$

![image-20240401175719333](D:\tool\typora\image\image-20240401175719333.png)总结下来就是论文中的两个公式：

![image-20240401175741984](D:\tool\typora\image\image-20240401175741984.png)

转换成矩阵相乘的形式表达

![image-20240522204908349](D:/tool/typora/image/image-20240522204908349.png)



#### 其他

Self-Attention与Multi-Head Attention计算量对比:

​	两者的计算量其实差不多,两者FLOPs的差异只是在最后的$W^O$上

##### Positional Encoding位置编码

​	Self-Attention和Multi-Head Attention模块，在计算中是没有考虑到位置信息的。输入的信息互换行是没有区别的。为了引入位置信息，在原论文中引入了位置编码`positional encodings`。

​	如下图所示，位置编码是直接加在输入的$a ={\{a_1,...,a_n}\}$中的，即$pe ={\{pe_1,...,pe_n}\}$和$a ={\{a_1,...,a_n}\}$拥有相同的维度大小。关于位置编码在原论文中有提出两种方案，一种是原论文中使用的固定编码，即论文中给出的`sine and cosine functions`方法，按照该方法可计算出位置编码；另一种是可训练的位置编码，作者说尝试了两种方法发现结果差不多（但在ViT论文中使用的是可训练的位置编码）。

<img src="D:\tool\typora\image\image-20240401180436274.png" alt="image-20240401180436274" style="zoom: 50%;" />

![image-20240522205724449](D:/tool/typora/image/image-20240522205724449.png)

**解码器**

编码器首先处理输入序列。然后，顶部编码器的输出被转换为一组注意力向量 K 和 V。

<img src="D:/tool/typora/image/transformer_decoding_1.gif" alt="transformer_decoding_1" style="zoom:67%;" />

指示变压器解码器已完成其输出。每个步骤的输出在下一个时间步骤中被馈送到底部解码器，并且解码器像编码器一样冒泡其解码结果。

<img src="D:/tool/typora/image/transformer_decoding_2.gif" alt="transformer_decoding_2" style="zoom:67%;" />

在解码器中，自注意力层只允许关注输出序列中较早的位置，即考虑前面单词的作用。这是通过在自注意力计算中的 softmax 步骤之前屏蔽未来位置（将它们设置为 `-inf` ）来完成的。

##### 超参对比

关于Transformer中的一些超参数的实验对比可以参考原论文的表3，如下图所示。其中：

- N表示重复堆叠Transformer Block的次数
- $d_model$表示Multi-Head Self-Attention输入输出的token维度(向量长度)。 
- $d_ff$表示在MLP (feed forward)中隐层的节点个数
- h表示Multi-Head Self-Attention中head的个数
- $d_k , d_y$表示Multi-Head Self-Attention中每个head的key (K)以及query (Q)的维度。
- $P_{drop}$表示dropout层的drop_rate

![image-20240401181016820](D:\tool\typora\image\image-20240401181016820.png)

### 8、Version Transformer

​		在这篇文章中，作者主要拿ResNet、ViT（纯Transformer模型）以及Hybrid（卷积和Transformer混合模型）三个模型进行比较，所以本博文除了讲ViT模型外还会简单聊聊Hybrid模型。

##### 模型架构

下图是原论文中给出的关于Vision Transformer(ViT)的模型框架。简单而言，模型由三个模块组成：

- Linear Projection of Flattened Patches(`Embedding层`)扁平化面片的线性投影

- Transformer Encoder
- MLP Head（最终用于分类的层结构）

输入的图像大小是固定的

![image-20240401210233610](D:\tool\typora\image\image-20240401210233610.png)

#####  Embedding层

​		对于图像数据而言，其数据格式为[H, W, C]是三维矩阵明显不是Transformer想要的。所以需要先通过一个Embedding层来对数据做个变换。

​		对于标准的Transformer模块，要求输入的是token（向量）序列，即二维矩阵[num_token, token_dim]，如下图，token0-9对应的都是向量，以ViT-B/16为例，将输入图片(224x224)按照16x16大小的Patch进行划分，会得到196个Patches，每个Patche数据shape为[16, 16, 3]通过映射得到一个长度为768的向量（后面都直接称为token）。`[16, 16, 3] -> [768]`![image-20240401210535392](D:\tool\typora\image\image-20240401210535392.png)

​		**在代码实现中，直接通过一个卷积层来实现**。 以ViT-B/16为例，直接使用一个卷积核大小为16x16，步距为16，卷积核个数为768的卷积来实现。通过卷积`[224, 224, 3] -> [14, 14, 768]`，然后把H以及W两个维度展平即可`[14, 14, 768] -> [196, 768]`，此时正好变成了一个二维矩阵。

​		**在输入Transformer Encoder之前注意需要加上[class]token以及Position Embedding。**参考BERT，这个[class]token是一个可训练的参数，数据格式和其他token一样都是一个向量；以ViT-B/16为例，就是一个长度为768的向量，与之前从图片中生成的tokens拼接在一起， `Cat([1, 768], [196, 768]) -> [197, 768]`。关于Position Embedding就是之前Transformer中讲到的Positional Encoding，这里的Position Embedding采用的是一个可训练的参数（`1D Pos. Emb.`），是直接叠加在tokens上的（add），所以shape要一样；以ViT-B/16为例，刚刚拼接[class]token后shape是`[197, 768]`，那么这里的Position Embedding的shape也是`[197, 768]`。

​		对于Position Embedding作者也有做一系列对比试验，在源码中默认使用的是`1D Pos. Emb.`，对比不使用Position Embedding准确率提升了大概3个点，和`2D Pos. Emb.`比起来没太大差别。

##### Transformer Encoder

​		Transformer Encoder其实就是重复堆叠Encoder Block L次，主要由以下几部分组成：

- Layer Norm，这种Normalization方法主要是针对NLP领域提出的，这里是对每个token进行Norm处理

- Multi-Head Attention

- Dropout/DropPath，在原论文的代码中是直接使用的Dropout层，在但rwightman实现的代码中使用的是DropPath（stochastic depth），可能后者会更好一点。

- MLP Block，如图右侧所示，就是全连接+GELU激活函数+Dropout组成也非常简单，需要注意的是第一个全连接层会把输入节点个数翻4倍`[197, 768] -> [197, 3072]`，第二个全连接层会还原回原节点个数`[197, 3072] -> [197, 768]`

  <img src="D:\tool\typora\image\image-20240403154156699.png" alt="image-20240403154156699" style="zoom: 50%;" />

##### MLP Head

​	上面通过Transformer Encoder后输出的shape和输入的shape是保持不变的，以ViT-B/16为例，输入的是`[197, 768]`输出的还是`[197, 768]`。注意，在Transformer Encoder后其实还有一个Layer Norm没有画出来。这里我们只是需要分类的信息，所以我们只需要提取出[class]token生成的对应结果就行，即`[197, 768]`中抽取出[class]token对应的`[1, 768]`。接着我们通过MLP Head得到我们最终的分类结果。MLP Head原论文中说在训练ImageNet21K时是由`Linear`+`tanh激活函数`+`Linear`组成。但是迁移到ImageNet1K上或者你自己的数据上时，只用一个Linear即可。

<img src="D:\tool\typora\image\image-20240403154503293.png" alt="image-20240403154503293" style="zoom: 33%;" />

[网络结构](D:\tool\typora\image\image-20240403154616981.png)

![image-20240403154616981](D:\tool\typora\image\image-20240403154616981.png)

不同大小的ViT模型

<img src="D:/tool/typora/image/image-20240523160202615.png" alt="image-20240523160202615" style="zoom: 33%;" />

##### *Hybrid模型

![image-20240403154828150](D:\tool\typora\image\image-20240403154828150.png)

### 9、Swin Transformer

Swin Transformer和Vision Transformer对比：

![image-20240403185555873](D:\tool\typora\image\image-20240403185555873.png)

- Swin Transformer使用了类似卷积神经网络中的层次化构建方法（Hierarchical feature maps），比如特征图尺寸中有对图像下采样4倍的，8倍的以及16倍的，这样的backbone有助于在此基础上构建目标检测，实例分割等任务。而在之前的Vision Transformer中是一开始就直接下采样16倍，后面的特征图也是维持这个下采样率不变。
- 在Swin Transformer中使用了Windows Multi-Head Self-Attention(W-MSA)的概念，比如在下图的4倍下采样和8倍下采样中，将特征图划分成了多个不相交的区域（Window），并且Multi-Head Self-Attention只在每个窗口（Window）内进行。相对于Vision Transformer中直接对整个（Global）特征图进行Multi-Head Self-Attention，这样做的目的是能够减少计算量的，尤其是在浅层特征图很大的时候。

##### 模型架构

Swin Transformer网络的架构图

![image-20240403185830960](D:\tool\typora\image\image-20240403185830960.png)

- 首先将图片输入到Patch Partition模块中进行分块，即每4x4相邻的像素为一个Patch，然后在channel方向展平（flatten）。假设输入的是RGB三通道图片，那么每个patch就有4x4=16个像素，然后每个像素有R、G、B三个值所以展平后是16x3=48，所以通过Patch Partition后图像shape由` [H, W, 3]`变成了` [H/4, W/4, 48]`。然后在通过Linear Embeding层对每个像素的channel数据做线性变换，由48变成C，即图像shape再由 `[H/4, W/4, 48]`变成了` [H/4, W/4, C]`。

- 然后就是通过四个Stage构建不同大小的特征图，除了Stage1中先通过一个Linear Embeding层外，剩下三个stage都是先通过一个Patch Merging层进行下采样（后面会细讲）。然后都是重复堆叠Swin Transformer Block注意这里的Block其实有两种结构，如图(b)中所示，这两种结构的不同之处仅在于一个使用了W-MSA结构，一个使用了SW-MSA结构。而且这两个结构是成对使用的，先使用一个W-MSA结构再使用一个SW-MSA结构。所以你会发现堆叠Swin Transformer Block的次数都是偶数（因为成对使用）。

- 最后对于分类网络，后面还会接上一个Layer Norm层、全局池化层以及全连接层得到最终输出。图中没有画，但源码中是这样做的。

##### Patch Merging

​		在每个Stage中首先要通过一个Patch Merging层进行下采样（Stage1除外）。如下图所示，假设输入Patch Merging的是一个4x4大小的单通道特征图（feature map），Patch Merging会将每个2x2的相邻像素划分为一个patch，然后将每个patch中相同位置（同一颜色）像素给拼在一起就得到了4个feature map。接着将这四个feature map在深度方向进行concat拼接，然后在通过一个LayerNorm层。最后通过一个全连接层在feature map的深度方向做线性变化，将feature map的深度由C变成C/2。通过这个简单的例子可以看出，通过Patch Merging层后，feature map的高和宽会减半，深度会翻倍。

<img src="D:\tool\typora\image\image-20240403191315869.png" alt="image-20240403191315869" style="zoom:33%;" />

##### W-MSA

​	引入Windows Multi-head Self-Attention（W-MSA）模块是为了减少计算量。如下图所示，左侧使用的是普通的Multi-head Self-Attention（MSA）模块，对于feature map中的每个像素（或称作token，patch）在Self-Attention计算过程中需要和所有的像素去计算。但在图右侧，在使用Windows Multi-head Self-Attention（W-MSA）模块时，首先将feature map按照MxM（例子中的M=2）大小划分成一个个Windows，然后单独对每个Windows内部进行Self-Attention。

![image-20240403191443882](D:\tool\typora\image\image-20240403191443882.png)

##### SW-MSA

​		前面采用W-MSA模块时，只会在每个窗口内进行自注意力计算，所以窗口与窗口之间是无法进行信息传递的。为了解决这个问题，作者引入了Shifted Windows Multi-Head Self-Attention（SW-MSA）模块，即进行偏移的W-MSA。如下图所示，左侧使用的是刚刚讲的W-MSA（假设是第L层），那么根据之前介绍的W-MSA和SW-MSA是成对使用的，那么第L+1层使用的就是SW-MSA（右侧图）。根据左右两幅图对比能够发现窗口（Windows）发生了偏移（可以理解成窗口从左上角分别向右侧和下方各偏移了$M/2$个像素。看下偏移后的窗口（右侧图），比如对于第一行第2列的2x4的窗口，它能够使第L层的第一排的两个窗口信息进行交流。再比如，第二行第二列的4x4的窗口，他能够使第L层的四个窗口信息进行交流，其他的同理。
![image-20240403192024244](D:\tool\typora\image\image-20240403192024244.png)

​		根据上图，可以发现通过将窗口进行偏移后，由原来的4个窗口变成9个窗口了。后面又要对每个窗口内部进行MSA，这样做感觉又变麻烦了。为了解决这个麻烦，作者又提出而了`Efficient batch computation for shifted configuration`，一种更加高效的计算方法。

![image-20240403193816719](D:\tool\typora\image\image-20240403193816719.png)

​		将AC移到最下方，将AB移到最右侧。这样又和原来一样是4个4x4的窗口了。为了防止不同区域合并到一起进行MSA出现信息乱窜的现象，在实际计算中使用的是`masked MSA`即带蒙板mask的MSA，这样就能够通过设置蒙板来隔绝不同区域的信息了。

​		关于mask如何使用，可以看下下面这幅图，下图是以上面的区域5和区域3为例。

![image-20240403194310524](D:\tool\typora\image\image-20240403194310524.png)

​		对于该窗口内的每一个像素（或称token，patch）在进行MSA计算时，都要先生成对应的query(q)，key(k)，value(v)。假设对于上图的像素0而言，得到$q^0$后要与每一个像素的k进行匹配（match），假设$α_{0,0}$代表$q^0$与像素0对应的$k^0$进行匹配的结果，那么同理可以得到$α_{0,0}$至$α_{0,15}$ 。按照普通的MSA计算，接下来就是SoftMax操作了。但对于这里的`masked MSA`，像素0是属于区域5的，我们只想让它和区域5内的像素进行匹配。那么我们可以将像素0与区域3中的所有像素匹配结果都减去100（例如$α_{0,2}, α_{0,3}, α_{0,6}, α_{0,7}$等等）由于α的值都很小，一般都是零点几的数，将其中一些数减去100后在通过SoftMax得到对应的权重都等于0了。所以对于像素0而言实际上还是只和区域5内的像素进行了MSA。那么对于其他像素也是同理，具体代码是怎么实现的，后面会在代码讲解中进行详解。注意，在计算完后还要把数据给挪回到原来的位置上。

##### Relative Position Bias

略



### 10、ConvNeXt

`ConvNeXt`“毫无亮点”，使用的全部都是现有的结构和方法。

**Changing stage compute ratio**：改变堆叠比例。在`Swin Transformer`中，`stage3`堆叠block的占比更高。所以作者就将`ResNet50`中的堆叠次数由`(3, 4, 6, 3)`调整成`(3, 3, 9, 3)`，和`Swin-T`拥有相似的FLOPs

**Changing stem to “Patchify”**：改变下采样模块。一般最初的下采样模块stem一般都是通过一个卷积核大小为7x7步距为2的卷积层以及一个步距为2的最大池化下采样共同组成，高和宽都下采样4倍。但在Transformer模型中一般都是通过一个卷积核非常大且相邻窗口之间没有重叠的（即stride等于kernel_size）卷积层进行下采样。所以作者将`ResNet`中的`stem`也换成了和`Swin Transformer`一样的`patchify`

**借鉴 ResNeXt**：借鉴了ResNeXt中的组卷积grouped convolution，采用的是更激进的depthwise convolution。接着作者将最初的通道数由64调整成96和`Swin Transformer`保持一致。

**Inverted Bottleneck**：采用`Inverted Bottleneck`倒残差模块

**Moving up depthwise conv layer**：将`depthwise conv`上移。原来是1x1 conv -> `depthwise conv` -> 1x1 con，现在变成了`depthwise conv` -> 1x1 conv-> 1x1 conv

**Increasing the kernel size**：增大卷积核，将`depthwise conv`的卷积核大小由`3x3`改成了`7x7`。

**Fewer activation functions**：更少的激活函数。在`ConvNeXt Block`中也减少激活函数的使用。

**Fewer normalization layers**：使用更少的Normalization。减少了`ConvNeXt Block`中的Normalization层，只保留了`depthwise conv`后的Normalization层。

**Substituting BN with LN**：将BN替换成LN。

**Separate downsampling layers**：单独的下采样层。`ConvNext`网络单独使用了一个下采样层，就是通过一个Laryer Normalization加上一个卷积核大小为2步距为2的卷积层构成。



##### [模型架构](D:\tool\typora\image\image-20240403200635863.png)

`Layer Scale`操作其实就是将输入的特征层乘上一个可训练的参数，该参数就是一个向量，元素个数与特征层channel相同，即对每个channel的数据进行缩放。



<img src="D:\tool\typora\image\image-20240403200635863.png" alt="image-20240403200635863" style="zoom:50%;" />



### 11、RepVGG

​		RepVGG整个模型就是在不断堆叠RepVGG Block

![image-20240403210219450](D:\tool\typora\image\image-20240403210219450.png)

##### RepVGG Block

​		其中图(a)是进行下采样（stride=2）时使用的RepVGG Block结构，图(b)是正常的（stride=1）RepVGG Block结构。通过图(b)可以看到训练时RepVGG Block并行了三个分支：一个卷积核大小为3x3的主分支，一个卷积核大小为1x1的shortcut分支以及一个只连了BN的shortcut分支。

![image-20240403203700451](D:\tool\typora\image\image-20240403203700451.png)

- 为什么训练时要采用多分支结构?

至少根据现有的一些经验来看，并行多个分支一般能够增加模型的表征能力。

- 为什么推理时作者要将多分支模型转换成单路模型?

采用单路模型会更快、更省内存并且更加的灵活。除此之外，在多分支转化成单路模型后很多算子进行了融合（比如Conv2d和BN融合），使得计算量变小了，而且算子减少后启动kernel的次数也减少了。

##### 结构重参数化

​		将训练好的RepVGG Block转成推理时的模型结构，即`structural re-parameterization technique`过程。结构重参数化主要分为两步，第一步主要是将Conv2d算子和BN算子融合以及将只有BN的分支转换成一个Conv2d算子，第二步将每个分支上的`3x3`卷积层融合成一个卷积层。

![image-20240403211609309](D:\tool\typora\image\image-20240403211609309.png)

**1、融合Conv2d和BN**

​	因为Conv2d和BN两个算子都是做线性运算，所以可以融合成一个算子。融合是在网络训练完之后做的，所以现在讲的默认都是推理模式，**注意BN在训练以及推理时计算方式是不同的**。对于BN层（推理模式），主要包含4个参数：$μ$均值、$σ、γ$和$β$，其中$μ$和$σ^2$ 是训练过程中统计得到的，$γ$和$β$是训练学习得到的。

**2、将1x1卷积转换成3x3卷积**

​		以`1x1`卷积层中某一个卷积核为例，只需在原来权重周围补一圈零就行了，这样就变成了`3x3`的卷积层，注意为了保证输入输出特征图高宽不变，此时需要将padding设置成1。最后将卷积层和BN层进行融合即可。

**3、将BN转换成3x3卷积**

​		对于只有BN的分支由于没有卷积层，所以我们可以先自己构建出一个卷积层来。构建了一个`3x3`的卷积层，该卷积层只做了恒等映射，即输入输出特征图不变，然后将卷积层和BN层进行融合。

**4、多分支融合**

​		前三点已经把每个分支融合转换成一个`3x3`的卷积层，接下来将多分支转换成一个单路`3x3`卷积层。合并的过程其实也很简单，直接将这三个卷积层的参数相加即可。

![image-20240403211407056](D:\tool\typora\image\image-20240403211407056.png)

### 12、MobileViT

##### 模型架构

​		MobileViT主要由普通卷积，MV2（MobiletNetV2中的`Inverted Residual block`），`MobileViT block`，全局池化以及全连接层共同组成。

![image-20240403204813405](D:\tool\typora\image\image-20240403204813405.png)

##### MobileViT block

​		MobileViT block的大致结构为：首先将特征图通过一个卷积核大小为n×n（代码中是3×3）的卷积层进行局部的特征建模，然后通过一个卷积核大小为1x1的卷积层调整通道数。接着通过`Unfold -> Transformer -> Fold`结构进行全局的特征建模，然后再通过一个卷积核大小为1x1的卷积层将通道数调整回原始大小。接着通过shortcut捷径分支与原始输入特征图进行Concat拼接（沿通道channel方向拼接），最后再通过一个卷积核大小为nxn（代码中是3x3）的卷积层做特征融合得到输出。

![image-20240403205304817](D:\tool\typora\image\image-20240403205304817.png)

**Unfold -> Transformer -> Fold结构**：

​		图中的`Patch`大小为`2x2`，即每个`Patch`由4个`Pixel`组成。在进行`Self-Attention`计算的时候，每个`Token`（图中的每个`Pixel`或者说每个小颜色块）只和自己颜色相同的`Token`进行`Attention`，这样就达到了减少计算量的目的。

![image-20240403205728484](D:\tool\typora\image\image-20240403205728484.png)

`Unfold`就是将相同颜色的`Token`展平在一个序列中，这样就可以直接使用普通的`Self-Attention`并行计算每个序列的`Attention`了。最后在通过`Fold`折叠回原特征图。

![image-20240403205811702](D:\tool\typora\image\image-20240403205811702.png)

**Patch Size对性能的影响**：

​		大的`patch_size`能够提升网络推理速度，但是会丢失一些细节信息。通过对比可以发现，在图像分类和目标检测任务中（对语义细节要求不高的场景），配置A和配置B在Acc和mAP上没太大区别，但配置B要更快。但在语义分割任务中（对语义细节要求较高的场景）配置A的效果要更好。

![image-20240403210110302](D:\tool\typora\image\image-20240403210110302.png)



## Pytorch实例

劈里啪啦教程代码：D:\ai\project\pytorch_study\test1_offical_demo_Pilipala

小土堆教程代码：    D:\ai\project\pytorch_study\pytorch_example_Tudui\src

1、**pytorch 中 tensor（也就是输入输出层）的 通道排序为：`[batch, channel, height, width]`，**一般图片默认的通道排序为：`[height, width, channel]`，电脑qq截屏的通道排序为：`[height, width, channel,trans]`，trans为透明度。

2、pytorch中的卷积、池化、输入输出层中参数的含义与位置如下图：

![image-20231115202407889](D:\tool\typora\image\image-20231115202407889.png)

### 1、网络搭建

#### （1）卷积Conv2d

​			1、卷积核的channel与输入特征层的channel相同

​			2、输出的特征矩阵channel与卷积核个数相同

![image-20231201165328698](D:\tool\typora\image\image-20231201165328698.png)

​		经卷积后的输出层尺寸计算公式为：

<img src="D:\tool\typora\image\image-20231115202646479.png" alt="image-20231115202646479" style="zoom:80%;" />

- 输入图片大小 W×W（一般情况下Width=Height）
- 卷积核(Filter)大小 F×F
- 步长 S
- 计算出小数了默认向下取整
- padding的像素数 P             注：2P表示上下(左右)填充的像素之和，可以不相等，即为2P=p1+p2

​		nn.Conv2d中padding参数可以为int或者turple类型，turple为(1,2)时代表上下补一列0，左右补两列0

PS：如果计算出来的结果不为整数时，卷积过程会直接忽略最后一行和最后一列，以保证n为整数

#### （2）池化 MaxPool2d

![image-20231201170532471](D:\tool\typora\image\image-20231201170532471.png)

![image-20231201170628627](D:\tool\typora\image\image-20231201170628627.png)

#### （3）Tensor的展平

​		view()，在经过第二个池化层后，数据还是一个三维的Tensor (32, 5, 5)，需要先经过展平后(32\*5\*5)再传到全连接层

```python
  x = self.pool2(x)            # output(32, 5, 5)
  x = x.view(-1, 32*5*5)       # output(32*5*5)
  x = F.relu(self.fc1(x))      # output(120)
```

#### （4）全连接 Linear

### 2、导入数据集

（1）导入包

```python
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
```

（2）数据预处理：由shape (H x W x C) in the range [0, 255] → shape (C x H x W) in the range [0.0, 1.0]

```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```

（3）导入加载训练测试集

利用`torchvision.datasets`函数可以在线导入pytorch中的数据集

以导入加载训练集为例：

```python
# 导入50000张训练图片
train_set = torchvision.datasets.CIFAR10(root='./data', 	 # 数据集存放目录
					   train=True,		 					# 表示是数据集中的训练集
                       download=True,  	 # 第一次运行时为True，下载数据集，下载完成后改为False
                       transform=transform) 				# 预处理过程
# 加载训练集，实际过程需要分批次（batch）训练                                        
train_loader = torch.utils.data.DataLoader(train_set, 	  # 导入的训练集
										   batch_size=50, # 每批训练的样本数
                                          shuffle=False,  # 是否打乱训练集
                                          num_workers=0)  # 使用线程数，在windows下设置为0
#（测试集中使用）获取测试集中的图像和标签，用于accuracy计算
test_data_iter = iter(test_loader)		        # iter(test_loader)创建了一个迭代器，用于遍历												test_loader中的数据
test_image, test_label = test_data_iter.next()
```

（4）模型训练

![image-20231115203707414](D:\tool\typora\image\image-20231115203707414.png)

```python
net = LeNet()						  				# 定义训练的网络模型
loss_function = nn.CrossEntropyLoss() 				# 定义损失函数为交叉熵损失函数 
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器（训练参数，学习率）

for epoch in range(5):  # 一个epoch即对整个训练集进行一次训练
    running_loss = 0.0
    time_start = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):   # 遍历训练集，step从0开始计算
        inputs, labels = data 	# 获取训练集的图像和标签
        optimizer.zero_grad()   # 清除历史梯度

        # forward + backward + optimize
        outputs = net(inputs)  				  # 正向传播
        loss = loss_function(outputs, labels) # 计算损失
        loss.backward() 					  # 反向传播
        optimizer.step() 					  # 优化器更新参数

        # 打印耗时、损失、准确率等数据
        running_loss += loss.item()
        if step % 1000 == 999:    # print every 1000 mini-batches，每1000步打印一次
            with torch.no_grad(): # 在以下步骤中（验证过程中）不用计算每个节点的损失梯度，防止内存占用
                outputs = net(test_image)# 测试集传入网络（test_batch_size=10000），output维度为[10000,10]
                predict_y = torch.max(outputs, dim=1)[1] # 以output中值最大位置对应的索引（标签）作为预测输出
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                print('%f s' % (time.perf_counter() - time_start))        # 打印耗时
                running_loss = 0.0
            
print('Finished Training')
# 保存训练得到的参数
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)
```

​    

（5）使用GPU/CPU训练

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 或者
net = LeNet()
net.to(device) # 将网络分配到指定的device中
```

### 3、预测（测验）

```python
# 导入包
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

# 数据预处理
transform = transforms.Compose(
    [transforms.Resize((32, 32)), # 首先需resize成跟训练集图像一样的大小
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 导入要测试的图像（自己找的，不在数据集中），放在源文件目录下
im = Image.open('horse.jpg')
im = transform(im)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  # 对数据增加一个新维度，因为tensor的参数是[batch, channel, height, width] 

# 实例化网络，加载训练好的模型参数
net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))

# 预测
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy()
print(classes[int(predict)])

```





# 目标检测

### 简介

#### 评价指标

##### IOU 

全称 intersection-over-union 重叠度，交并比

<img src="D:\tool\typora\image\image-20240418205006561.png" alt="image-20240418205006561" style="zoom: 25%;" />

#### 标注软件

###### Labelme

在label环境中，在终端输入`labelme`自动打开该标注软件，[教学链接](http://t.csdnimg.cn/mGTe8)

###### EISeg

在label环境中，在终端输入`eiseg`自动打开该标注软件，[教学链接](http://t.csdnimg.cn/5oGFj)

###### Labelimg

在label环境中，在终端输入`labelimg`自动打开该标注软件，[教学链接](https://blog.csdn.net/knighthood2001/article/details/125883343)

可以输出txt或者xml文件

### 1、RCNN

​		经典的目标检测算法使用滑动窗法依次判断所有可能的区域（遍历）。RCNN算法提出在图像中创建多个边界框，检查这些边框中是否含有目标物体。RCNN使用选择性搜索来从一张图片中提取这些边框，之后仅在这些候选区域上(采用CNN)提取特征，进行判断。

​		RCNN还可以称为基于推荐区域的卷积神经网络。在这篇论文中，我们提出了一种简单和可扩展的检测算法。我们的方法结合了两种重要的观点：

1. 大容量的卷积神经网络应用到自下而上的推荐区域用来定位和分割目标。
2. 当已有标签的数据集样本不足时，监督的预训练辅助任务，以及对特定领域的微调，会产生显著的性能提升。因为我们结合了CNNs和推荐区域的思想，所以我们称我们的方法为R-CNN：Regins with CNN feature。

（2014年）在过去十年里，各种视觉识别任务的进展都大量地**基于SIFT和HOG特征**的应用。SIFT和HOG是块方向的直方图，我们可以将它与灵长类视觉路径中的第一个皮层区域中的复杂的细胞进行粗略地联系起来。

![image-20240418104534061](D:\tool\typora\image\image-20240418104534061.png)

​		我们的系统：1）输入一张图片 （2）从图片中提取出大概2000个自下而上的候选区域 （3）使用一个大的卷积神经网络对每一个候选区域进行计算特征，得到特征向量（4）使用具体的分类线性SVMs给每一个候选区域进行分类。（5）使用回归器精细修正候选框位置

​		我们的目标检测系统由三个模块组成。第一个方案生成与类别无关的区域方案。这些方案定义了可供检测器使用的候选检测集。第二个模块是一个大型卷积神经网络，它从每个区域提取一个固定长度的特征向量。第三个模块是一组特定于类的线性支持向量机。在本节中，我们将介绍每个模块的设计决策，描述它们的测试时间使用，详细说明如何学习它们的参数，并在PASCAL VOC 2010-12和ILSVRC2013上显示检测结果。

<img src="D:\tool\typora\image\image-20240419150007810.png" alt="image-20240419150007810" style="zoom: 33%;" />

​		Bbox reg：全称Bounding-box regression，是用来微调窗口。



#### Selective Search(SS算法)

**选择性搜索 ：**组成目标物体通常有四个要素：变化尺度、颜色、结构（材质）、所占面积。选择性搜索会确定物体在图片中的这些特征，然后基于这些特征突出不同区域。

![image-20240418213055450](D:\tool\typora\image\image-20240418213055450.png)

#### SVM

​		中文名支持向量机（support vector machines, SVM）是一种二分类模型，针对每一种物体分类都有一个分类器。它的基本模型是定义在特征空间上的**间隔最大的线性分类器**，间隔最大使它有别于感知机；SVM还包括**核技巧**，这使它成为实质上的非线性分类器。SVM的学习策略就是间隔最大化，可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题。SVM的的学习算法就是求解凸二次规划的最优化算法。

​		SVM是一种经典的监督学习算法，用于解决二分类和多分类问题。其核心思想是通过在特征空间中找到一个最优的超平面来进行分类，并且间隔最大。

![image-20240418215038987](D:\tool\typora\image\image-20240418215038987.png)

​	SVM 最初被设计用于二分类问题。它通过找到最优的决策边界（或超平面），将数据集中的点根据其类别分开。对于非线性可分问题，SVM 使用核技巧（kernel trick）将数据映射到更高维的空间，以便于找到合适的分离超平面。



**用RCNN检测目标物体的具体步骤如下：**

1. 我们首先取一个预训练卷积神经网络。
2. 根据需要检测的目标类别数量，训练网络的最后一层。
3. 得到每张图片的感兴趣区域（Region of Interest），对这些区域重新改造，以让其符合CNN的输入尺寸要求。对每个候选区域，使用深度卷积网络提取特征 （CNN）。
4. 特征送入向量机（SVM）来辨别目标物体和背景。**对每个类别，我们都要训练一个二元SVM**。
5. 最后，我们训练一个线性回归模型，使用回归器精细修正候选框位置。

详细解释：

- 特征送入每一类的SVM分类器，判定类别:（假设提取出2000个特征框）

​		将2000×4096的特征矩阵与20个SVM组成的权值矩阵4096×20相乘，获得2000×20的概率矩阵，每一行代表一个建议框归于每个目标类别的概率。分别对上述2000×20维矩阵中每一列即每一类进行非极大值抑制剔除重叠建议框，得到该列即该类中得分最高的一些建议框。

<img src="D:\tool\typora\image\image-20240419153632197.png" alt="image-20240419153632197" style="zoom:33%;" />

- 使用回归器精细修正候选框位置

![image-20240419154154897](D:\tool\typora\image\image-20240419154154897.png)

**非极大值抑制（**NMS**）：**

​	目的是为了去除冗余的检测框。RCNN会从一张图片中找出n个可能是物体的矩形框，然后为每个矩形框为做类别分类概率。非极大值抑制（NMS）顾名思义就是抑制不是极大值的元素，搜索局部的极大值。

寻找得分最高的目标  ======>  计算其他目标与该目标的iou值  ======>  删除所有iou值大于给定阈值的目标

![image-20240418214248048](D:\tool\typora\image\image-20240418214248048.png)

<img src="D:\tool\typora\image\image-20240419150252815.png" alt="image-20240419150252815" style="zoom: 33%;" />

<img src="D:\tool\typora\image\image-20240419153111177.png" alt="image-20240419153111177" style="zoom:33%;" />

**问题**：

- 不论是训练或者预测，对delective search选出的的2000 ROI全部都得通过CNN来取得特征。整个时间花费很长。一张47s。

- CNN神经网路的特征提取器和用来预测分类的SVM式分开的,也就是特征提取器的过程不会因SVM和回归器的调整而更新。

- RCNN具有非常复杂的操作流程，而且每都分裂的，如特征提取器通过CNN获得，而最终分类结果由SVM获得，BBOX位置则是通过归器调整而获得,不是端到端的操作。

- Region proposal需事先提取并保存，占用空间较大。对于SVM和bbox回归训练，需要从每个图像中的每个目标候选框提取特征，并写入磁盘，对于非常深的网络需要储存空间过大。

  

RCNN：将候选框区域输入卷积神经网络得到特征。

Fast RCNN：将整张图像送入网络，从特征图像上提取相应的候选区域，这些候选区域的特征不需要再重复计算。通过这种操作一次性计算整张图像特征，只用计算一次，节省了时间。

### 2、Fast RCNN

​		CNN的作者Ross Girshick提出了一种想法，在每张照片上只运行一次CNN，然后找到一种方法在2000个区域中进行计算。在Fast RCNN中，我们将图片输入到CNN中，会相应地生成传统特征映射。利用这些映射，就能提取出感兴趣区域。之后，我们使用一个Rol池化层将所有提出的区域重新修正到合适的尺寸，以输入到完全连接的网络中。

简单地说，这一过程含有以下步骤：

1. 输入图片。
2. 输入到卷积网络中，得到特征图，SS算法生成候选框(1k~2k个)投影到特征图上获得相应的特征矩阵。
3. 将每个特征矩阵通过Rol池化层对这些区域重新调整(缩放到7×7大小的特征图)，将特征图展平，输入到全连接网络层得到预测结果。
4. 在网络的顶层用softmax层输出类别。同样使用一个线性回归层，输出相对应的边界框。

​	注: RoI: Region of Interest感兴趣区域

将识别矩形框和分类器结合到一个网络之中

![image-20240419192859655](D:\tool\typora\image\image-20240419192859655.png)

和RCNN所需要的三个模型不同，Fast RCNN只用了一个模型就同时实现了区域的特征提取、分类、边界框生成。

<img src="D:\tool\typora\image\image-20240419111715842.png" alt="image-20240419111715842" style="zoom:67%;" />

训练数据的采样（正样本——需要检测的目标，负样本——背景）

​	IoU>0.5就认为是正样本，0.1~0.5之间认为是负样本。并不是训练了所有SS算法选出来的框，只是随机采样了一部分？

#### RoI Pooling层

​		不限制输入图像的尺寸。将所有大小的图片分成7*7份分别进行最大池化下采样，得到7\*7的特征矩阵。

<img src="D:\tool\typora\image\image-20240419193853439.png" alt="image-20240419193853439" style="zoom: 33%;" />

**分类器：** 

<img src="D:\tool\typora\image\image-20240419194444567.png" alt="image-20240419194444567" style="zoom: 33%;" />

**边界框回归器：**

$(d_x,d_y,d_w,d_h)$ 分别为候选边界框的中心x, y坐标，以及宽高。

<img src="D:\tool\typora\image\image-20240419194538865.png" alt="image-20240419194538865" style="zoom:33%;" />

<img src="D:\tool\typora\image\image-20240419195525950.png" alt="image-20240419195525950" style="zoom:33%;" />

**损失： **

略

**问题：**

​		Fast RCNN也有某些局限性。它同样用的是选择性搜索作为寻找感兴趣区域的，这一过程通常较慢。与RCNN不同的是，Fast RCNN处理一张图片大约需要2秒。但是在大型真实数据集上，这种速度仍然不够理想。

### 3、Faster R-CNN

https://blog.csdn.net/wjinjie/article/details/105962056

​		RPN是一个完全卷积网络，可以同时预测每个位置的目标边界和目标得分。对RPN进行端到端的训练，以生成高质量的Fast R-CNN用于检测推荐区域。通过使用最近流行的“注意力”机制的神经网络共享RPN和Fast R-CNN的卷积特征，将RPN和Fast R-CNN进一步合并为一个单个网络，RPN各部分告诉单个网络要看的地方。**Faster R-CNN网络实际上就是RPN + Fast R-CNN，**计算将RPN替代选择性搜索。

​		我们的观察结果是，基于推荐区域的检测器（如Fast RCNN）使用的卷积特征图也可用于生成推荐区域。在这些卷积特征之上，我们通过添加一些其他卷积层来构建RPN，这些卷积层同时回归规则网格上每个位置的区域边界和客观性得分。 因此，RPN是一种全卷积网络（FCN），可以专门针对生成推荐区域的检测任务进行端到端训练。

​		RPN旨在以各种比例和纵横比有效预测推荐区域。与使用图像金字塔（图1，a）或卷积金字塔（图1，b）的流行方法相比，我们介绍了新颖的"anchor box"作为多种比例和纵横比的参考。**我们的方案可以看作是回归参考的金字塔（图1，c），它避免了枚举具有多个比例或纵横比的图像或卷积核。当使用单比例尺图像进行训练和测试时，该模型表现良好，从而提高了运行速度。**

![image-20240418145502818](D:\tool\typora\image\image-20240418145502818.png)

Faster RCNN工作的大致过程：

1. 输入图像到卷积网络中，生成该图像的特征图。
2. 在特征映射上应用RPN结构生成候选框，将RPN生成的候选框投影到特征图上获得相应的特征矩阵。
3. 将每个特征矩阵通过ROI pooling层缩放到7x7大小的特征图，接着将特征图展平通过一系列全连接层得到预测结果。

<img src="D:\tool\typora\image\image-20240418150427972.png" alt="image-20240418150427972" style="zoom: 50%;" />

网络框架

![image-20240420191447610](D:\tool\typora\image\image-20240420191447610.png)

**卷积层**

采用的是VGG16

- 所有的conv层都是：kernel_size=3，pad=1，stride=1
- 所有的pooling层都是：kernel_size=2，pad=0，stride=2

#### Region Proposal Networks (RPN)

**介绍**

​		介绍了用于**推荐区域网络**（RPN）的设计和特性。区域推荐网络（RPN）接收（任意大小的）图像作为输入，并输出一组矩形的目标推荐框，每个目标推荐框都有一个客观评分。因为我们的最终目标是与Fast R-CNN目标检测网络共享计算，所以我们假设两个网络共享一组共同的卷积层。

​		为了生成推荐区域，我们在最后共享的卷积层输出的卷积特征图上滑动一个小型网络。这个小网络将输入卷积特征图的n×n空间窗口作为输入。每个滑动窗口都映射到一个较低维的特征（ZF为256-d，VGG为512-d，后面是ReLU）。此功能被馈入两个同级的全连接层——边界框回归层（reg）和框分类层（cls）。在图3的单个位置（左）显示了此微型网络。请注意，由于微型网络以滑动窗口的方式运行，因此完全连接的层将在所有空间位置上**共享**。自然地，该体系结构由n×n卷积层和两个同级1×1卷积层（分别用于reg和cls）实现。

![image-20240418150856233](D:\tool\typora\image\image-20240418150856233.png)

**RPN原理**	

​	在卷积生成的特征图上生成一个滑动窗口(padding=1,stride=1经过每一个特征图上的点)，每滑动一个地方生成一个行向量，有256个元素(此个数跟网络输出的channel有关)。最后得到的map跟特征图大小一样，深度为256.

![image-20240420194421624](D:\tool\typora\image\image-20240420194421624.png)

**anchor**为当前特征图上的点按照比例尺(卷积中的步距)找到在原图中的位置(记为中心点)，以这个中心点为中心定义一系列不同长宽比的矩形(如上左图所示，实际有9个)。**$k$**为anchor的个数。

**cls layer**生成类别分数，$2k$表示分别为是前景or背景的概率。每两个对应一个anchor,分别对应不同框中的内容为背景/所需目标的概率。

**reg layer**是边界框回归参数，根据其调整边框。$4k$表示分别为边界框参数(中心点，宽，高)。

<img src="D:\tool\typora\image\image-20240420200712892.png" alt="image-20240420200712892" style="zoom: 33%;" />

针对边界框的比例和尺寸大小各给了三种尺寸比例：

![image-20240420201459784](D:\tool\typora\image\image-20240420201459784.png)

<img src="D:\tool\typora\image\image-20240420201514920.png" alt="image-20240420201514920" style="zoom:25%;" />

- 低的感受野(eg:171)可以预测大的目标(256×256)。
- anchor不是候选框(proposal)，利用RPN生成的边界框回归参数将anchor调整到所需要的候选框。

**训练数据的采样（正样本、负样本）**

​	(初始会生成上万个anchor，忽略边界，采用非极大值抑制，IoU设为0.7,每张图片生成大概2000个候选框)

一共随机的选取128个正样本，128个负样本，正样本不足时用负样本补充，总数保持在256。

- 正样本：与ground-truth相交(IoU)最大的anchor和IoU>0.7的anchor。
- 负样本：IoU都<0.3。

对正负样本之外的样本全部丢弃达到减少anchor的目的。

ground-truth——人工标注的真正的标注框

**RPN的损失计算**
$$
L\left(\left\{p_{i}\right\},\left\{t_{i}\right\}\right)=\frac{1}{N_{c l s}} \sum_{i} L_{c l s}\left(p_{i}, p_{i}^{*}\right)+\lambda \frac{1}{N_{\text {reg }}} \sum_{i} p_{i}^{*} L_{\text {reg }}\left(t_{i}, t_{i}^{*}\right)
$$
![image-20240420204857964](D:\tool\typora\image\image-20240420204857964.png)
$$
\frac{1}{N_{cls} }\approx  \lambda \frac{1}{N_{reg} }
$$
多分类的交叉熵损失（只有两类，论文中使用）

<img src="D:\tool\typora\image\image-20240420205430272.png" alt="image-20240420205430272" style="zoom: 50%;" />

二分类的交叉熵损失(pytorch代码中使用)

<img src="D:\tool\typora\image\image-20240420210015729.png" alt="image-20240420210015729" style="zoom: 50%;" />

<img src="D:\tool\typora\image\image-20240420210157103.png" alt="image-20240420210157103" style="zoom: 50%;" />

**Faster RCNN训练**（晕）

略



问题：

- 首先是anchor就是目标物体的概率（不考虑类别）
- 第二个就是anchor经过调整能更合适目标物体的边界框回归量

总结：

![image-20240419145435594](D:\tool\typora\image\image-20240419145435594.png)



# yolo模型

You Only Look Once

- Backbone： Feature Extractor提取特征的网络，其作用就是提取图片中的信息，供后面的网络使用。
- Neck ： 放在backbone和head之间的，是为了更好的利用backbone提取的特征，起着“特征融合”的作用。
- Head：利用前面提取的特征，做出识别。

常见的一些Backbone, Neck, Head网络

![image-20240508195528270](D:/tool/typora/image/image-20240508195528270.png)

## **性能指标评价**

**IoU**(Intersection over Union)

交并比，用于量化预测边界框和真实边界框之间的重叠。

**混淆矩阵**(Confusion Matrix)

其是一种用于评估分类模型性能的表格形式。它以实际类别(真实值)和模型预测类别为基础，将样本分类结果进行统计和汇总。

<img src="D:/tool/typora/image/image-20240513171611810.png" alt="image-20240513171611810" style="zoom: 67%;" />

- **T(True)：**最终预测结果正确。
- **F(False)：**最后预测结果错误。
- **P(Positive)：**模型预测其是正样本(目标本身是**狗**，模型也预测它是**狗**)。
- **N(Negative)：**模型预测其是负样本(目标本身是**狗**，但模型预测它是个**猫**)。
- **TP：**正例正确分类为正例           (目标本身是**狗**，模型也预测它是**狗**，预测正确)。
- **TN：**把负例正确分类为负例      (目标本身不是**狗**，模型预测它不是**狗**，是个其他的东西，预测正确)。
- **FP：**把负例错误分类为正例       (目标本身不是**狗**，模型预测它是**狗**，预测错误)。
- **FN**：把正例错误分类为负例      (目标本身是**狗**，模型预测它不是**狗**，是个其他的东西，预测错误)。

**Accuracy**

准确率；所有预测中正确的百分比。

**Precision**

精确率，指模型识别出的正确正样本数占所有被识别为正样本的样本数的比例。一般针对某一类别。
$$
precision = \frac{TP}{TP+FP}
$$
**Recall**

**召回率，指模型识别出的正样本数占真实正样本数的比例。**召回率越高，实际为正样本（P）被预测出来的概率越高。
$$
recall = \frac{TP}{TP+FN}
$$
**PR曲线**

Precision-Recall曲线，代表的是精准率与召回率的关系，一般情况下，将recall设置为横坐标，precision设置为纵坐标。

<img src="D:/tool/typora/image/image-20240513202924703.png" alt="image-20240513202924703" style="zoom: 33%;" /><img src="D:/tool/typora/image/image-20240513210710528.png" alt="image-20240513210710528" style="zoom: 25%;" />

**AP**(Average Precision)

平均精度。PR曲线下面的面积，通常来说一个越好的模型，AP值越高。

**mAP**(mean Average Precision)

均值平均精度或平均精确度。即各个类别AP的平均值，用于表达多类标签预测的性能，如AP一样，mAP越高，性能越好。

对于二分类问题，AP和mAP都等于精确度。

- **mAP@.5**：		当IoU为0.5时的mAP。

- **mAP@.5 : .95**   当IoU为range(0.5 : 0.95 : 0.05)时的mAP的平均数。

- **mAP50**:           表示在多个类别上的平均精度。mAP50表示在50%的loU阈值下的mAP值。

- **mAP50-95**:     这是一个更严格的评价指标，它计算了在50-95%的loU阈值范围内的mAP值，然后取平均。这能够更准确地评估模型在不同loU阈值下的性能。

**F1-Score**

综合考虑精确度和召回率的指标，$F1=2*(Precision*Recal)/(Precision + Recal)$

其余输出结果的一些信息

**Labels图片**

<img src="D:/tool/typora/image/image-20240513205516248.png" alt="image-20240513205516248" style="zoom: 33%;" />

Labels图片代表每个检测到的目标的类别和边界框信息。每个目标都由一个矩形边界框和一个类别标签表示，逆时针来看这个图
片。

1. 目标类别:该像素点所检测到的目标类别，例如飞机等。
2. 目标位置:该像素点所检测到的目标在图像中的位置，即该像素点在图像中的坐标。
3. 目标大小:该像素点所检测到的目标的大小，即该像素点所覆盖的区域的大小。
4. 其他信息:例如目标的旋转角度等其他相关信息。

**labels correlogram**

​		labels_corelogram是一张颜色矩阵图，用于**显示目标检测算法在训练过程中预测标签之间的相关性**。它可以帮助我们理解目标检测算法在训练过程中的行为和表现，以及预测标签之间的相互影响。通过观察labels corelogram，我们可以了解到目标检测算法在不同类别之间的区分能力，以及对于不同类别的预测精度。此外，我们还可以通过比较不同算法或不同数据集labels correlogram，来评估算法的性能和数据集的质量。

<img src="D:/tool/typora/image/image-20240513205822238.png" alt="image-20240513205822238" style="zoom: 33%;" />

**P-curve**

准确率和置信度的关系。

<img src="D:/tool/typora/image/image-20240513205928197.png" alt="image-20240513205928197" style="zoom: 33%;" />

**R_curve**

召回率和置信度的关系。

<img src="D:/tool/typora/image/image-20240513210030756.png" alt="image-20240513210030756" style="zoom:33%;" />

results.csv

记录了一些我们训练过程中的参数信息，包括损失和学习率等。

result.png

主要看后面的四幅图mAP50、mAP50-95、metrics/precision、metrics/recall四张图片。

<img src="D:/tool/typora/image/image-20240513210341114.png" alt="image-20240513210341114" style="zoom: 50%;" />

> - **mAP50**:表示在多个类别上的平均精度。mAP50表示在50%的loU阈值下的mAP值。
> - **mAP50-95**:这是一个更严格的评价指标，它计算了在50-95%的loU阈值范围内的mAP值，然后取平均。这能够更准确地评估模型在不同loU阈值下的性能。
> - **metrics/precision**:精度(Precision)是评估模型预测正确的正样本的比例。在目标检测中，如果模型预测的边界框与真实的边界框重合，则认为预测正确。
> - **metricsrecal**:召回率(Recal)是评估模型能够找出所有真实正样本的比例。在目标检测中，如果真实的边界框与预测的边界框重合，则认为该样本被正确召回。

其余几张图为检测效果图。

## 1、yolov1

2016年发表

<img src="D:\tool\typora\image\image-20240423150854887.png" alt="image-20240423150854887" style="zoom:50%;" />

具体实现过程如下：

1. 将一幅图像分成 S×S个网格（grid cell），如果某个object 的中心(通过人工标定得到)落在这个网格中，则这个网格就负责预测这个object。

2. 每个网格要预测B个 bounding box，每个bounding box要预测 (x, y, w, h) 和confidence共5个值。
3. 每个网格还要预测一个类别信息，记为 C 个类。
4. 总的来说，S×S 个网格，每个网格要预测B个bounding box ，还要预测C个类。网络输出就是一个 S × S × (5×B+C) 的张量。

- 注：$confidence = Pr(Object)*IOU^{truth}_{pred}$。该网格中有目标$Pr(Object)=1$,可以简单的理解为等于iou,无目标时$Pr(Object)=0$。bounding box为边界框。

​		在实际过程中，YOLOv1把一张图片划分为了7×7个网格，并且每个网格预测2个Box（Box1和Box2），20个类别。所以实际上，S=7，B=2，C=20。那么网络输出的shape也就是：7×7×30。

<img src="D:\tool\typora\image\image-20240423151231303.png" alt="image-20240423151231303" style="zoom: 33%;" />

最终给出的目标概率为：

![image-20240423152544921](D:\tool\typora\image\image-20240423152544921.png)

**网络结构**

![image-20240423152634998](D:\tool\typora\image\image-20240423152634998.png)

**损失函数**

![image-20240423153047201](D:\tool\typora\image\image-20240423153047201.png)

问题：

- YOLO对相互靠近的物体，以及很小的群体检测效果不好，这是因为一个网格只预测了2个框，并且都只属于同一类。
- 由于损失函数的问题，定位误差是影响检测效果的主要原因，尤其是大小物体的处理上，还有待加强。（因为对于小的bounding boxes，small error影响更大）
- YOLO对不常见的角度的目标泛化性能偏弱。

## 2、yolov2

2017年发布，改进方法：

**Batch Normalization**

​		简称 BN ，意思是批量标准化。BN 对数据进行预处理（统一格式、均衡化、去噪等）能够大大提高训练速度，提升训练效果。基于此，YOLOv2 对每一层输入的数据都进行批量标准化，这样网络就不需要每层都去学数据的分布，收敛会变得更快。

在卷积或池化之后，激活函数之前，对每个数据输出进行标准化，实现方式如下图所示：

<img src="D:\tool\typora\image\image-20240423154518669.png" alt="image-20240423154518669" style="zoom:50%;" />

**更高分辨率的分类器**

​		224×224 =====> 448×448

#### **使用基于anchor的目标边界框的预测**

​		在YOLOv1中，作者设计了端对端的网路，直接对边界框的位置（x, y, w, h）进行预测。这样做虽然简单，但是由于没有类似R-CNN系列的推荐区域，所以网络在前期训练时非常困难，很难收敛。于是，自YOLOv2开始，引入了 Anchors box 机制。在 Faster R-CNN 算法中，是通过预测 bounding box 与 ground truth 的位置偏移值$t_x, t_y$，间接得到bounding box的位置。其公式如下:

<img src="D:\tool\typora\image\image-20240423160332237.png" alt="image-20240423160332237" style="zoom:33%;" />

​		这个公式是无约束的，预测的边界框很容易向任何方向偏移。因此，每个位置预测的边界框可以落在图片任何位置，这会导致模型的不稳定性。因此 YOLOv2 在此方法上进行了一点改变：预测边界框中心点相对于**该网格左上角坐标 $(C_x,C_y)$** 的相对偏移量，同时**为了将bounding box的中心点约束在当前网格中**，使用 sigmoid 函数将$t_x, t_y$归一化处理，将值约束在0-1，这使得模型训练更稳定，$\sigma (x)=Sigmoid(x)$。下图为 Anchor box 与 bounding box 转换示意图，其中蓝色的是要预测的bounding box，黑色虚线框是Anchor box。

<img src="D:\tool\typora\image\image-20240423203751477.png" alt="image-20240423203751477" style="zoom:33%;" />

​		YOLOv2 在最后一个卷积层输出 13×13 的 feature map，意味着一张图片被分成了13×13个网格。每个网格有5个anchor box来预测5个bounding box，每个bounding box预测得到5个值：$t_x, t_y, t_w, t_h和t_0$(类似yolov1中的confidence).引入Anchor Box 机制后，通过间接预测得到的 bounding box 的位置的计算公式为：

<img src="D:\tool\typora\image\image-20240423160828233.png" alt="image-20240423160828233" style="zoom:50%;" />

#### **Fine-Grained Features 细粒度特征**

​		细粒度特征，可理解为不同层之间的特征融合。YOLOv2通过添加一个**Passthrough Layer**，把高分辨率的浅层特征连接到低分辨率的深层特征（把特征堆积在不同Channel中）而后进行融合和检测，Passthrough层与ResNet网络的shortcut类似。具体操作是：先获取前层的26×26的特征图，将其同最后输出的13×13的特征图进行连接，而后输入检测器进行检测，以此来**提高对小目标的检测能力**。
<img src="D:\tool\typora\image\image-20240423161729789.png" alt="image-20240423161729789" style="zoom: 33%;" />



<img src="D:\tool\typora\image\image-20240423162148574.png" alt="image-20240423162148574" style="zoom:80%;" />

#### **Darknet-19**

​	19个卷积层。

<img src="D:\tool\typora\image\image-20240423162602227.png" alt="image-20240423162602227" style="zoom: 67%;" />

## 3、yolov3

​		2018年，在v2基础上进行了更新。特征提取部分采用darknet-53网络结构代替原来的darknet-19，利用特征金字塔网络结构实现了多尺度检测，分类方法使用逻辑回归代替了softmax，在兼顾实时性的同时保证了目标检测的准确性。

​		在YOLOv3中，作者不仅提供了darknet-53，还提供了轻量级的tiny-darknet。如果你想检测精度与速度兼具，可以选择darknet-53作为backbone；如果你希望达到更快的检测速度，精度方面可以妥协，那么tiny-darknet是你很好的选择。

yolov3**[网络结构](D:\tool\typora\image\image-20240423211813223.png)**

#### [Darknet-53](D:\tool\typora\image\image-20240423171211085.png)

53个卷积层，没有最大池化层。

![image-20240423211813223](D:\tool\typora\image\image-20240423211813223.png)

![image-20240423171211085](D:\tool\typora\image\image-20240423171211085.png)

​		特征图一大小是13*13，Predict one预测尺度较大的目标；特征图一大小是26\*26，Predict three预测中等的目标。特征图一大小是52\*52，Predict three预测较小的目标。

**多尺度预测**

​		为了能够预测多尺度的目标，YOLOv3 选择了三种不同shape的Anchors，同时每种Anchors具有三种不同的尺度，一共9种不同大小的Anchors。

<img src="D:\tool\typora\image\image-20240423202220449.png" alt="image-20240423202220449" style="zoom: 50%;" />

​		借鉴特征金字塔网的思想，YOLOv3设计了3种不同尺度的网络输出Y1、Y2、Y3，目的是预测不同尺度的目标。由于在每一个尺度网格都负责预测3个边界框，且COCO数据集有80个类。所以网络输出的张量应该是：N ×N ×[3∗(4 + 1 + 80)]。由下采样次数不同，得到的N不同，最终Y1、Y2、Y3的shape分别为：[13, 13, 255]、[26, 26, 255]、[52, 52, 255]。N为预测特征图的大小。

正负样本的匹配

​		针对每一个GT都会分配一个正样本，将与GT重合程度最大的视为正样本。如果重合超过阈值但不是最大的话也丢弃，剩余的样本为负样本。

**多标签分类**

​		YOLOv3在类别预测方面将YOLOv2的单标签分类改进为多标签分类，在网络结构中将YOLOv2中用于分类的softmax层修改为逻辑分类器。在YOLOv2中，算法认定一个目标只从属于一个类别，根据网络输出类别的得分最大值，将其归为某一类。然而在一些复杂的场景中，单一目标可能从属于多个类别。
​		为实现多标签分类就需要用逻辑分类器来对每个类别都进行二分类。逻辑分类器主要用到了sigmoid函数，它可以把输出约束在0到1，如果某一特征图的输出经过该函数处理后的值大于设定阈值，那么就认定该目标框所对应的目标属于该类。

eg:预测概率$[0.1,0.8,0.9]$，概率之和不为1

**损失函数**

​		对比YOLOv1中的损失函数很容易知道：位置损失部分并没有改变，仍然采用的是sum-square error的损失计算方法。但是置信度损失和类别预测均由原来的sum-square error改为了交叉熵的损失计算方法。对于类别以及置信度的预测，使用交叉熵的效果应该更好。

![image-20240423210230795](D:\tool\typora\image\image-20240423210230795.png)

<img src="D:\tool\typora\image\image-20240423210305369.png" alt="image-20240423210305369" style="zoom: 33%;" />

<img src="D:\tool\typora\image\image-20240423210336544.png" alt="image-20240423210336544" style="zoom:33%;" />

<img src="D:\tool\typora\image\image-20240423210348599.png" alt="image-20240423210348599" style="zoom:33%;" />

## yolov3 spp

**Mosaic图像增强**

​		Mosaic图像增强算法**将多张图片按照一定比例组合成一张图片**，**使模型在更小的范围内识别目标**。在该网络中通过将四张图片随机裁剪，再拼接到一张图片上作为训练数据，这种做法丰富了图片的背景，增加数据的多样性和目标个数，并且四张图片拼接在一起提高了batch_size。batch_size大小越大BN效果越好。

#### **SPP模块**

(和SPPnet中不同)，全称全称为Spatial Pyramid Pooling（空间金字塔池化结构）。

目的是为了**实现不同尺度的特征融合**。在经过SPP模块后，张量中channel应变为原来的4倍，B,H,W不变。

<img src="D:\tool\typora\image\image-20240423212459783.png" alt="image-20240423212459783" style="zoom: 50%;" />

由何凯明大神提出，主要是为了解决两个问题：

> 1.有效避免了R-CNN算法对图像区域剪裁、缩放操作导致的图像物体剪裁不全以及形状扭曲等问题。
>
> 2.解决了卷积神经网络对图像重复特征提取的问题，大大提高了产生候选框的速度，且节省了计算成本。

SPP优点

> 1.不管输入尺寸是怎样，SPP 可以产生固定大小的输出
>
> 2.使用多个窗口(pooling window)
>
> 3.SPP 可以使用同一图像不同尺寸(scale)作为输入, 得到同样长度的池化特征。
>
> 4.由于对输入图像的不同纵横比和不同尺寸，SPP同样可以处理，所以提高了图像的尺度不变(scale-invariance)和降低了过拟合(over-fitting)
>
> 5.实验表明训练图像尺寸的多样性比单一尺寸的训练图像更容易使得网络收敛(convergence)
>
> 6.SPP 对于特定的CNN网络设计和结构是独立的。(也就是说，只要把SPP放在最后一层卷积层后面，对网络的结构是没有影响的， 它只是替换了原来的pooling层)
>
> 7.不仅可以用于图像分类而且可以用来目标检测



各种Loss

**IoU Loss**

使用它作为损失函数会出现问题，其损失函数一般有两种表达式：

> 1.IoU loss = -ln(IoU)
>
> 2.IoU Loss = 1 - IoU	# 第二种比较常用

IoU Loss 可以更好地反映出重合程度；且具有尺度不变性，无论重叠地框是大是小，重叠占比一定 IoU 一样大 。但是当预测框和 GT 框不重叠时，损失为 0。

<img src="D:\tool\typora\image\image-20240423215016674.png" alt="image-20240423215016674" style="zoom:33%;" />

**GIoU**

DIoU要比GIou更加符合目标框回归的机制，**将目标与anchor之间的距离，重叠率以及尺度都考虑进去**，使得目标框回归变得更加稳定

<img src="D:\tool\typora\image\image-20240423215342819.png" alt="image-20240423215342819" style="zoom:33%;" />

​		上面公式的意思是：先计算两个框的最小闭包区域面积 Ac (同时包含了预测框和真实框的最小框的面积(蓝色))，再计算出IoU，再计算闭包区域中不属于两个框的区域占闭包区域的比重，u为并集的面积，最后用IoU减去这个比重得到GIoU。 在上图中绿色是真实目标边界框，红色是预测目标边界框。

优点：与IoU只关注重叠区域不同，**GIoU不仅关注重叠区域，还关注其他的非重合区域**，能更好的反映两者的重合度。

缺点：当两个预测框高宽相同，且处于同一水平面时，GIOU就退化为IOU。此外，GIOU和IOU还有两个缺点：收敛较慢、回归不够准确。

**DIoU**

<img src="D:\tool\typora\image\image-20240424094405544.png" alt="image-20240424094405544" style="zoom:50%;" />

​		其中$b和b^{gt}$分别代表了预测框和真实框的中心点，且分子代表的是计算两个中心点间的欧式距离(即直线距离)。 c代表的是能够同时包含预测框和真实框的最小闭包区域的对角线距离。

#### **CIoU**

<img src="D:\tool\typora\image\image-20240424094703427.png" alt="image-20240424094703427" style="zoom:33%;" />

α和v为长宽比，计算公式如上图所示：w、h和w(gt)、h(gt)分别代表预测框的高宽和真实框的高宽。 

#### **Focal loss**

​		Focal loss主要是**为了解决one-stage目标检测中正负样本比例严重失衡的问题**。该损失函数降低了大量简单负样本在训练中所占的权重，也可理解为一种困难样本挖掘。由于一张图像中能够匹配到目标的候选框（正样本）个数一般只有十几个或者几十个，而未被匹配的候选框（负样本）大概有数万个，在这些负样本中，大部分都是简单易分的，对训练网络起不到作用，但由于数量太多会淹没掉少量但有助于训练的样本。

​		在YOLOV3 SPP中，使用Focal Loss通过对损失函数计算中二值交叉熵损失添加因子γ来降低易分负样本的损失贡献。

<img src="D:\tool\typora\image\image-20240424101157360.png" alt="image-20240424101157360" style="zoom:50%;" />

​		y是样本的标签值，而p是模型预测某一个样本为正样本的概率，对于真实标签为正样本的样本，它的概率p越大说明模型预测的越准确，对于真实标签为负样本的样本，它的概率p越小说明模型预测的越准确。



## 4、yolov4

[网络结构](D:/tool/typora/image/image-20240506153727889.png)：

- **Backbone:** [CSPDarknet53](D:/tool/typora/image/image-20240506153850932.png)
- **Neck:** [SPP](D:/tool/typora/image/image-20240506154252523.png)，[PAN](D:\tool\typora\image\image-20240424105232005.png)
- **Head:** YOLOv3

![image-20240506153727889](D:/tool/typora/image/image-20240506153727889.png)

优化策略：

​		相比之前的`YOLOv3`，改进了下Backbone，在`Darknet53`中引入了`CSP`模块（来自CSPNet）。在Neck部分，采用了`SPP`模块（Ultralytics版的YOLOv3 SPP就使用到了）以及`PAN`模块（来自PANet）。Head部分没变还是原来的检测头。

**`CSP结构`**

减少网络的计算量以及对显存的占用，同时保证网络的能力不变或者略微提升。在`CSPNet`中是直接按照通道均分，但在`YOLOv4`网络中是通过两个`1x1`的卷积层来实现的。

<img src="D:/tool/typora/image/image-20240506154105365.png" alt="image-20240506154105365" style="zoom:33%;" />

**`PAN结构`**（Path Aggregation Network路径聚合网络）其实就是在`FPN`（从顶到底信息融合）的基础上加上了从底到顶的信息融合，如下图所示。

<img src="D:\tool\typora\image\image-20240424105232005.png" alt="image-20240424105232005" style="zoom:50%;" />

​		但YOLOv4的PAN结构和原始论文的融合方式又略有差异，如下图所示。图(a)是原始论文中的融合方式，即特征层之间融合时是直接通过相加的方式进行融合的，但在YOLOv4中是通过在通道方向Concat拼接的方式进行融合的。

**Eliminate grid sensitivity**	

​		v2,v3中采用基于anchor的目标边界框的预测，但在YOLOv4的论文中作者认为这样做不太合理，比如当真实目标中心点非常靠近网格的左上角点($\sigma(t_x)$和$\sigma(t_y)$应该趋近与0)，或者右下角点($\sigma(t_x)$和$\sigma(t_y)$应该趋近与1)时，网络的预测值需要负无穷或者正无穷时才能取到，而这种很极端的值网络一般无法达到。为了解决这个问题，作者引入了一个大于1的缩放系数${\rm scale}_{xy}$。

<img src="D:\tool\typora\image\image-20240424110100984.png" alt="image-20240424110100984" style="zoom:50%;" />

通过引入这个系数，网络的预测值能够很容易达到0或者1，现在比较新的实现方法包括YOLOv5都将$ {\rm scale}_{xy}$设置为2。

通过引入缩放系数scale以后，x在同样的区间内，y的取值范围更大，或者说y对x更敏感了。

##### IoU threshold（正样本匹配）

​		在`YOLOv3`中针对每一个GT都只分配了一个Anchor。但在`YOLOv4`包括之前讲过的YOLOv3 SPP以及`YOLOv5`中一个GT可以同时分配给多个Anchor，它们是直接使用Anchor模板与GT Boxes进行粗略匹配，然后在定位到对应cell的对应Anchor。

​		但在YOLOv4以及YOLOv5中关于匹配正样本的方法又有些许不同。主要原因在于Eliminate grid sensitivity中提到的缩放因子$scale_{xy}$，通过缩放后网络预测中心点的偏移范围已经从原来的(0, 1)调整到了(-0.5, 1.5)。所以对于同一个GT Boxes可以分配给更多的Anchor，即正样本的数量更多了。



## 5、yolov5

特点：适用于移动端部署，模型小，速度快。YOLOv5针对不同大小（`n`, `s`, `m`, `l`, `x`）的网络整体架构都是一样的，只不过会在每个子模块中采用不同的深度和宽度，分别应对`yaml`文件中的`depth_multiple`和`width_multiple`参数

- **Backbone**: `New CSP-Darknet53`
- **Neck**: [SPPF](D:/tool/typora/image/image-20240506152559520.png), `New CSP-PAN`
- **Head**: `YOLOv3 Head`

yolov5l[网络结构](D:\tool\typora\image\yolov5.png)：

![yolov5](D:\tool\typora\image\yolov5.png)

**Backbone**部分

`Focus`模块:

​	将每个`2x2`的相邻像素划分为一个`patch`，然后将每个`patch`中相同位置（同一颜色）像素给拼在一起就得到了4个`feature map`，然后在接上一个`3x3`大小的卷积层。这和直接使用一个`6x6`大小的卷积层等效。



<img src="D:/tool/typora/image/image-20240506152412463.png" alt="image-20240506152412463" style="zoom: 33%;" />

作用: 可以使信息不丢失的情况下提高计算力。

不足：Focus 对某些设备不支持且不友好，开销很大。

​	YOLOv5在`v6.0`版本后相比之前版本有一个小的改动，把网络的第一层（原来是`Focus`模块）换成了一个`6x6`大小的卷积层。**两者在理论上其实等价的**，但是对于现有的一些GPU设备（以及相应的优化算法）使用`6x6`大小的卷积层比使用`Focus`模块更加高效。

**Neck**部分

首先是将`SPP`换成成了`SPPF`，两者的作用是一样的，但后者效率更高。

`SPP结构：`目的是为了实现不同尺度的特征融合

<img src="D:/tool/typora/image/image-20240506152536235.png" alt="image-20240506152536235" style="zoom:33%;" />

##### SPPF结构

<img src="D:/tool/typora/image/image-20240506152559520.png" alt="image-20240506152559520" style="zoom: 33%;" />

**Neck**部分另外一个不同点就是`New CSP-PAN`了，在YOLOv4中，**Neck**的`PAN`结构是没有引入`CSP`结构的，但在YOLOv5中作者在`PAN`结构中加入了`CSP`。每个`C3`模块里都含有`CSP`结构。在**Head**部分，YOLOv3, v4, v5都是一样的

**数据增强策略**

- **Mosaic**：将四张图片拼成一张图片

- **Copy paste**，将部分目标随机的粘贴到图片中，前提是数据要有`segments`数据才行，即每个目标的实例分割信息。

  <img src="D:/tool/typora/image/image-20240506152830074.png" alt="image-20240506152830074" style="zoom:33%;" />

- **Random affine(Rotation, Scale, Translation and Shear)**：随机进行仿射变换，但根据配置文件里的超参数发现只使用了`Scale`和`Translation`即缩放和平移。

- **MixUp**：就是将两张图片按照一定的透明度融合在一起。代码中只有较大的模型才使用到了`MixUp`，而且每次只有10%的概率会使用到。

- **Albumentations**：主要是做些滤波、直方图均衡化以及改变图片质量等等，我看代码里写的只有安装了`albumentations`包才会启用。默认不启用。

- **Augment HSV(Hue, Saturation, Value)**，随机调整色度，饱和度以及明度。

- **Random horizontal flip**，随机水平翻转。



**训练策略**

- Multi-scale training(0.5~1.5x)，多尺度训练，假设设置输入图片的大小为 $640 \times 640$，训练时采用尺寸是在 $0.5 \times 640 \sim 1.5 \times 640$之间随机取值，注意取值时取得都是32的整数倍（因为网络会最大下采样32倍）。
- AutoAnchor(For training custom data)，训练自己数据集时可以根据自己数据集里的目标进行重新聚类生成Anchors模板。
- Warmup and Cosine LR scheduler，训练前先进行Warmup热身，然后在采用Cosine学习率下降策略。
- EMA(Exponential Moving Average)，可以理解为给训练的参数加了一个动量，让它更新过程更加平滑。
- Mixed precision，混合精度训练，能够减少显存的占用并且加快训练速度，前提是GPU硬件支持。
- Evolve hyper-parameters，超参数优化，没有炼丹经验的人勿碰，保持默认就好。



**损失计算**

YOLOv5的损失主要由三个部分组成：

- Classes loss，分类损失，采用的是BCE loss，注意只计算正样本的分类损失。

- Objectness loss，obj置信度损失，采用的依然是BCE loss，注意这里的obj指的是网络预测的目标边界框与GT Box的CIoU。这里计算的是所有样本的obj损失。
- Location loss，定位损失，采用的是CIoU loss，注意只计算正样本的定位损失。

![image-20240506155722060](D:/tool/typora/image/image-20240506155722060.png)

## 6、yolov6

[网络结构](D:/tool/typora/image/image-20240507203734911.png)

- **Backbone**: 小模型：`RepBlock `块构建，大模型：`CSPStackRep`块构建
- **Neck**: 小模型：`RepPAN`，大模型：`CSPRepPAN`
- **Head**: `Efficient Decoupled Head`

在YOLOv6中，提出了 **两个缩放的可重参数化backbone和neck**，以适应不同尺寸的模型，以及一个 **具有混合通道(hybrid-channel)策略的高效解耦head(decoupled head)**。

![image-20240507170542160](D:/tool/typora/image/image-20240507170542160.png)

![image-20240507210204968](D:/tool/typora/image/image-20240507210204968.png)

**[backbone](D:/tool/typora/image/image-20240507204141530.png)**

​		backbone的设计中，多分支的网络(ResNet,DenseNet,GoogLeNet)相比单分支(ImageNet,VGG)的通常能够有更好的分类性能。但是，它通常伴随着并行性的降低，并导致推理延迟的增加。相反，普通单路径网络具有高并行性和较少内存占用的优点，从而导致更高的推理效率。参考RepVGG，提出了一种结构重参数化方法，将训练时多分支拓扑与推理时普通架构解耦，以实现更好的速度-准确度权衡。(yolov6s的backbone网络结构中s表示stride, o为outchannel, i为inchannel)

​		受以上启发设计了一个高效的重参数化的backbone，命名为 **EfficientRep**。对于**小型模型**，backbone的主要组成部分是训练阶段的 **RepBlock**，如图3(a)所示。在 推理阶段，每个RepBlock被转换为具有ReLU激活函数的3×3卷积层（表示为 **RepConv**）的堆叠，如图3(b)所示。通常，3×3卷积在主流GPU和CPU上高度优化，并且具有更高的计算密度。

<img src="D:/tool/typora/image/image-20240507211440139.png" alt="image-20240507211440139" style="zoom: 50%;" />

​		随着模型尺寸的进一步扩展，单路径普通网络的计算成本和参数数量呈指数增长。因此对于**大型模型**，修改了 **CSPStackRep Block**以构建中大型网络的backbone。 CSPStackRep Block由三个1×1卷积层和由两个具有残差连接的RepVGG block(训练阶段)或RepConv(推理阶段)组成的子块堆叠组成，如图3(c)所示。

**neck**

​		采用来自YOLOv4和YOLOv5的改进的 PAN结构作为我们检测neck的基础。此外，将YOLOv5中使用的CSPBlock替换为RepBlock（适用于小型模型）或CSPStackRep Block（用于大型模型），并相应调整宽度和深度。YOLOv6的neck表示为 **Rep-PAN**，结构图如下图所示：

![image-20240507195628845](D:/tool/typora/image/image-20240507195628845.png)

**head**

Efficient decoupled head：YOLOv5的检测头是一个耦合头，在分类和定位分支之间共享参数，而FCOS和YOLOX中的对应部分将两个分支解耦，并在每个分支中引入另外两个3×3卷积层以提高性能。

在YOLOv6中，采用了一种 **混合通道(hybrid-channel)策略**，以构建更高效的解耦头。具体而言，将中间3×3卷积层的数量减少到只有一个。head的宽度由backbone和neck的宽度乘数共同缩放。这些修改进一步降低了计算成本，以实现更低的推理延迟。

Anchor-free：anchor-free检测器因其更好的泛化能力和解码预测结果的简单性而脱颖而出。其后处理的时间成本显著降低。有两种类型的anchor-free检测器：point-based(YOLOX,FCOS)和keypoing-based(CenterNet)。在YOLOv6中，采用了 **anchor-free point-based**范式。

**损失函数**

**1、Classification Loss**

提高分类器的性能是优化检测器的关键部分。`Focal Loss` 修改了传统的交叉熵损失，以解决正负样本之间或难易样本之间的类别不平衡问题。为了解决训练和推理之间质量估计和分类的不一致使用，`Quality Focal Loss`（`QFL`）进一步扩展了`Focal Loss`，联合表示分类分数和分类监督的定位质量。而 `VariFocal Loss` (`VFL`) 源于 `Focal Loss`，但它不对称地对待正样本和负样本。通过考虑不同重要性的正负样本，它平衡了来自两个样本的学习信号。`Poly Loss` 将常用的分类损失分解为一系列加权多项式基。它在不同的任务和数据集上调整多项式系数，通过实验证明比交叉熵损失和`Focal Loss`损失更好。

在 `YOLOv6` 上评估所有这些高级分类损失，最终采用 `VFL`。

**2、Box Regression Loss**

框回归损失提供了精确定位边界框的重要学习信号。`L1 Loss` 是早期作品中的原始框回归损失。逐渐地，各种精心设计的框回归损失如雨后春笋般涌现，例如 `IoU-series` 损失和概率损失。

`IoU-series Loss IoU loss` 将预测框的四个边界作为一个整体进行回归。它已被证明是有效的，因为它与评估指标的一致性。`IoU`的变种有很多，如`GIoU`、`DIoU`、`CIoU`、`α-IoU`和`SIoU`等，形成了相关的损失函数。我们在这项工作中对 `GIoU`、`CIoU` 和 `SIoU` 进行了实验。并且`SIoU`应用于`YOLOv6-N`和`YOLOv6-T`，而其他的则使用`GIoU`。

`Probability Loss Distribution Focal Loss` (`DFL`) 将框位置的基本连续分布简化为离散化的概率分布。它在不引入任何其他强先验的情况下考虑了数据中的模糊性和不确定性，这有助于提高框定位精度，尤其是在`ground-truth`框的边界模糊时。在 `DFL` 上，`DFLv2` 开发了一个轻量级的子网络，以利用分布统计数据与真实定位质量之间的密切相关性，进一步提高了检测性能。然而，`DFL` 输出的回归值通常比一般框回归多 17 倍，从而导致大量开销。额外的计算成本阻碍了小型模型的训练。而 `DFLv2` 由于额外的子网络，进一步增加了计算负担。在实验中，`DFLv2` 在模型上为 `DFL` 带来了类似的性能提升。因此，只在 `YOLOv6-M/L` 中采用 `DFL`。

**3、Object Loss**

`Object loss` 最早是在 `FCOS` 中提出的，用于降低低质量边界框的得分，以便在后处理中将其过滤掉。它还被用于 `YOLOX` 以加速收敛并提高网络精度。作为像 `FCOS` 和 `YOLOX` 这样的`Anchor-free`框架，在 `YOLOv6` 中尝试过 `object loss`。不幸的是，它并没有带来很多积极的影响。

**YOLOv6 训练和推理示例**

```python
from ultralytics import YOLO

# Build a YOLOv6n model from scratch
model = YOLO('yolov6n.yaml')
# Display model information (optional)
model.info()
# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='coco8.yaml', epochs=100, imgsz=640)
# Run inference with the YOLOv6n model on the 'bus.jpg' image
results = model('path/to/bus.jpg')
```



## 7、yolov7

[网络结构](D:/tool/typora/image/image-20240508143637448.png)

![image-20240508143637448](D:/tool/typora/image/image-20240508143637448.png)

YOLOV7主要的贡献在于：

> 1.模型重参数化
> YOLOV7将模型重参数化引入到网络架构中，重参数化这一思想最早出现于REPVGG中。
> 2.标签分配策略
> YOLOV7的标签分配策略采用的是YOLOV5的跨网格搜索，以及YOLOX的匹配策略。
> 3.ELAN高效网络架构
> YOLOV7中提出的一个新的网络架构，以高效为主。
> 4.带辅助头的训练
> YOLOV7提出了辅助头的一个训练方法，主要目的是通过增加训练成本，提升精度，同时不影响推理的时间，因为辅助头只会出现在训练过程中。

**REP模块**

REP模块或RepConv被称为**重参数化卷积**。分为两个，一个是训练，一个deploy，也就是推理。

- 训练模块：它有三个分支。最上面的分支是3x3的卷积，用于特征提取。中间的分支是1x1的卷积，用于平滑特征。
  最后分支是一个Identity，不做卷积操作，直接移过来。
- 推理模块：包含一个3x3的卷积，stride(步长为1)。是由训练模块重参数化转换而来。

<img src="D:/tool/typora/image/image-20240508150224105.png" alt="image-20240508150224105" style="zoom: 33%;" />

**ELAN模块**

ELAN模块是一个高效的网络结构，它通过控制最短和最长的梯度路径，使网络能够学习到更多的特征，并且具有更强的鲁棒性——抗干扰能力。

![image-20240508150522633](D:/tool/typora/image/image-20240508150522633.png)

![image-20240508151206482](D:/tool/typora/image/image-20240508151206482.png)

E-ELAN

在ELAN的基础上采用了分组卷积。图d等价于下图。

<img src="D:/tool/typora/image/image-20240508151456518.png" alt="image-20240508151456518" style="zoom:50%;" />

**模型缩放**

类似于YOLOv5、Scale YOLOv4、YOLOX，一般是对depth、width或者module scale进行缩放，实现扩大或缩小baseline的目的。

![image-20240508162717238](D:/tool/typora/image/image-20240508162717238.png)

**引入了卷积重参化并进行了改进**

​		采用梯度传播路径来分析不同的重参化模块应该和哪些网络搭配使用。同时分析出RepConv中的identity破坏了ResNet中的残差结构和DenseNet中的跨层连接，因此作者做了改进，采用没有Identity连接的RepConv结构进行卷积重参数化。

<img src="D:/tool/typora/image/image-20240508162825161.png" alt="image-20240508162825161" style="zoom:50%;" />

**引入了辅助训练模块-coarse-to-fine（由粗到细）引导标签分配策略**

硬标签-软标签

常用的方式是图(c)所示，即辅助头和引导头各自独立，分别利用ground truth和它们（辅助头、引导头）各自的预测结果实现标签分配。YOLOV7算法中提出了利用引导头的预测结果作为指导，生成从粗到细的层次标签，将这些层次标签分别用于辅助头和引导头的学习，如下图（d）和（e）所示。

![image-20240508162854043](D:/tool/typora/image/image-20240508162854043.png)

## 8、yolov8

[网络结构](D:/tool/typora/image/222869864-1955f054-aa6d-4a80-aed3-92f30af28849.jpg)

YOLOv8 是一个 SOTA 模型，并没有直接将开源库命名为 YOLOv8，而是直接使用 ultralytics 这个词，原因是 ultralytics 将这个库定位为算法框架，而非某一个特定算法，一个主要特点是可扩展性。总而言之，ultralytics 开源库的两个主要优点是：

- **融合众多当前 SOTA 技术于一体**
- **未来将支持其他 YOLO 系列以及 YOLO 之外的更多算法**

![222869864-1955f054-aa6d-4a80-aed3-92f30af28849](D:/tool/typora/image/222869864-1955f054-aa6d-4a80-aed3-92f30af28849.jpg)





YOLOv8 也可以在 Python 环境中直接使用：

```python
from ultralytics import YOLO
# 加载模型
model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）
# 使用模型
model.train(data="coco8.yaml", epochs=3)  # 训练模型
metrics = model.val()  # 在验证集上评估模型性能
results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
# 调用摄像头
model = YOLO("yolov8n.pt")
result = model.predict(source="0", show=True)
```

yolov8的一些实用程序

**自动标注**

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(data='path/to/new/data',
    det_model='yolov8n.pt',
    sam_model='mobile_sam.pt',
    device="cuda",
    output_dir="path/to/save_labels",)
```

**获取边界框尺寸**

```python
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # Load pretrain or fine-tune model
# Process the image
source = cv2.imread('path/to/image.jpg')
results = model(source)
# Extract results
annotator = Annotator(source, example=model.names)
for box in results[0].boxes.xyxy.cpu():
    width, height, area = annotator.get_bbox_dimension(box)
    print("Bounding Box Width {}, Height {}, Area {}".format(
        width.item(), height.item(), area.item()))
```

**边界框的转换**

```python
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.ops import xywh2xyxy
```

注意事项

支持的格式：

```python
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.data.utils import VID_FORMATS
print(IMG_FORMATS)
>>> ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm')
print(VID_FORMATS)
>>> ("asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm")
```

[参数](D:/tool/typora/image/image-20240514160031036.png)

![image-20240514160031036](D:/tool/typora/image/image-20240514160031036.png)

性能

![image-20240924150459962](D:/tool/typora/image/image-20240924150459962.png)

## 9、yolov9



# 语义分割

## 简介

常见的分割任务：

1. 语义分割 (semantic segmentation) FCN，图c

2. 实例分割 (Instance segmentation) Mask R-CNN，图d

3. 全景分割 (Panoramic segmentation) Panoptic FPN

   <img src="D:\tool\typora\image\image-20240418105123837.png" alt="image-20240418105123837" style="zoom: 33%;" />

语义分割任务常见的数据集格式：

- PASCAL VOC

​		注意，在语义分割中对应的标注图像（.png）用PIL的Image.open()函数读取时，默认是P模式，即一个单通道的图像。在背景处的像素值为0，目标边缘处用的像素值为255(训练时一般会忽略像素值为255的区域)，目标区域内根据目标的类别索引信息进行填充，例如人对应的目标索引是15，所以目标区域的像素值用15填充。

![image-20240416151334054](D:\tool\typora\image\image-20240416151334054.png)

- MS COCO

![image-20240416151547223](D:\tool\typora\image\image-20240416151547223.png)



### 常见语义分割评价指标7

- Pixel Accuracy
- mean Accuracy
- **mean IoU**

![image-20240416151707470](D:\tool\typora\image\image-20240416151707470.png)

实例：

![image-20240416153121844](D:\tool\typora\image\image-20240416153121844.png)

## 卷积

### 转置卷积 Transposed Convolution

- 转置卷积不是卷积的逆运算
- 转置卷积也是卷积

转置卷积的主要就是起到上采样的作用，上采样（Upsampling）是一种增加数据空间分辨率的操作，通常用于将低分辨率的特征图（feature maps）转换为高分辨率的特征图。上采样的主要目的是扩大特征图的尺寸，它可以恢复网络中丢失的空间信息。

转置卷积的运算步骤可以归为以下几步：

- 在输入特征图元素间填充s-1行、列0（其中s表示转置卷积的步距）

- 在输入特征图四周填充k-p-1行、列0（其中k表示转置卷积的kernel_size大小，p为转置卷积的padding，注意这里的padding和卷积操作中有些不同）

- 将卷积核参数上下、左右翻转

- 做正常卷积运算（填充0，步距1）

  ![image-20240416160138964](D:\tool\typora\image\image-20240416160138964.png)

  |                        s=1, p=0, k=3                         |                        s=2, p=0, k=3                         |                        s=2, p=1, k=3                         |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | ![img](https://img-blog.csdnimg.cn/dbb10ea62b89456ca567eb69fd31d18b.gif) | ![img](https://img-blog.csdnimg.cn/94191375edb942a087c54173a1dd4e75.gif) | ![img](https://img-blog.csdnimg.cn/dc6050f7df5042f886054f16d8e522d1.gif) |

  ![image-20240416160604036](D:\tool\typora\image\image-20240416160604036.png)

滑动卷积效率低下，采用新的形式：

将卷积核转化为等效矩阵，将每个等效矩阵和输入相乘得到输出

![image-20240416162027411](D:\tool\typora\image\image-20240416162027411.png)

将输入feature map进行展平

![image-20240416162230805](D:\tool\typora\image\image-20240416162230805.png)

将等效矩阵进行展平

![image-20240416162317002](D:\tool\typora\image\image-20240416162317002.png)

将展平后的矩阵进行相乘

![image-20240416162342888](D:\tool\typora\image\image-20240416162342888.png)

已知C和O可否还原I？两侧都乘以C的转置

![image-20240416162621216](D:\tool\typora\image\image-20240416162621216.png)

为何需要将卷积核翻转处理的原因

![image-20240416163857627](D:\tool\typora\image\image-20240416163857627.png)



### 膨胀卷积 Dilated Convolution

- 增大感受野

- 保持原输入特征图W、H

- **适用情况**：在图像需要全局信息、语音文本需要较长的sequence信息依赖的问题中，都能很好的应用dilated conv

  ​		保持参数个数不变的情况下增大了卷积核的感受野，让每个卷积输出都包含较大范围的信息；同时它可以保证输出的特征映射（feature map）的大小保持不变。一个扩张率为2的3×3卷积核，感受野与5×5的卷积核相同，但参数数量仅为9个，是5×5卷积参数数量的36%。

  参数：k = 3, r = 2, p = 0, s = 1（r为膨胀系数，当r=1时就为普通卷积）

  

  <img src="https://img-blog.csdn.net/20181007200558639?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTI3NDgwMQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="å¨è¿éæå¥å¾çæè¿°" style="zoom:67%;" />

#### gridding effect现象

Layer4并没有利用到所有信息（感受野13×13）

![image-20240416203804651](D:\tool\typora\image\image-20240416203804651.png)

采用不同的膨胀系数

![image-20240416204050693](D:\tool\typora\image\image-20240416204050693.png)

使用普通的卷积核

![image-20240416205028632](D:\tool\typora\image\image-20240416205028632.png)

连续使用膨胀卷积时如何设置膨胀系数？

#### Hybrid Dilated Convolution (HDC)设计准则

![image-20240416210253450](D:\tool\typora\image\image-20240416210253450.png)

另外公约数不能大于1（因此膨胀系数都有1）

## 网络模型

### 1、FCN

全称Fully Convolutional Networks for Semantic Segmentation，首个端对端的针对像素级预测的全卷积网络。

网络结构：

![image-20240416195121012](D:\tool\typora\image\image-20240416195121012.png)

![image-20240416194758518](D:\tool\typora\image\image-20240416194758518.png)

![image-20240416194910508](D:\tool\typora\image\image-20240416194910508.png)

### 2、DeepLab V1

- 速度更快，论文中说是因为采用了膨胀卷积的原因，但fully-connected CRFs很耗时
- 准确率更高，相比之前最好的网络提升了7.2个点
- 模型结构简单，主要由DCNNs和CRFs联级构成

# Transformer

Transformer最早是Google在2017年的 <u>Attention Is Al You Need</u> 论文中提出，用于解决解决传统的序列到序列模型在处理可变长序列时遇到的问题。



**Query**：Query（**查询**）是一个特征向量，是你想要了解的信息，即我们可能想要注意什么。

**Keys：**每个输入元素有一个**键**，它也是一个特征向量。该特征向量粗略地描述了该元素“提供”什么，或者它何时可能很重要。键的设计应该使得我们可以根据Query来识别我们想要关注的元素。

**Values：**值是与每个词语相关的具体信息或特征。这个向量就是我们想要平均的向量。

**Score function：**评分函数，为了对想要关注的元素进行评分，我们需要指定一个评分函数`f`该函数将查询和键作为输入，并输出查询-键对的得分/注意力权重。它通常通过简单的相似性度量来实现，例如点积或MLP。

<img src="D:/tool/typora/image/image-20240522194722203.png" alt="image-20240522194722203" style="zoom: 67%;" />





# 少样本缺陷检测

#### **数据集**

mvtec

visa

#### **评价指标**

<img src="D:/tool/typora/image/image-20240816153311420.png" alt="image-20240816153311420" style="zoom: 25%;" />

**准确率**（Accuracy）的计算方式是所有分类对的样本数除以总样本数：

<img src="D:/tool/typora/image/image-20240816153408322.png" alt="image-20240816153408322" style="zoom:50%;" />

**召回率**（recall）指实际为异常样本中有多少被预测为异常，也被称为查全率、真阳性率（TPR）：

<img src="D:/tool/typora/image/image-20240816153452647.png" alt="image-20240816153452647" style="zoom: 50%;" />

**精确率**（Precision）表示预测为异常的样本中有多少是实际为异常样本，也被称为查准率：

<img src="D:/tool/typora/image/image-20240816154458644.png" alt="image-20240816154458644" style="zoom:50%;" />

**误检率**（FPR）表示有多少负样本被错误地预测为正样本：

<img src="D:/tool/typora/image/image-20240816154557347.png" alt="image-20240816154557347" style="zoom:50%;" />

**ROC 曲线**——（Receiver Operating Characteristic Curve）是一种统计图表，用于描述分类器在不同阈值下的真阳性率（True Positive Rate, TPR）与假阳性率（False Positive Rate, FPR）之间的关系。在不同阈值下，以误检率（FPR）作为横坐标轴，召回率（TPR）作为纵坐标轴绘制的曲线。

<img src="D:/tool/typora/image/image-20240816192505552.png" alt="image-20240816192505552" style="zoom: 50%;" />

- TPR越大，FPR越小则模型的性能就越好；
- ROC曲线上的点是一个模型上取不同阈值产生的不同的结果；
- **理想的决策阈值是TPR越接近1，FPR越接近0。**即越接近上图中蓝色的点。

**AUROC**——Area Under the ROC是一种用于评估图像异常检测或缺陷检测模型性能的指标。AUROC 是 ROC 曲线下方的面积，其值介于 0 和 1 之间，值越大表示分类器的性能越好。

**Pixel-AUROC** ——在图像异常检测任务中，我们通常关心的是能否准确地识别出图像中的异常区域，而不是整个图像是否正常。因此，评估方法需要在像素级别上进行。这意味着每个像素都被视为一个分类任务，目标是判断该像素是否属于异常类别。

### PromptAD



##### **CLIP**

CLIP（Contrastive Language-Image Pre-Training）模型是一种多模态预训练神经网络，由OpenAI在2021年发布，是从自然语言监督中学习的一种有效且**可扩展**的方法。多模态——（Multimodal）是指涉及两种或更多不同类型的数据或信号的机器学习任务，eg图像和文本，音频和视频等等。

该模型的核心思想是使用大量图像和文本的配对数据进行预训练，以学习图像和文本之间的对齐关系。CLIP模型有两个模态，一个是文本模态，一个是视觉模态，包括两个主要部分：
> 1. Text Encoder：用于将文本转换为低维向量表示-Embeding。
> 2. Image Encoder：用于将图像转换为类似的向量表示-Embedding。
>

在预测阶段，CLIP模型通过计算文本和图像向量之间的**余弦相似度**来生成预测。这种模型特别适用于**零样本学习**任务，即模型不需要看到新的图像或文本的训练示例就能进行预测。CLIP模型在多个领域表现出色，如图像文本检索、图文生成等。

![image-20240816100821034](D:/tool/typora/image/CLIP.png)

one-shot时的在其他数据集时对比resnet50的性能

<img src="D:/tool/typora/image/image-20240816161056859.png" alt="image-20240816161056859" style="zoom:33%;" />

##### CoOp

Context Optimization上下文优化

##### CoCa

- **Coca**（或 CoCa）是一种基于Transformer架构的多模态预训练模型，它可以同时处理文本和图像数据。这种模型主要用于处理文本和图像的联合任务，旨在通过同时学习文本和图像数据来提高模型的理解能力和生成能力。

<img src="D:/tool/typora/image/coca.png">

### 单词

Args					 ——是用于解释函数参数的标记

PIL						——Python Imaging Library		tensor、numpy图像类型的转换

State-Of-The-Art ——最先进

OOM					——GPU显存不足

NLP					   ——自然语言处理( Natural Language Processing）

AIGC					 ——AI-generated content，即人工智能生成内容的领域

**backbone**			——**主干网络**，主干网络大多时候指的是提取特征的网络，其作用就是提取图片中的信息，共后面的网络使用。backbone主要决定特征表示能力，同时，其设计对推理效率有着至关重要的影响，因为它承担了大量的计算成本。

**neck**					——是放在backbone和head之间的，是为了更好的利用backbone提取的特征。neck用于将低层物理特征与高层语义特征聚合，然后在各级建立金字塔特征图。

**head**					——head是获取网络输出内容的网络，利用之前提取的特征，head利用这些特征，做出预测。head由多个卷积层组成，并根据neck部的多级特征预测最终检测结果。从结构的角度来看，它可以分为anchor-based和anchor-free，或者更确切地说，分为参数耦合head和参数解耦head。

GAP					——Global Average Pool全局平均池化，就是将某个通道的特征取平均值。说人话就是将某个通道的特征取平均值。

AP 		 			——Average Precision平均精确度。

**mAP** 				 ——Mean Average Precision的缩写，即**均值平均精度**。mAP = 所有类别的平均精度求和除以所有类别。

P						——precision即准确率，准确率表示预测样本中实际正样本数占所有正样本数的比例

R 					  ——recall即召回率，召回率表示预测样本中实际正样本数占所有预测的样本的比例

P-R曲线			——表示了召回率和准确率之间的关系，精度越高，召回率越低。

GT					——Ground True人工标注的真正的标注框

LLM				 ——Large Language Model大型语言模型。这是一种基于深度学习技术的自然语言处理模型，LLMs通过在海量文本数据上进行训练，学习语言的复杂模式、语法结构、语义以及上下文依赖，从而能够在各种自然语言处理任务中表现出色。

HTTP协议	   ——全称是超文本传输协议，英文是 Hypertext Transfer Protocol 。

TCP				——Transmission Control Protocol，传输控制协议。是一种面向连接的、可靠的、基于字节流的传输层通信协议

API				——Application Programming Interface，应用程序编程接口，是一组规则、协议和工具，用于构建软件应用程序。它定义了软件系统不同部分之间应该如何交互，使得开发人员可以更容易地创建功能丰富的应用程序。

### 操作：

nvidia-smi		查看英伟达显卡信息

pip list		  	查看该环境安装的包

dir()——打开文件

help()——帮助文件		(在jupyter中用 函数名??——更加详细的帮助文件)

## pytorch常用函数

#### **torch.Flatten()**

输入n维的torch矩阵，将指定的维度展平到一起，默认1(第二个位置)到后面所有

```python
input = torch.randn(32, 1, 5, 5)
# With default parameters
m = nn.Flatten()
output = m(input)
output.size()		# torch.Size([32, 25])

# With non-default parameters
m = nn.Flatten(0, 2)
output = m(input)
output.size()		# torch.Size([160, 5])
```

#### torch.transpose()

输入n维torch张量，将指定的两个维度dim0和dim1位置交换

```python
x = torch.randn(2, 3)
# tensor([[ 1.0028, -0.9893,  0.5809],
#         [-0.1669,  0.7299,  0.4942]])
torch.transpose(x, 0, 1)
# tensor([[ 1.0028, -0.1669],
#         [-0.9893,  0.7299],
#         [ 0.5809,  0.4942]])
```

#### **torch.stack()**

创建新维度,**沿新维度连接**一系列张量,所有张量都需要具有相同的大小。

```python
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.3367,  0.1288,  0.2345],
        [ 0.2303, -1.1229, -0.1863]])
>>> x = torch.stack((x, x)) # same as torch.stack((x, x), dim=0)
>>> x
tensor([[[ 0.3367,  0.1288,  0.2345],			# [`2`, 2, 3]
         [ 0.2303, -1.1229, -0.1863]],
        [[ 0.3367,  0.1288,  0.2345],
         [ 0.2303, -1.1229, -0.1863]]])
>>> x = torch.stack((x, x), dim=1)
tensor([[[ 0.3367,  0.1288,  0.2345],			# [2, `2`, 3]
         [ 0.3367,  0.1288,  0.2345]],
        [[ 0.2303, -1.1229, -0.1863],
         [ 0.2303, -1.1229, -0.1863]]])
>>> x = torch.stack((x, x), dim=2)
tensor([[[ 0.3367,  0.3367],					# [2, 3, `2`]
         [ 0.1288,  0.1288],
         [ 0.2345,  0.2345]],
        [[ 0.2303,  0.2303],
         [-1.1229, -1.1229],
         [-0.1863, -0.1863]]])
>>> x = torch.stack((x, x), dim=-1)
tensor([[[ 0.3367,  0.3367],					# [2, 3, `2`]
         [ 0.1288,  0.1288],
         [ 0.2345,  0.2345]],
        [[ 0.2303,  0.2303],
         [-1.1229, -1.1229],
         [-0.1863, -0.1863]]])

```

#### torch.cat()

在**给定维度中连接**给定的 `seq` 张量序列。所有张量必须具有相同的形状.

```python
>>> x = tensor([[ 0.6580, -1.0969, -0.4614],		# [2, 3]
        		[-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],				# [6, 3]
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 1)							# [2, 9]
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
```



#### torch.reshape()

将输入的n维张量改变成指定的新形状

```python
a = torch.arange(4.)		# tensor([0., 1., 2., 3.])
torch.reshape(a, (2, 2))
# tensor([[ 0.,  1.],
#         [ 2.,  3.]])
b = torch.tensor([[0, 1], [2, 3]])
torch.reshape(b, (-1,))		# 当为-1时会根据另一个维度自动补全
# tensor([ 0,  1,  2,  3])
b.reshape(4,-1)
# tensor([[0],
#         [1],
#         [2],
#         [3]])
```

#### torch.from_numpy()

将numpy数组转化为tensor，返回的张量和 `ndarray` 共享相同的内存。对张量的修改将反映在 `ndarray` 中，反之亦然。

```python
>>> a = numpy.array([1, 2, 3])
>>> t = torch.from_numpy(a)
>>> t		# tensor([ 1,  2,  3])
>>> t[0] = -1
>>> a		# array([-1,  2,  3])
```

#### torch.permute()

重新排列

- torch.permute(*input*, *dims*)

```python
>>> x = torch.randn(2, 3, 5)
>>> x.size()
torch.Size([2, 3, 5])
>>> torch.permute(x, (2, 0, 1)).size()
torch.Size([5, 2, 3])
```

#### torch.squeeze()

删除张量中大小为 1 的所有指定维度  。

- torch.squeeze(*input*, *dim=None*)

```python
>>> x = torch.zeros(2, 1, 2, 1, 2)
>>> x.size()		# torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x)
>>> y.size()		# torch.Size([2, 2, 2])
>>> y = torch.squeeze(x, 0)
>>> y.size()		# torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x, 1)
>>> y.size()		# torch.Size([2, 2, 1, 2])
>>> y = torch.squeeze(x, (1, 2, 3))
					# torch.Size([2, 2, 2])
```

#### torch.unsqueeze()

用于在输入张量input的指定维度dim上增加一个尺寸为1的新维度。这个操作常用于调整张量的形状。

dim (int) – 插入单例维度的索引

```python
x = torch.tensor([1, 2, 3, 4])	# torch.Size(4) 
torch.unsqueeze(x, 0)	# tensor([[ 1,  2,  3,  4]])
print(x.size())			# torch.Size(1, 4) 
torch.unsqueeze(x, 1)	
print(x.size())			# torch.Size(1, 1, 4) 
```

#### nn.Linear()

- nn.Linear(*in_features*, *out_features*, *bias=True*, *device=None*, *dtype=None*)



```python
>>> m = nn.Linear(20, 30)
>>> input = torch.randn(128, 20)
>>> output = m(input)
>>> print(output.size())
torch.Size([128, 30])
```

![image-20240531153156470](D:/tool/typora/image/image-20240531153156470.png)

X为输入的矩阵，为特征矩阵；W为权重，是模型想要学习的参数；b为o维的向量偏置，n为输入向量的行数，i为神经元的个数，o为输出神经元的个数。

#### nn.Softmax()

<img src="D:/tool/typora/image/image-20240523150817362.png" alt="image-20240523150817362" style="zoom: 80%;" />

- nn.Softmax(*dim=None*)

dim——计算 Softmax 的维度（沿着 dim 的每个切片的总和将为 1）,dim=1时为行和为1,dim=0时为列和为1；exp为e的次方

```python
>>> m = nn.Softmax(dim=1)
>>> input = torch.randn(2, 3)
>>> output = m(input)
```

#### nn.Dropout()

- nn.Dropout(*p=0.5*, *inplace=False*)

在训练期间，以 `p` 概率将输入张量的一些元素随机归零。

此外，在训练期间，输出会按**$\frac{1}{1-p}$**倍缩放。这意味着在评估过程中，模块仅计算恒等函数。

```python
m = nn.Dropout(p=0.5)
input = torch.randn(2, 2)
# tensor([[-2.5136, -1.2692],
#         [-0.8642,  0.2201]])
output = m(input)
# tensor([[-0.0000, -2.5383],
#         [-0.0000,  0.4401]])
```

#### 模型加载

`torch.jit.load` 和 `torch.load` 都是 PyTorch 中用于加载模型或数据的方法，但它们有一些重要的区别：

##### torch.jit.load

- **用途**：主要用于加载使用 `torch.jit.trace` 或 `torch.jit.script` 生成的 TorchScript 模型。
- **返回值**：返回一个 `torch.jit.ScriptModule` 对象，这是一个 TorchScript 编译过的模块，可以直接用于推理，而不需要知道原始模型的定义。
- **优势**：TorchScript 模型可以被序列化并部署到生产环境中，支持在没有 Python 解释器的环境下运行，也可以在移动设备和边缘设备上运行。

##### torch.load

- **用途**：用于加载 PyTorch 模型的权重、状态字典或其他数据。它可以加载 TorchScript 模型，但主要用途是加载普通的 PyTorch 模型状态字典。
- **返回值**：可以返回各种类型的对象，通常是一个字典，包含模型的权重、优化器的状态等。如果你加载的是一个 TorchScript 模型，它会返回一个 `torch.jit.ScriptModule` 对象。
- **灵活性**：`torch.load` 提供了更多的灵活性，因为它不仅可以加载 TorchScript 模型，还可以加载非 TorchScript 的模型状态字典，这使得它可以用于加载更广泛的模型数据。

##### 注意事项

- **安全性**：`torch.load` 默认情况下使用 Python 的 `pickle` 模块来反序列化数据，这可能带来安全风险。为了提高安全性，可以使用 `weights_only=True` 参数来限制可以加载的对象类型。
- **兼容性**：`torch.jit.load` 专门用于 TorchScript 模型，而 `torch.load` 可以用于加载更广泛的模型数据。

### 总结

**其他**

**[GLEE](https://github.com/FoundationVision/GLEE)：大规模图像和视频的通用对象基础模型**

![glee_func](D:/tool/typora/image/glee_func.gif)

**[YOLO-World](https://github.com/AILab-CVC/YOLO-World)：实时开放式词汇目标检测**

![yolo_arch](D:/tool/typora/image/yolo_arch.png)

# **孪生神经网络 （SNN）** 

A **siamese** neural network是一类神经网络架构。

<img src="D:/tool/typora/image/image-20240924100109597.png" alt="image-20240924100109597" style="zoom: 50%;" />

​		不同输入$X_1, X_2$通过统一$G_W$得到两个向量$G_W(X_1), G_W(X_2)$，计算两个向量之间的L1距离获得$E_W$。其中，两个network是两个**共享权值的网络**。如果左右两边不共享权值，而是两个不同的神经网络，叫做pseudo-siamese network，伪孪生神经网络。

训练

训练数据

<img src="D:/tool/typora/image/image-20240924095100241.png" alt="image-20240924095100241" style="zoom:33%;" />

训练网络

<img src="D:/tool/typora/image/image-20240924095008869.png" alt="image-20240924095008869" style="zoom: 33%;" />

损失函数：对比损失Contrastive Loss

**训练的目标是让两个相似的输入距离尽可能的小，两个不同类别的输入距离尽可能的大。**

![image-20240924100549681](D:/tool/typora/image/image-20240924100549681.png)

​		$D_w$代表两个样本特征$X_1和X_2 $的欧氏距离（二范数），P 表示样本的特征维数，Y 为两个样本是否匹配的标签，Y=1 代表两个样本相似或者匹配，Y=0 则代表不匹配，m 为设定的阈值，N 为样本个数。

​		这里设置了一个阈值ｍargin，表示我们只考虑不相似特征欧式距离在０～ｍargin之间的，当距离超过ｍargin的，则把其loss看做为０(即不相似的特征离的很远，其loss应该是很低的；而对于相似的特征反而离的很远，我们就需要增加其loss，从而不断更新成对样本的匹配程度)

预测：计算相似度

<img src="D:/tool/typora/image/image-20240924095223046.png" alt="image-20240924095223046" style="zoom:33%;" />

训练方式二

三元组损失

​		三元组损失是一种损失函数，其中我们将基线（锚点）输入与正（真）输入和负（假）输入进行比较。从基线 （锚点） 输入到正 （真） 输入的距离最小，从基线 （锚点） 输入到负 （假） 输入的距离最大化。

![image-20240924094634478](D:/tool/typora/image/image-20240924094634478.png)

在上面的方程中， $\alpha$ 是一个边际项，用于拉伸三元组中相似和不同对之间的距离。`Fa`、`Fp`、`Fn` 是锚点、正片和负片图像的特征嵌入。

<img src="D:/tool/typora/image/image-20240924094527712.png" alt="image-20240924094527712" style="zoom: 33%;" />

预测：计算距离

<img src="D:/tool/typora/image/image-20240924095343774.png" alt="image-20240924095343774" style="zoom:33%;" />

# 其他

CPU GPU NPU的区别

CPU——Central Processing Unit中央处理单元

GPU——Graphics Processing Unit图形处理单元

NPU——Neural Network Processing Unit嵌入式神经网络处理器。加速神经网络运算，擅长处理视频、图像等海量多媒体数据。

APU——Accelerated Processing Unit加速处理器。AMD公司推出，CPU+独立显卡核心。

显卡——Graphics card/Video card = 显存＋供电＋GPU

SoC——System On a Chip系统级芯片 = CPU + GPU + NPU(+声卡内存5g基带)

MPU——微处理器

FPU——浮点计算单元

TPU——张量处理单元

DPU——深度学习处理器

WPU——可穿戴处理器
