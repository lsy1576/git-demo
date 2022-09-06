import numpy as np
import random

##导入有误的话就直接下载anaconda 再导入相应的库函数

##1、创建数组，判断维度，读取形状和大小即(3,)or(3,2),和数组内元素的个数
##这里需要区分的是属于numpy数据库中的函数和矩阵本身性质的函数本身的话直接在赋值变量后追加就可以
array = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

array.ndim	 ##秩，即轴的数量或维度的数量,python中矩阵的维数可以理解为，确定一个元素需要几个数据，
                 ##上方数组.ndim=2,是因为每一个元素的确定均要2个数字。

array.shape	 ##数组的维度，对于矩阵，n 行 m 列
array.size	 ##数组元素的总个数，相当于 .shape 中 n*m 的值
array.dtype	 ##array对象的元素类型，实际中可以设置
array.itemsize   ##array对象中每个元素的大小，以字节为单位
array.flags	 ##array对象的内存信息
array.real	 ##array元素的实部
array.imag	 ##array元素的虚部
array.data	 ##包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。

#创建特殊数组
aa=np.zeros((3,2),np.uint8)
bb=np.array([2,23,4],dtype=np.int64) ##数组本身性质也可以是np.int64 np.float 16,32,64
cc=np.empty((3,2),dtype=np.uint8)
dd=np.ones((3,2),dtype=np.uint8)

a= np.arange(12).reshape((3,4))
b= np.linspace(1,10,5)                ##把1到10分成5段并进行线性排列成一维数组，同样也可以采用reshape函数
a1 = np.array([1, 2, 3, 4, 5], ndmin =  2)  #输出[[1 2 3 4 5]]
a2 = np.array([1, 2, 3, 4, 5])    #输出[1 2 3 4 5] 维度为1

##2、矩阵的运算 +,-,*,/,以及数学函数计算

a = np.arange(9, dtype = np.int64).reshape((3,3)) 
print ('第一个数组：')
print (a)
print ('\n')
print ('第二个数组：')
b = np.array([10,10,10])  
print(b)
print('\n')
print('两个数组相加：')   #a+b 对应元素相加减
print(np.add(a,b))
print('\n')
print('两个数组相减：')
print(np.subtract(a,b))   #a-b
print('\n')
print('两个数组相乘：')
print(np.multiply(a,b))   #a*b
print('\n')
print('两个数组相除：')   #a/b
print(np.divide(a,b))


# 通过乘 pi/180 转化为弧度  
print(np.sin(a*np.pi/180))   #sin函数是numpy所特有的sin(),cos(),tan()
#矩阵乘法不是对应元素相乘
a= np.array([[1,1],
             [0,1]])
b = np.arange(4).reshape((2,2))
c_dot = np.dot(a,b)
#特殊函数
d = np.random.random((2,4))  #产生随机的2*4的矩阵，矩阵内的元素为0-1之间的随机数

print(d)
print(np.sum(a,axis=1))  #axis=1按行进行累加求和  axis=0按列进行累加求和
print(np.min(a,axis=0))
print(np.max(a,axis=1))

#3、索引
A = np.arange(2,14).reshape((3,4))
print(np.argmax(A))   #找到最大值的索引
print(np.mean(A))
print(np.average(A))
print(np.median(A))
print(np.cumsum(A))  #不改变A矩阵的形状大小，元素等于前面的元素相加
print(np.diff(A))    #不改变A矩阵的形状大小，元素等于前面的元素相减求差
print(np.nonzero(A))   #非0的元素序列的行和列索引组合



A = np.arange(14,2,-1).reshape((3,4))
print(np.sort(A))
#矩阵的转置
B= np.transpose(A)
b= A.T
#矩阵的截取 小于最小全为最小阈值 大于最大都为最大阈值
print(np.clip(A,5,9))

A = np.arange(3,15).reshape((3,4))
B= A[:] # A[1,:]或A[:,1]  表示选取矩阵A的全部数据 ，选取A的第二行所有数据
#遍历列需要将原矩阵转置之后在遍历
for column in A.T:
    print(column)

#遍历矩阵中的元素需要矩阵平化
A.flatten()
for item in A.flat:
    print(item)
#矩阵合并与分割
A = np.array([1,1,1])
B = np.array([2,2,2])

C = np.vstack((A,B)) #垂直方向进行拼接   水平方向函数为hstack
#在列方向增加一个维度，此阵为2维矩阵
print(A[:np.newaxis].shape)
CC= np.concatenate((A,B,B,A),axis = 0) #在列方向上进行拼接


# 矩阵的分割神经网络常用
A = np.arange(12).reshape(3,4)
print(np.split(A,2,axis =1))

#矩阵的不等分割
print(np.array_split(A,3,axis=1))


#同一个如何加以区分 改变任意一个变量其余的均会改变 
a = np.arange(4)
b=a
c=a
d=b

#要想不改变，使得各个变量被关联必须
b = a.copy()  #deep copy


#广播机制  numpy.tile(A , reps)
#这里的 A 就是数组，reps 可以是一个数，一个列表、元组或者数组等，就是类数组的类型。先要理解准确，先把 A 当作一个块（看作一个整体，别分开研究每个元素）。
#（1）如果 reps 是一个数，就是简单的将 A 向右复制 reps - 1 次形成新的数组，就是 reps 个 A 横向排列：

import numpy as np

a = np.array([[1,2],[3,4]],dtype='i1')
print(a,'\n')
b = np.tile(a,2)  #向右复制，两个 A 横向排列
print(b)

#（2）如果 reps 是一个 array-like（类数组的，如列表，元组，数组）类型的，它有两个元素，如 [m , n]，实际上就是将 A 这个块变成 m * n 个 A 组成的新数组，有 m 行，n 列 A：

import numpy as np

a = np.array([[1,2],[3,4]],dtype='i1')
print(a,'\n')
b = np.tile(a,(2,3))  #2 * 3 个 A 组成新数组
print(b)
print('-----end-----')

