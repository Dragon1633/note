

# Python笔记





### 数据类型

![1697286079682](C:\Users\Dragon\AppData\Roaming\Typora\typora-user-images\1697286079682.png)

字符串：

首字母大写：变量名.title()		全大写：变量名.upper()		全小写：变量名.lower()

字符串中使用变量：前引号前加f

删除空白：变量名.rstrip()

**常用运算操作**

*\*表示乘方

//表示整除     eg：9//2=4

```python
# 幂运算，2的3次方，与2**3的作用一样
pow(2,3)
 
# 四舍五入，对x四舍五入，d是小数截取位数若不填默认为0，
# round(10.126, 2) 为 10.13，round(1.6)为2
round(x, d)
 
# 下列为求绝对值、最大值、最小值操作
abs(x)
max(x1, x2, x3)
min(x1, x2, x3)
 
# 求商余，返回输出商和余数(x//y, x%y)
# divmod(10, 3) 结果为 (3,1)
divmod(x, y)

# 转型
int(x)
float(x)
complex(x)
```

常量：将变量字母全部大写

输入：变量名=input()——获取的都为字符串，括号里可以增加一些提示性句子

输出：print("hello",end="")——输出不换行

占位：不同类型或变量一起输出时

![1697287684266](C:\Users\Dragon\AppData\Roaming\Typora\typora-user-images\1697287684266.png)

eg：![1697287734665](C:\Users\Dragon\AppData\Roaming\Typora\typora-user-images\1697287734665.png)

![1697289019592](C:\Users\Dragon\AppData\Roaming\Typora\typora-user-images\1697289019592.png)

浮点型数字精度控制：

![1697287806329](C:\Users\Dragon\AppData\Roaming\Typora\typora-user-images\1697287806329.png)







### 列表（List）

```python
cars = [ "bmw" , "audi" , "toyota" , "subaru" ]
```

列表元素索引从0开始，访问最后一个元素可以将索引指定为-1

在列表中添加元素：列表名.append(" ") 将括号中的元素添加到末尾

在列表中插入元素：列表名.insert(位置，“ ”)                位置：0，1，2，，，

在列表中删除元素： 1、del 列表名[位置]

​									2、列表名.pop()或列表名.pop(位置)		默认删除列表末尾的元素

​									3、列表名.remove("要删除的值")			删除元素的指定值(第一个)

对列表永久排序： 列表名.sort()							按首字母顺序进行排列

​								列表名.sort(reverse=True)	按与首字母顺序相反的顺序排列

对列表临时排序：print(sorted(列表名))			  输出一次顺序排序，不影响原始顺序

反转列表元素的排列顺序：变量名.reverse()	   永久性

确定列表的长度：len(列表名)			



|   函数或方法   |                            描述                             |
| :------------: | :---------------------------------------------------------: |
|  ls.append(x)  |                  在列表ls最后增加一个元素x                  |
|   ls.clear()   |                    删除列表ls中所有元素                     |
|   ls.copy()    |             生成一个新的列表，赋值ls中所有元素              |
| ls.insert(i,x) |                 在列表ls的第i位置增加元素x                  |
|   ls.pop(i)    | 将列表中第i位置元素去除并删除该元素，默认最后一个位置的元素 |
|  ls.remove(x)  |                将列表中出现的第一个元素x删除                |
|  ls.reverse()  |                    将列表ls中的元素反转                     |
|   Is.sort()    |                   对列表中的元素进行排序                    |
|   Is.count()   |                 返回 x 在列表中出现的次数。                 |

打印整个列表：for	新变量名	in	变量名:

​										print(新变量名)				   //利用for循环打印列表中所有变量

利用min(), max(), sum()算出数字列表的最小值，最大值，求和

切片：处理列表的部分元素——变量名[0:3]   处理索引为0，1，2的三个元素

​													  变量名[ :4]    从表头开始处理到第四个元素		变量名[2: ]    返回第三个元素到列表末

​														变量名[::-1]逆序返回



字符串类型及操作:	字符串由一对单引号或一对双引号表示（三引号也可以），字符串是字符的有序序列，可以对其中的字符进行索引。

```python
str = "请输入带有符号的温度值："
str[0]    # '请'
str[-1]   # ':'
str[:2]  # '请输'
str[1:5:3] # '输有'
str[::-1] # ':值度温的号符有带入输请'
print(str[3:-2]) # 带有符号的温度
```

输出占位符：

```python
# 若槽{}中内容为空，按默认顺序排列
# '2019-9-13:计算机C的CPU占用率为10%'
"{}:计算机{}的CPU占用率为{}%".format("2019-9-13", "C", 10)
```



##### 元组(tuple)

元组是一种序列类型，**一旦创建就不能被修改**。元祖可以理解为不可变的列表。使用小括号()或tuple()创建，可以使用也可以不使用小括号，元素间用逗号分隔。不可变的列表	元组名=(参数1，参数2)				只包含一个的元组元素后需要有逗号。

```python
tuple1 = (3,)   # 创建元组 （只有一个元素时，在元素后面加上逗号）
tuple2 = (1, 2, 3, 4, 5 )
list_name = ["python book","Mac","bile","kindle"]
tup = tuple(list_name) # 将列表转为元组
print(type(tup))    #  查看tup类型，并将结果打印出来
#报错操作
tuple1[0] = 2		# error
```

​			不能修改元组的元素，但可以给储存元组的变量赋值。eg:元组名=(新参数1，新参数2)√        元组名[0]=1×





### 字典（Dict）

```python
alien = {"color" : "green" , "point" : 5}
print(alien["color"])
```

用内建函数dict()创建字典				eg:字典名=dict()

len()计算个数，str()输出字典，type()返回输入的变量类型

get(key,"默认值")——返回指定键的值，如不存在返回默认值

items()——将字典中每个键值对转换成一个元组，并以列表的形式返回这些元组，其中每个元组包含键和其对应的值。通常用于遍历字典中的键值对。

![1697338085582](D:\tool\typora\image\1697338085582.png)

keys()方法用于返回字典中所有键组成的列表。

![1697338124315](D:\tool\typora\image\1697338124315.png)



![image-20240327101330942](D:\tool\typora\image\image-20240327101330942.png)

**字典推导式**

> { key_expr: value_expr for value in collection if condition }

```python
listdemo = ['Google','Runoob', 'Taobao']
# 将列表中各字符串值为键，各字符串的长度为值，组成键值对
>>> newdict = {key:len(key) for key in listdemo}	#{'Google': 6, 'Runoob': 6, 'Taobao': 6}
```



### 集合类型及操作

集合类型与数学中的集合概念一致，集合中每个元素都是唯一的不存在相同的元素，且无序（故无法更改）。由于集合中每个元素都是唯一的，故集合方法常用于去重。

注意：如果要创建一个空集合，你必须用 set() 而不是 {} 

```python
# 使用{}建立集合，注意，不能使用该方法建立空集合，否则为空字典
A = {"Python", 123, "你好"}  
 
# 使用set建立集合
B = set("pypy123")  
# {'1', 'y', '2', 'p', '2'} 无序且自动去除了重复的元素
```

![image-20240327100219202](D:\tool\typora\image\image-20240327100219202.png)

![image-20240327100140145](D:\tool\typora\image\image-20240327100140145.png)



### if语句

![1697335913732](D:\tool\typora\image\1697335913732.png)

用and和or检查多个条件，用in(not)判断特定值是否(不)在列表中

在 Python 中没有 switch...case 语句，但在 Python3.10 版本添加了 match...case

```python
match subject:
    case <pattern_1>:
        <action_1>
    case <pattern_2>:
        <action_2>
    case <pattern_3>:
        <action_3>
    case _:
        <action_wildcard>
```





### 循环语句

while循环语句

```python
while 判断条件(condition)：
    执行语句(statements)
```

 while循环使用else语句：如果 while 后面的条件语句为 false 时，则执行 else 的语句块。

无限循环你可以使用 CTRL+C 来中断循环。



for循环语句

```python
for <variable> in <sequence>:
    <statements>
else:
    <statements>
```

for...else：for...else 语句用于在循环结束后执行一段代码。



### 函数

函数定义：

函数的文档说明：

```python
def add():
    """
    说明
    """
    sum=20
    return sum

print("sum")
```

默认参数：def printinfo( name, age = 35 ):

**不定长参数**

加了星号 * 的参数会以元组(tuple)的形式导入，存放所有未命名的变量参数。

```python
def printinfo( arg1, *vartuple ):
"打印任何传入的参数"
    print ("输出: ")
    print (arg1)
    print (vartuple)

调用printinfo 函数
printinfo( 70, 60, 50 )
```

```python
输出: 
70
(60, 50)
```

加了两个星号 ** 的参数会以字典的形式导入:

```py
def printinfo( arg1, **vardict ):
''''''
printinfo(1, a=2,b=3)
输出: 
1
{'a': 2, 'b': 3}
```

单独出现星号 *，则星号 * 后的参数必须用关键字传入:

```python
def f(a,b,*,c):
f(1,2,c=3)
```

**关键字参数**

使用关键字参数允许函数调用时参数的顺序与声明时不一致

```python
def printme( str ):
   print (str)
   return
printme( str = "菜鸟教程")
```

**lambda:匿名函数**

lambda 函数是一种小型、匿名的、内联函数，它可以具有任意数量的参数，但只能有一个表达式。

```python
lambda [arg1 [,arg2,.....argn]]:expression
eg：
x = lambda a, b : a * b
print(x(5, 6))		# 30
```



在 python 中，类型属于对象，对象有不同类型的区分，变量是没有类型的

```python
a=[1,2,3]
a="Runoob"
```

[1,2,3] 是 List 类型，"Runoob" 是 String 类型，而变量 a 是没有类型，她仅仅是一个对象的引用（一个指针），可以是指向 List 类型对象，也可以是指向 String 类型对象。

### 类

**类(Class):** 用来描述具有相同的属性和方法的对象的集合。它定义了该集合中每个对象所共有的属性和方法。对象是类的实例。

**方法：**类中定义的函数。

**类变量：**类变量在整个实例化的对象中是公用的。类变量定义在类中且在函数体之外。类变量通常不作为实例变量使用。

**数据成员：**类变量或者实例变量用于处理类及其实例对象的相关的数据。

**实例化：**创建一个类的实例，类的具体对象。

**对象：**通过类定义的数据结构实例。对象包括两个数据成员（类变量和实例变量）和方法



**\_\_init\_\_:**类有一个名为 __init__() 的特殊方法（**构造方法**），该方法在类实例化时会自动调用，self 代表类的实例，而非类。self 是一个惯用的名称，用于表示类的实例（对象）自身。它是一个指向实例的引用，使得类的方法能够访问和操作实例的属性。当你定义一个类，并在类中定义方法时，第一个参数通常被命名为 self。

```python
#类定义
class people:
    #定义基本属性
    name = ''
    age = 0					#定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0
    #定义构造方法
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self.__weight = w
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name,self.age))
 
# 实例化类
p = people('runoob',10,30)
p.speak()

# 继承
class man(people):
```

**\_\_call\_\_:**call()的本质是将一个类变成一个函数（使这个类的实例可以像函数一样调用）

方法重写

```python
class Parent:        # 定义父类   
    def myMethod(self):      
        print ('调用父类方法')  

class Child(Parent): # 定义子类   
    def myMethod(self):      
        print ('调用子类方法')  
        
c = Child()          			# 子类实例         
c.myMethod()         			# 子类调用重写方法 
super(Child,c).myMethod() 		#用子类对象调用父类已被覆盖的方法

#输出为：
调用子类方法
调用父类方法
```

类属性与方法

**__private_attrs**：两个下划线开头，声明该属性（方法）为私有，不能在类的外部被使用或直接访问。在类内部的方法中使用时 **self.__private_attrs**。

类的专有方法：

- **__init__ :** 构造函数，在生成对象时调用
- **__del__ :** 析构函数，释放对象时使用
- **__repr__ :** 打印，转换
- **__setitem__ :** 按照索引赋值
- **__getitem__:** 按照索引获取值
- **__len__:** 获得长度
- **__cmp__:** 比较运算
- **__call__:** 函数调用
- **__add__:** 加运算
- **__sub__:** 减运算
- **__mul__:** 乘运算
- **__truediv__:** 除运算
- **__mod__:** 求余运算
- **__pow__:** 乘方

### 文件的使用

文件是存储在辅助器上的数据序列，文件展现形态主要有两种：**文本文件**和**二进制文件**。文本文件是由单一特定编码组成的文件，如UTF-8编码。

![image-20240327101617223](D:\tool\typora\image\image-20240327101617223.png)

**文件内容的读取**：示例中文本只有一行文字“中国是一个伟大的国家”

|       操作方法       |                             描述                             |
| :------------------: | :----------------------------------------------------------: |
|   f.read(size=-1)    | 读入全部内容，如果给出参数，读入前size长度>>>s = f.read(2)中国 |
| f.readline(size=-1)  | 读入一行内容，如果给出参数，读入该行前size长度>>>s = f.readline()中国是一个伟大的国家 |
| f.readlines(hint=-1) | 读入文件所有行，以每行为元素形成列表，如果给出参数，读入前hint行>>>s = readlines()['中国是一个伟大的国家 '] |

**文件内容的写入**

|      操作方法       |                             描述                             |
| :-----------------: | :----------------------------------------------------------: |
|     f.write(s)      | 向文件写入一个字符串或字符流>>>f.write("中国是一个伟大的国家") |
| f.writelines(lines) | 将一个元素全为字符串的列表写入文件>>>ls = ["中国", "法国", ''美国"]>>>f.writelines(ls) |
|   f.seek(offset)    | 改变当前文件操作指针的位置，offset含义如下：0-文件开头； 1-当前位置； 2-文件结尾>>>f.seek(0) # 回到文件开头 |

```python
# 文件的逐行操作
fname = input("请输入要打开的文件名称")
fo = open(fname, "r")
for line in fo.readlines():  # 一次读入，分行处理
    print(line)
fo.close
 
 
fname = input("请输入要打开的文件名称")
fo = open(fname, "r")
for line in fo:  # 分行读入，逐行处理
    print(line)
fo.close
```

### 迭代器

迭代是访问集合元素的一种方式，迭代器是一个可以记住遍历的位置的对象。迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。

迭代器有两个基本的方法：**iter()** 和 **next()**。字符串，列表或元组对象都可用于创建迭代器。

```python
list = [1,2,4,5]
it = iter(list)    # 创建迭代器对象
for x in it:
    print(x, end=" ")  # 1 2 4 5
#或者
try:
    print(next(it), end=" ")	# 1 2 4 5
except StopIteration:
    pass
```

**生成器**

使用了 **yield** 的函数被称为生成器（generator）。**yield** 是一个关键字，用于定义生成器函数，生成器函数是一种特殊的函数，可以在迭代过程中逐步产生值，而不是一次性返回所有结果。

当在生成器函数中使用 **yield** 语句时，函数的执行将会暂停，并将 **yield** 后面的表达式作为当前迭代的值返回。每次调用生成器的 **next()** 方法或使用 **for** 循环进行迭代时，函数会从上次暂停的地方继续执行，直到再次遇到 **yield** 语句。这样，生成器函数可以逐步产生值，而不需要一次性计算并返回所有结果。

```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1
 
# 创建生成器对象
generator = countdown(5)
 
# 通过迭代生成器获取值
print(next(generator))  # 输出: 5
print(next(generator))  # 输出: 4
print(next(generator))  # 输出: 3
 
# 使用 for 循环迭代生成器
for value in generator:
    print(value)  # 输出: 2 1
```



### 线程

Python3 线程中常用的两个模块为：

- **_thread**
- **threading(推荐使用)**

多线程类似于同时执行多个不同程序，多线程运行有如下优点：

- 使用线程可以把占据长时间的程序中的任务放到后台去处理。
- 用户界面可以更加吸引人，比如用户点击了一个按钮去触发某些事件的处理，可以弹出一个进度条来显示处理的进度。
- 程序的运行速度可能加快。
- 在一些等待的任务实现上如用户输入、文件读写和网络收发数据等，线程就比较有用了。在这种情况下我们可以释放一些珍贵的资源如内存占用等等。

**_thread模块**

```python
import _thread
_thread.start_new_thread ( function, args[, kwargs] )
```

参数说明:

- function - 线程函数。
- args        - 传递给线程函数的参数,必须是个tuple类型。
- kwargs   - 可选参数。

**threading模块**

threading 模块除了包含 _thread 模块中的所有方法外，还提供的其他方法：

- **threading. current_thread()**: 返回当前的线程变量。
- **threading.enumerate()**: 返回一个包含正在运行的线程的列表。正在运行指线程启动后、结束前，不包括启动前和终止后的线程。
- **threading.active_count()**: 返回正在运行的线程数量，与 len(threading.enumerate()) 有相同的结果。
- **threading.Thread(target, args=(), kwargs={}, daemon=None)**
  - 创建`Thread`类的实例。
  - `target`：线程将要执行的目标函数。
  - `args`：目标函数的参数，以元组形式传递。
  - `kwargs`：目标函数的关键字参数，以字典形式传递。
  - `daemon`：指定线程是否为守护线程。

- threading.Thread 类提供了以下方法与属性:
- 1. **`__init__(self, group=None, target=None, name=None, args=(), kwargs={}, \*, daemon=None)`：**
     - 初始化`Thread`对象。
     - `group`：线程组，暂时未使用，保留为将来的扩展。
     - `target`：线程将要执行的目标函数。
     - `name`：线程的名称。
     - `args`：目标函数的参数，以元组形式传递。
     - `kwargs`：目标函数的关键字参数，以字典形式传递。
     - `daemon`：指定线程是否为守护线程。
  2. **`start(self)`：**
     - 启动线程。将调用线程的`run()`方法。
  3. **`run(self)`：**
     - 线程在此方法中定义要执行的代码。
  4. **`join(self, timeout=None)`：**
     - 等待线程终止。默认情况下，`join()`会一直阻塞，直到被调用线程终止。如果指定了`timeout`参数，则最多等待`timeout`秒。
  5. **`is_alive(self)`：**
     - 返回线程是否在运行。如果线程已经启动且尚未终止，则返回`True`，否则返回`False`。
  6. **`getName(self)`：**
     - 返回线程的名称。
  7. **`setName(self, name)`：**
     - 设置线程的名称。
  8. **`ident`属性：**
     - 线程的唯一标识符。
  9. **`daemon`属性：**
     - 线程的守护标志，用于指示是否是守护线程。

**线程同步**

使用 Thread 对象的 Lock 和 Rlock 可以实现简单的线程同步，这两个对象都有 acquire 方法和 release 方法，对于那些需要每次只允许一个线程操作的数据，可以将其操作放到 acquire 和 release 方法之间，之间的代码一次只会有一个线程进行。

```python
import threading
import time

class myThread (threading.Thread):
    def __init__(self, threadID, name, delay):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.delay = delay
    def run(self):
        # 获取锁，用于线程同步
        threadLock.acquire()
        print ("开启线程： " + self.name)
        print_time(self.name, self.delay, 3)
        # 释放锁，开启下一个线程
        threadLock.release()

def print_time(threadName, delay, counter):
    while counter:
        time.sleep(delay)
        print ("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1

threadLock = threading.Lock()
threads = []

# 创建新线程
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)

# 开启新线程
thread1.start()
thread2.start()

# 添加线程到线程列表
threads.append(thread1)
threads.append(thread2)

# 等待所有线程完成
for t in threads:
    t.join()			# join()为等待指定线程结束再继续执行后续代码
print ("退出主线程")
```

**线程优先级队列（ Queue）**

Python 的 Queue 模块中提供了同步的、线程安全的队列类，这些队列都实现了锁原语，能够在多线程中直接使用，可以使用队列来实现线程间的同步。

包括FIFO（先入先出）队列Queue，LIFO（后入先出）队列LifoQueue，和优先级队列 PriorityQueue(优先返回权重最小的)。

Queue 模块中的常用方法:

- Queue.qsize() 返回队列的大小
- Queue.empty() 如果队列为空，返回True,反之False
- Queue.full() 如果队列满了，返回True,反之False
- Queue.full 与 maxsize 大小对应
- Queue.get([block[, timeout]])获取队列，timeout等待时间
- Queue.get_nowait() 相当Queue.get(False)
- Queue.put(item) 写入队列，timeout等待时间
- Queue.put_nowait(item) 相当Queue.put(item, False)
- Queue.task_done() 在完成一项工作之后，Queue.task_done()函数向任务已经完成的队列发送一个信号
- Queue.join() 实际上意味着等到队列为空，再执行别的操作

```python
import queue
import threading
import time

exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
    def run(self):
        print ("开启线程：" + self.name)
        process_data(self.name, self.q)
        print ("退出线程：" + self.name)

def process_data(threadName, q):
    while not exitFlag:
        queueLock.acquire()
        if not q.empty():
            data = q.get()
            queueLock.release()
            print("%s processing %s" % (threadName, data))
        else:
            queueLock.release()
        time.sleep(1)

threadList = ["Thread-1", "Thread-2", "Thread-3"]
nameList = ["One", "Two", "Three", "Four", "Five"]
queueLock = threading.Lock()
workQueue = queue.Queue(10)
# workQueue = queue.LifoQueue(10)
# workQueue = queue.PriorityQueue(10)
threads = []
threadID = 1

# 创建新线程
for tName in threadList:
    thread = myThread(threadID, tName, workQueue)
    thread.start()
    threads.append(thread)
    threadID += 1

# 填充队列
queueLock.acquire()
for word in nameList:
    workQueue.put(word)
queueLock.release()

# 等待队列清空
while not workQueue.empty():
    pass

# 通知线程是时候退出
exitFlag = 1

# 等待所有线程完成
for t in threads:
    t.join()
print ("退出主线程")
```



### 库函数

#### time库

time库是python中处理时间的标准库

```python
# 获取当前时间戳，计算机内部时间值，是一个浮点数
time.time()  # 1568360352.4165237
# 获取当前时间并以易读方式表示，返回是个字符串
time.ctime()  # 'Fri Sep 13 15:40:41 2019'
# 获取当前时间，表示为计算机可处理的时间格式
# time.struct_time(tm_year=2019, tm_mon=9, tm_mday=13, tm_hour=7, tm_min=41, tm_sec=30, tm_wday=4, tm_yday=256, tm_isdst=0)
time.gmtime() 
# 时间格式化，按照定义输出效果
# '2019-09-13 07:48:54'
t = time.gmtime()
time("%Y-%m-%d %H:%M:%S", t)
```

![image-20240327101906241](D:\tool\typora\image\image-20240327101906241.png)

```python
# 返回一个CPU级别的精确时间计数值，单位为秒，计数的起点不确定，所以要连续调用才有意义
# 通常连续两用两次计算其时间差
start = time.perf_counter()
end = time.perf_counter()
end - start  # 时间差
#获取当时时间
time = time.time()
 
# 程序等待s秒
time.sleep(s)
```

使用文本格式输出进度条示例：

```python
import time
 
scale = 50
print("执行开始".center(scale//2, "-"))
start = time.perf_counter()
for i in range(scale + 1):
    a = '*' * i
    b = '.' * (scale - i)
    c = (i/scale)*100
    dur = time.perf_counter() - start
    print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c,a,b,dur), end='')
    time.sleep(0.05)
print("\n" + "执行结束".center(scale//2, '-'))
# eg:100%[**************************************************->]2.53s
```

#### random库

random库是使用随机数的Python标准库

需要说一下random中的随机数种子seed，可以理解为生成随机序列的一种规则，相同的随机数种子生成的随机数顺序一样，使得随机实验具有可重复性。一般可不设置这个参数，这个参数默认为当前系统时间，所以能够保证生成的随机性。

```python
# 设置生成随机数种子
random.seed(10)
# 生成一个[0, 1)之间的随机小数
random.random() 
# 生成一个[a, b]之间的随机整数
random.randint(a, b) 
# 生成一个[m, n)之间以k为步长的随机整数
random.randrange(m, n[, k])  // random.randrange(10, 100, 10) 
# 生成一个k比特长的随机整数
random.getrandbits(k)
# 生成一个[a, b]之间的随机小数
random.uniform(a, b) 
# 从序列中随机选择一个元素
random.choice([1, 2, 3, 5]) 
# 将序列中的元素随机排序
random.shuffle([1, 5, 6, 8])
```

#### os库

os库提供通用的、基本的操作系统交互功能（包括windows、mac os以及linux）。os库包括路径操作、进程管理以及环境参数。

|              函数              |                             描述                             |
| :----------------------------: | :----------------------------------------------------------: |
|   **os.path.abspath**(path)    | 返回path在当前系统中的绝对路径>>>os.path.abspath("file.txt")'D:\\python\\file.txt' |
|     os.path.normpath(path)     | 归一化path的表示形式，统一用\\分隔路径>>>os.path.normpath("D://python//file.txt")'D:\\python\\file.txt' |
|     os.path.relpath(path)      |      返回当前程序与文件之间的相对路径（relative path）       |
|     os.path.dirname(path)      | 返回path中的目录名称>>>os.path.dirname("D://python//file.txt")"D://python" |
|     os.path.basename(path)     | 返回path中最后的文件名称>>>os.path.basename(”D://pyhton//file.txt“)"file.txt" |
| **os.path.join**(path, *paths) | 组合path和paths，返回一个路径字符串>>>os.path.join("D:/", "python/file.txt")'D:/python/file.txt' |
|    **os.path.exists**(path)    |       判断path对应文件或目录是否存在，返回True或False        |
|    **os.path.isfile**(path)    |      判断path所对应是否为已存在的文件，返回True或False       |
|    **os.path.isdir**(path)     |      判断path所对应是否为已存在的目录，返回True或False       |
|     os.path.getatime(path)     |            返回path对应文件或目录上一次的访问时间            |
|     os.path.getmtime(path)     |           返回path对应文件或目录最近一次的修改时间           |
|     os.path.getctime(path)     |               返回path对应文件或目录的创建时间               |
|     os.path.getsize(path)      |             返回path对应文件的大小，以字节为单位             |

os.system(command),执行程序或命令command

os库环境参数 ，获取或改变系统环境信息

|      函数      |                      描述                       |
| :------------: | :---------------------------------------------: |
| os.chdir(path) |             修改当前程序操作的路径              |
|  os.getcws()   |               返回程序的当前路径                |
| os.getlogin()  |            获得当前系统登录用户名称             |
| os.cpu_count() |              获得当前系统的CPU数量              |
| os.urandom(n)  | 获得n个字节长度的随机字符串，通常用于加解密运算 |



#### json库

| 常量、类或方法名 |                         注解                         |
| :--------------: | :--------------------------------------------------: |
|    json.dump     | 传入一个python对象，将其编码为json格式后存储到IO流中 |
|    json.dumps    | 传入一个python对象，将其编码为json格式后存储到str中  |
|    json.load     |    传入一个json格式的文件流，将其解码为python对象    |
|    json.loads    |     传入一个json格式的str，将其解码为python对象      |



#### math库



#### **copy库**



#### pathlib库

```python
from pathlib import Path		# 创建一个路径对象
# 获取上层目录
Path.cwd().parent
# 获取上上层目录
Path.cwd().parent.parent
# 获取绝对路径
Path().cwd()
Path().resolve()
# 创建目录
p = Path(r'E:\material\test')
p.mkdir(parents=True, exist_ok=True)	# parents：默认为 False，如果父目录不存在，会抛出异常，True 则创建这些目录。
# 创建文件
p = Path('E:/material/test1.txt')
p.touch(exist_ok=True)
# 文件/目录判断
Path('E:/material/pathlib用法').is_dir()		# True 	判断是否为文件夹
Path('E:/material/pathlib用法/txt文件.txt').is_file()	# True	判断是否为文件
Path('E:/material/error.txt').exists()		 # False  判断路径是否存在
# 使用 '/' 进行路径拼接
Path('E:/') / Path('/material/')
'E:/' / Path('/material/')
# 按照分隔符将文件路径分割
p.parts		# ('E:\\', 'material', 'pathlib用法')
# 获取文件/目录信息
# 获取文件/目录名
p = Path('E:/material/pathlib用法/excel文件.xlsx')
p.name		# 'excel文件.xlsx'
# 获取不包含后缀的文件名
p.stem		#'excel文件'
# 获取文件后缀名
p.suffix	# '.xlsx'
# 获取锚，最前面的部分
p.anchor	# 'E:\\'
# 获取文件/目录属性
p.stat()	# os.stat_result(st_mode=33206, st_ino=562949953976250, st_dev=503425376, st_nlink=1, st_uid=0, st_gid=0, st_size=6611, st_atime=1642130252, st_mtime=1642062067, st_ctime=1642066962)
p.stat().st_size	# 获取文件/目录大小，单位字节(B)
p.stat().st_mtime	# 获取文件/目录修改时间
p.stat().st_ctime	# 获取文件/目录创建时间
"""
上面获取的时间都是时间戳,通过 datetime 模块转成标准日期格式
from datetime import datetime
date = datetime.utcfromtimestamp(p.stat().st_ctime)
date.strftime("%Y-%m-%d %H:%M:%S")      # '2022-01-13 09:42:42'
"""
# 遍历目录
p = Path.cwd()
[path for path in p.iterdir()]		# 遍历目录下所有文件
# 重命名文件,当新命名的文件重复时，会抛出异常。
p = Path('E:/material/test1.txt')
new_name = p.with_name('test2.txt')
p.rename(new_name)		# WindowsPath('E:/material/test2.txt')
# 移动文件
p = Path('E:/material/test2.txt')
p.rename('E:/material/pathlib用法/test3.txt')
p.replace('E:/material/pathlib用法/test3.txt')	# 当新命名的文件重复时，直接覆盖。
# 删除文件
p.unlink(missing_ok=True)	# missing_ok=True 设置文件不存在不会抛出异常。
# 删除目录
p.rmdir()					# 删除目录，目录必须为空，否则抛出异常
```



#### argparse 模块

argparse是一个Python模块：命令行选项、参数和子命令解析器

1. argparse 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数
2. 其次，argparse 将弄清如何从 sys.argv 解析出那些参数。
3. argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。

```python
# 创建解析器
parser = argparse.ArgumentParser(description='Process some integers.')
# 添加参数
parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
# 解析参数
args = parser.parse_args()  # 获取所有参数
# 使用参数
print(args.integers)
```

```python
class argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True)
# description - 在参数帮助文档之前显示的文本（默认值：无）
ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
```

- name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。

- action - 当参数在命令行中出现时使用的动作基本类型。
- nargs - 命令行参数应当消耗的数目。
- const - 被一些 action 和 nargs 选择所需求的常数。
- default - 当参数未在命令行中出现时使用的值。
- type - 命令行参数应当被转换成的类型。
- choices - 可用的参数的容器。
- required - 此命令行选项是否可省略 （仅选项可用）。
- help - 一个此选项作用的简单描述。
- metavar - 在使用方法消息中使用的参数值示例。
- dest - 被添加到 parse_args() 所返回对象上的属性名。

#### pandas库

```python
import pandas as pd
# 导入CSV或者xlsx文件
df = pd.DataFrame(pd.read_csv('name.csv',header=1))
df = pd.DataFrame(pd.read_excel('name.xlsx'))
# 用pandas创建数据表
df = pd.DataFrame({"id":[1001,1002,1003,1004,1005,1006], 
                   "date":pd.date_range('20130102', periods=6),
                   "city":['Beijing ', 'SH', ' guangzhou ', 'Shenzhen', 'shanghai', 'BEIJING '],
                   "age":[23,44,54,32,34,32],
                   "category":['100-A','100-B','110-A','110-C','210-A','130-F'],
                   "price":[1200,np.nan,2133,5433,np.nan,4432]},
					columns =['id','date','city','category','age','price'])
# 数据表信息查看
df.info()		# 维度查看
# 查看所有列的列名
df.columns 			# 得到的是一个series对象
df.columns.values 	# 得到的是一个列表
# 定位表格中的指定元素 使用at，iat，loc，iloc。
print(df.at['行标签名', '列标签名'])
print(df.iat['行索引号', '列索引号'])
print(df.loc['行标签名', '列标签名'])
print(df.iloc[行索引数字, 列索引数字])
print(df.loc['行标签名1':'行标签名2', '列标签名1': '列标签名2'])
print(df.iloc[行索引数字1:行索引数字2, 列索引数字1:列索引数字2])
# 获取某一列的所有数值 
df["姓名"].values 
df.head() 	 #默认前5行数据
df.tail()    #默认后5行数据
"""导出为excel或csv文件"""
#单条件
dataframe_1 = data.loc[data['部门'] == 'A', ['姓名', '工资']]
#单条件
dataframe_2 = data.loc[data['工资'] < 3000, ['姓名', '工资']]
#多条件
dataframe_3 = data.loc[(data['部门'] == 'A')&(data['工资'] < 3000), ['姓名', '工资']]
#导出为excel
dataframe_1.to_excel('dataframe_1.xlsx')
dataframe_2.to_excel('dataframe_2.xlsx')
```



### [内置函数](https://www.runoob.com/python/python-built-in-functions.html)

**map()**

map() 会根据提供的函数对指定序列做映射。

```python
map(function, iterable, ...)	# iterable一个或多个序列
```

```python
>>> def square(x) :            # 计算平方数
...     return x ** 2
...
>>> map(square, [1,2,3,4,5])   # 计算列表各个元素的平方
[1, 4, 9, 16, 25]
>>> map(lambda x: x ** 2, [1, 2, 3, 4, 5])  # 使用 lambda 匿名函数
[1, 4, 9, 16, 25]
# 提供了两个列表，对相同位置的列表数据进行相加
>>> map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
[3, 7, 11, 15, 19]
```

**enumerate()**

enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标。

```python
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
for i, season in enumerate(seasons):
    print("{0}: {1}".format(i, season), end=" ")	# 0: Spring 1: Summer 2: Fall 3: Winter 
b = list(enumerate(seasons, start=1))       # 下标从1开始，指定下标
print(b)		# [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

**zip()**

zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

```python
>>> a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]

>>> a1, a2 = zip(*zip(a,b))          # 与 zip 相反，zip(*) 可理解为解压，返回二维矩阵式
>>> list(a1)
[1, 2, 3]
>>> list(a2)
[4, 5, 6]
```

**round()**

round() 方法返回浮点数x的四舍五入值。

```python
print(round(80.23456, 2))	# 80.23
print(round(100.000056, 3))	# 100.0
```

**sorted()**

sorted() 函数对所有可迭代的对象进行排序操作。

```python
>>>a = [5,7,6,3,4,1,2]
>>> b = sorted(a) 			# [1, 2, 3, 4, 5, 6, 7]

>>> L=[('b',2),('a',1),('c',3),('d',4)]
>>> sorted(L, key=lambda x:x[1])               # 利用key   [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
```

**ord()**

ord()函数是chr()函数（对于8位的ASCII字符串）或unichr() 函数（对于Unicode对象）的配对函数，它以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值，或者Unicode数值

```python
print(ord('a'))		# 97
```

**all()**

all() 函数用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE，如果是返回 True，否则返回 False。元素除了是 0、空、None、False 外都算 True。

```python
>>> all(['a', 'b', 'c', 'd'])  # 列表list，元素都不为空或0
True
>>> all(['a', 'b', '', 'd'])   # 列表list，存在一个为空的元素
False
>>> all([])             # 空列表
True
```

> 注：空元组、空列表返回值为True。

**any()**

any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True。

```python
>>> any(['a', 'b', '', 'd'])   # 列表list，存在一个为空的元素
True
>>> any([0, '', False])        # 列表list,元素全为0,'',false
False
>>> any([])			 # 空列表
False
```

**format()**

```python
print("{} {}".format("hello", "world"))				# 'hello world'
print("{1} {0} {1}".format("hello", "world"))		# 'world hello world'
# 网站名：菜鸟教程, 地址 www.runoob.com
print("网站名：{name}, 地址 {url}".format(name="菜鸟教程", url="www.runoob.com"))
# 通过字典设置参数
site = {"name": "菜鸟教程", "url": "www.runoob.com"}
print("网站名：{name}, 地址 {url}".format(**site))
# 通过列表索引设置参数
my_list = ['菜鸟教程', 'www.runoob.com']
print("网站名：{0[0]}, 地址 {0[1]}".format(my_list))  # "0" 是必须的
```

数字格式化

```python
print("{:.2f}".format(3.1415926))	# 3.14
```

| 数字       | 格式                                                         | 输出                 | 描述                         |
| :--------- | :----------------------------------------------------------- | :------------------- | :--------------------------- |
| 3.1415926  | {:.2f}                                                       | 3.14                 | 保留小数点后两位             |
| 3.1415926  | {:+.2f}                                                      | +3.14                | 带符号保留小数点后两位       |
| -1         | {:-.2f}                                                      | -1.00                | 带符号保留小数点后两位       |
| 2.71828    | {:.0f}                                                       | 3                    | 不带小数                     |
| 5          | {:0>2d}                                                      | 05                   | 数字补零 (填充左边, 宽度为2) |
| 5          | {:x<4d}                                                      | 5xxx                 | 数字补x (填充右边, 宽度为4)  |
| 10         | {:x<4d}                                                      | 10xx                 | 数字补x (填充右边, 宽度为4)  |
| 1000000    | {:,}                                                         | 1,000,000            | 以逗号分隔的数字格式         |
| 0.25       | {:.2%}                                                       | 25.00%               | 百分比格式                   |
| 1000000000 | {:.2e}                                                       | 1.00e+09             | 指数记法                     |
| 13         | {:>10d}                                                      | 13                   | 右对齐 (默认, 宽度为10)      |
| 13         | {:<10d}                                                      | 13                   | 左对齐 (宽度为10)            |
| 13         | {:^10d}                                                      | 13                   | 中间对齐 (宽度为10)          |
| 11         | '{:b}'.format(11) '{:d}'.format(11) '{:o}'.format(11) '{:x}'.format(11) '{:#x}'.format(11) '{:#X}'.format(11) | 1011 11 13 b 0xb 0XB | 进制                         |

**global()**

globals() 函数会以字典类型返回当前位置的全部全局变量。

```python
>>>a='runoob'
>>> print(globals()) # globals 函数返回一个全局变量的字典，包括所有导入的变量。
{'__builtins__': <module '__builtin__' (built-in)>, '__name__': '__main__', '__doc__': None, 'a': 'runoob', '__package__': None}
```

**eval()**

eval() 函数用来执行一个字符串表达式，并返回表达式的值。

```python
>>>x = 7
>>> eval( '3 * x' )
21
```

> **注意：** *eval() 函数执行的代码具有潜在的安全风险。如果使用不受信任的字符串作为表达式，则可能导致代码注入漏洞，因此，应谨慎使用 eval() 函数，并确保仅执行可信任的字符串表达式。*

**hasattr()**

​		函数是Python内置函数之一，用于判断对象是否具有指定的属性或方法。它接受两个参数：对象和属性或方法的名称。函数返回一个布尔值，如果对象具有指定的属性或方法，则返回`True`，否则返回`False`。

```python
class MyClass:
    def __init__(self):
        self.my_attribute = "Hello"

my_object = MyClass()

if hasattr(my_object, "my_attribute"):
    print("my_object具有my_attribute属性")
else:
    print("my_object没有my_attribute属性")
```

**isinstance()**	

来判断一个对象是否是一个已知的类型，还可以判断子类父类

```python
a = 1
isinstance (a,int)	  # True
isinstance (a,(str,int,list)) 	# True
class A:
    pass
class B(A):
    pass 
isinstance(A(), A)    # returns True
type(A()) == A        # returns True
isinstance(B(), A)    # returns True
type(B()) == A        # returns False
```

**vars()**

获取对象的属性和属性值，并以字典的形式返回。该函数在不同的情况下可用于获取对象的属性和属性值，局部变量，全局变量以及模块的全局变量，并将它们以字典的形式返回。

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
person = Person("Alice", 25)
attributes = vars(person)
print(attributes)  # {'name': 'Alice', 'age': 25}

def my_function():
    x = 10
    y = 20
    local_vars = vars()
    print(local_vars)  # {'x': 10, 'y': 20}
my_function()
```

**print()**	

```python
print(b, end=',')		# 将输出结果后加‘，’默认换行符
print(b, a, sep=' ')	# 在多个输出结果中间加字符串，默认空格
```

**exec()**

exec 执行储存在字符串或文件中的Python语句，相比于 eval，exec可以执行更复杂的 Python 代码。

**slice()**

slice() 函数实现切片对象，主要用在切片操作函数里的参数传递。









**super()**

super() 函数是用于调用父类(超类)的一个方法，super()是用来解决多重继承问题的。

```python
#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
class A(object):   # Python2.x 记得继承 object
    def add(self, x):
         y = x+1
         print(y)
class B(A):
    def add(self, x):
        super(B, self).add(x)
b = B()
b.add(2)  # 3
```









### 装饰器

装饰器（decorators）是 Python 中的一种高级功能，它允许你动态地修改函数或类的行为。装饰器是一种函数，它接受一个函数作为参数，并返回一个新的函数或修改原来的函数。

装饰器的语法使用 **@decorator_name** 来应用在函数或方法上。Python 还提供了一些内置的装饰器，比如 **@staticmethod** 和 **@classmethod**，用于定义静态方法和类方法。

应用场景：

> - **日志记录**: 装饰器可用于记录函数的调用信息、参数和返回值。
> - **性能分析**: 可以使用装饰器来测量函数的执行时间。
> - **权限控制**: 装饰器可用于限制对某些函数的访问权限。
> - **缓存**: 装饰器可用于实现函数结果的缓存，以提高性能。

Python 装饰允许在不修改原有函数代码的基础上，动态地增加或修改函数的功能，装饰器本质上是一个接收函数作为输入并返回一个新的包装过后的函数的对象。

```python
def decorator_function(original_function):
    def wrapper(*args, **kwargs):
        before_call_code()	 # 这里是在调用原始函数前添加的新功能
        result = original_function(*args, **kwargs)
        after_call_code()	 # 这里是在调用原始函数后添加的新功能     
        return result
    return wrapper

# 使用装饰器
@decorator_function
def target_function(arg1, arg2):
    pass  # 原始函数的实现
# 等同于target_function = ecorator_function(target_function)
```

decorator_function是装饰器，`wrapper` 是内部函数，它是实际会被调用的新函数，它包裹了原始函数的调用，并在其前后增加了额外的行为。

当我们使用 `@decorator_function` 前缀在 `target_function` 定义前，Python会自动将 `target_function` 作为参数传递给 `decorator_function`，然后将返回的 `wrapper` 函数替换掉原来的 `target_function`。

**类装饰器**

除了函数装饰器，Python 还支持类装饰器。类装饰器是包含 **__call__** 方法的类，它接受一个函数作为参数，并返回一个新的函数。

```python
class DecoratorClass:
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        # 在调用原始函数之前/之后执行的代码
        result = self.func(*args, **kwargs)
        # 在调用原始函数之后执行的代码
        return result

@DecoratorClass
def my_function():
    pass
```

**`@classmethod`**



**`@staticmethod`**



### 生成可执行文件

指令：

```python
pyinstaller --noconsole --onefile main3.py --exclude-module PyQt5 --exclude-module PySide6 --icon="icon.ico"	
--add-data "path;path"  --clean
```

```python
pyinstaller -w --icon="icon.ico" main.py
```

```python
pyinstaller -w main.py --icon="icon.ico" --exclude-module PyQt5 --exclude-module PySide2
```

```python
pyinstaller -w main.py --icon="icon.ico" --exclude-module PyQt6 --exclude-module PySide2 -y 
```

`--noconsole`    `-w`:启动无控制台黑框

--onefile:	  打包成一个文件(只有exe文件)

--exclude：选项来排除模块或文件

--exclude-module： 排除一些指定的包

--excludepath：     选项来排除文件或目录

--add-data：		   选项来添加文件或目录

--icon：		图标

--workpath: 指定了制作过程中临时文件的存放目录

--distpath: 指定了最终的可执行文件目录所在的父目录，无该参数默认dist文件夹

**注意事项：**

虚拟环境中同时存在pyqt5和pyqt6时，需要用exclude排除一个包再进行打包

### 特殊语句

`range`		  1、range(num)——获取一个从0开始num结束的数字序列

​				      2、range(num1,num2)——获取一个从num1开始num2结束的数字序列（不含num2）

​				      3、range(num1,num2,step)——同上step为步长，默认为1

`len()` 	 	确定列表的长度

`pass`		    空语句，占位语句，为了保持程序结构的完整性。

`.strip()`	去除字符串两端的空格

`lower()`	  将字符串转换为小写形式

`.split()`	将字符串根据,分割

```
s = "apple,banana,grape"
result = s.split(',')
print(result)  # 输出：['apple', 'banana', 'grape']
```

`id()`	来查看内存地址变化

```python
a = 10
print(id(a))	# 140728248645632
```

`ord()`		以一个字符作为参数，返回对应的 ASCII 数值

```python
print(ord('a'))			# 97
# 用户按q退出循环
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

`lambda函数`		lambda函数是一种匿名函数，即没有名字的函数，常用于定义简单能够用一行表示的函数。（建议还是老老实实使用def函数）

```python
<函数名> = lambda <参数>: <表达式>
 
f = lambda x, y : x + y
f(10, 15)  # 25
```

**`with`**			用于简化异常处理，并且可以在退出代码块时自动关闭文件、网络连接等资源。它被称为上下文管理器(context manager)。with语句的主要优点是它可以自动管理资源，比如打开和关闭文件，建立和关闭网络连接等。无论 with 中的代码块在执行的过程中发生任何情况，文件最终都会被关闭。

```python
with open("a.txt", "r", encoding="utf-8") as f:
    print(f.read())
```

**`异常处理函数`**  

```python
# finally对应的语句4一定会执行，else对应的语句3在不发生异常时执行
try:
    <语句1>
except:
    <语句2>
else:
    <语句3>
finally:
    <语句4>
```

**`raise`**

```python
raise [exceptionName [(reason)]]
```

其中，用 [] 括起来的为可选参数，其作用是指定抛出的异常名称，以及异常信息的相关描述。如果可选参数全部省略，则 raise 会把当前错误原样抛出；如果仅省略 (reason)，则在抛出异常时，将不附带任何的异常描述信息。

```python
a= 0

if a == 0:
    raise []
    # raise ZeroDivisionError("除数不能为零")
    c = 0
```

`eval`	用于将字符串解析并执行为Python表达式，括号中需要为一个表达式。

```python
x = 1
print(eval('x+1'))  # 输出：2
print(eval('x+y', {'x': 1, 'y': 2}))  # 输出：3
eval('x = 5') 		 # 这会导致语法错误，因为'x = 5'不是一个表达式
```

因为`eval`函数可以解析并执行任何Python表达式，所以如果你在`eval`函数中执行了不可信的或恶意的代码，可能会带来严重的安全问题。

**`@函数修饰符`**		把当前函数当做参数传入到修饰函数里执行，然后再修饰函数里做一些操作

```python
def funcA(A):
    print("function A")
    
@funcA
def funcC():
    print("function C")
    return 2
# 输出：function A
```

```python
def funcA(A):
    print("function A")
    # A()   操作非法
    print(A)

def funcB(B):
    print("function B")
    B()

@funcA
@funcB
def funcC():
    print("function C")
    return 2
# 输出：function B
#function C
#function A
#None
```



`item`	tem()的作用是取出单元素张量的元素值并返回该值，保持该元素类型不变。

```python
x = torch.randn(3, 3)
print(x[1,1])			# tensor(1.4651)
print(x[1,1].item())	# 1.465139389038086
```

item()函数取出的元素值的精度更高，所以在求损失函数等时我们一般用item（）

`items()`	把字典中的每对key和value组成一个元组，并把这些元祖放在列表中返回。

```python
a = {'a': 1, 'b': 2}
print(a.items())		# dict_items([('a', 1), ('b', 2)])
```



### 其他

- 确保所有网络层的通道数都可以被8整除。那么为什么要这样呢？主要还是出于计算机处理单元的架构上考虑：

> 在大多数硬件中，size 可以被 d = 8, 16， ... 整除的矩阵乘法比较快，因为这些 size 符合处理器单元的对齐位宽。 

​		总之就是为了**快。**

字符串的处理操作

```python
a = "aBc DeFg"
print(a.title())	#首字母大写  Abc Defg
print(a.upper())	#所有字母大写ABC DEFG
print(a.lower())	#所有字母小写abc defg
```

### git仓库

Git 是一个开源的分布式版本控制系统。

不同状态的文件在 Git 中处于不同的工作区域。

- **工作区（working**） - 当你 git clone 一个项目到本地，相当于在本地克隆了项目的一个副本。工作区是对项目的某个版本独立提取出来的内容。这些从 Git 仓库的压缩数据库中提取出来的文件，放在磁盘上供你使用或修改。
- **暂存区（staging）**- 暂存区是一个文件，保存了下次将提交的文件列表信息，一般在 Git 仓库目录中。有时候也被称作“索引”，不过一般说法还是叫暂存区。
- **本地仓库（local）** - 提交更新，找到暂存区域的文件，将快照永久性存储到 Git 本地仓库。
- **远程仓库（remote）** - 以上几个工作区都是在本地。为了让别人可以看到你的修改，你需要将你的更新推送到远程仓库。同理，如果你想同步别人的修改，你需要从远程仓库拉取更新。

常用的命令

|               命令名称                |             作用             |
| :-----------------------------------: | :--------------------------: |
| git config --global user.name  用户名 |         设置用户签名         |
| git config --global user.email  邮箱  |         设置用户签名         |
|             **git init**              |         初始化本地库         |
|            **git status**             |        查看本地库状态        |
|          **git add 文件名**           |         添加到暂存区         |
|  **git commit -m "日志信息”文件名**   |         提交到本地库         |
|              git restore              | 放弃更改，还原文件到上一版本 |
|       git reset --hard  版本号        |   版本穿梭，回到想要的版本   |
|                                       |                              |

推送文件至远程github

**克隆github的项目**

```python
git clone git@github.com:用户名/仓库名.git
```

> 注：用户名Dragon1633

与远程建立链接测试

```git
ssh -T git@github.com		# Hi Dragon1633! You've successfully authenticated
```

将文件上传到github指定目录

```python
git init //把这个目录变成Git可以管理的仓库
git add README.md //文件添加到仓库
git add . //不但可以跟单一文件，还可以跟通配符、目录。一个点就把当前目录下所有未追踪的文件全部add了 
git commit -m "first commit" //把文件提交到仓库
git remote add origin git@github.com:Dragon1633/note.git //关联远程仓库
git push -u origin master //把本地库的所有内容推送到远程库上
```

