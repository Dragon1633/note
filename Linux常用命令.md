# Linux常用命令

## 常用命令汇总

**touch：创建空文件**

**cat：显示文件内容**

**more：逐页显示文本文件内容**

**head：显示文件前几行。tail：显示文件后几行**

**grep：文件中搜索指定文本**

**ps：显示当前运行的进程**

**kill：终止进程**

**ifconfig：查看网络接口信息**

**wget/curl：从网络下载文件**

**chmod：修改文件或目录权限**

**tar：用于压缩和解压缩**（打包：tar -cvf 解压：tar - xvf。打包并压缩：tar -czvf，解压并提取：tar -xzvf。打包生成.tar格式，压缩生成.gz格式）

**find：在文件系统中查找文件和目录**

**echo：在终端输出指定的文本字符串**

**free：显示系统内存使用情况**





## 文件与目录

### 1、文件操作

```python
# 1、创建文件
touch [文件名] 	# 创建单个文件
touch [文件名1] [文件名2] [文件名3] 			# 创建多个文件

# 2、编辑文件
vim [文件名] 	# 进入文件之后，按i进入编辑模式，编辑完成，按ESC键退出编辑模式，输入:wq回车保存退出！

# 3、查看文件
cat [文件名]	 # 查看文件
cat -n [文件名] # 查看文件，显示行号

# 4、查看文件头几行内容
head -n [行数] [文件名]

# 5、删除文件
rm [文件名] # 删除文件，会询问
rm -f [文件名] # 强制删除文件，不会询问

# 6、复制文件1到文件2位置
cp [文件1] [文件2]

# 7、移动文件
mv [文件1] [文件2] # 移动文件，可实现重命名
```

### 2、目录操作

```python
# 1、显示当前工作目录
pwd

# 2、显示当前目录下的内容
ls
            # -a：显示当前目录所有的文件和目录，包括隐藏的；
            # -l：以列表的方式显示信息；
        
# 3、进入目录
cd [目录地址]  # 进入目录
cd ..		 # 返回上级目录
cd 			 # 回到root目录
cd /		 # 返回根目录
cd - 		 # 返回上一次使用的目录

# 4、创建目录
mkdir [目录地址] 	# 创建单级目录
mkdir -p [目录地址] # 创建多级目录，不存在的子目录也一起创建

# 5、删除目录
rmdir [目录地址] 	# 删除空目录，不可删除非空目录
rm -rf [目录地址] 	# 强制删除目录，即使非空

# 6、清空目录（删除当前目录下所有内容）
rm -f *			   # 清空当前目录下所有文件（不包括目录）
rm -rf *		   # 清空当前目录下所有文件（包括目录）

# 7、复制目录
cp -r [目录1] [目录2] # 复制目录1到目录2，目录2不存在
cp -r [目录1]/. [目录2] # 复制目录1到目录2，目录2已存在（不再演示）
			# 如果目录2存在，且里面有内容，需要先清空！命令：rm -rf /home/www/statics/*
    
# 8、移动目录
mv [目录1] [目录2] # 移动目录1到目录2下，可实现目录的重命名
```

## 压缩与打包

### 1、zip和unzip命令

```python
# 1、压缩
zip -r 压缩包名 源文件/目录 # 压缩指定目录下的所有目录和文件（演示）
zip -r 压缩包名 源文件/目录1 源文件/目录2 源文件/目录3 # 压缩多个文件或目录
zip -rm 压缩包名 源文件/目录 # 压缩指定目录下的所有目录和文件，然后删除原来的文件和目录
# 选项
-r 递归压缩目录，及将制定目录下的所有文件以及子目录全部压缩（演示）
-m 将文件压缩之后，删除原始文件，相当于把文件移到压缩文件中
-v 显示详细的压缩过程信息
-q 在压缩的时候不显示命令的执行过程
-压缩级别 压缩级别是从 1~9 的数字，-1 代表压缩速度更快，-9 代表压缩效果更好
-u 更新压缩文件，即往压缩文件中添加新文件

# 2、解压缩
unzip [选项] 压缩包名
# 选项
-d 目录名 将压缩文件解压到指定目录下
-n 解压时并不覆盖已经存在的文件
-o 解压时覆盖已经存在的文件，并且无需用户确认
-v 查看压缩文件的详细信息，包括压缩文件中包含的文件大小、文件名以及压缩比等，但并不做解压操作
-t 测试压缩文件有无损坏，但并不解压
-x 文件列表	解压文件，但不包含文件列表中指定的文件
# 常用
unzip 压缩包名 # 解压到当前目录
unzip -d 目录 压缩包名 # 解压到指定目录
```

### 2、gzip和gunzip命令

```python
# 1、压缩
gzip [选项] 源文件/目录
# 选项
-c	将压缩数据输出到标准输出中，并保留源文件。
-d	对压缩文件进行解压缩。
-r	递归压缩指定目录下以及子目录下的所有文件。
-v	对于每个压缩和解压缩的文件，显示相应的文件名和压缩比。
-l	对每一个压缩文件，显示以下字段：
	- 压缩文件的大小；
	- 未压缩文件的大小；
	- 压缩比；
	- 未压缩文件的名称。
-数字	用于指定压缩等级，-1 压缩等级最低，压缩比最差；-9 压缩比最高。默认压缩比是 -6
# 常用
gzip 文件名 # 压缩文件，文件会被删除
gzip -c 文件名 > 文件名.gz # 压缩文件，文件不会被删除
gzip -r 目录 # 压缩目录下每一个文件（gzip不能压缩目录）

# 2、解压
gunzip [选项] 文件
# 选项
-r	递归处理，解压缩指定目录下以及子目录下的所有文件
-c	把解压缩后的文件输出到标准输出设备
-f	强制解压缩文件，不理会文件是否已存在等情况
-l	列出压缩文件内容
-v	显示命令执行过程
-t	测试压缩文件是否正常，但不对其做解压缩操作
# 常用
gunzip 压缩包名 # 解压文件
gunzip -r 目录 # 解压目录下所有压缩文件
```

### tar打包和解打包

```python
# 1、打包文件或目录
tar [选项] 源文件或目录
# 选项
-c	将多个文件或目录进行打包
-A	追加 tar 文件到归档文件
-f 包名  指定包的文件名。包的扩展名是用来给管理员识别格式的，所以一定要正确指定扩展名
-v	显示打包文件过程
# 常用
tar -cvf 包名（.tar） 文件/目录 # 常用打包文件或目录
# 选项 "-cvf" 一般是习惯用法，记住打包时需要指定打包之后的文件名，而且要用 ".tar" 作为扩展名

# 2、打包并压缩目录（分开）
# gzip和bzip2不能直接压缩目录，要先使用tar打包，然会再对tar包继续压缩

# 3、解打包
# 格式
tar [选项] tar包
# 选项
-x	对 tar 包做解打包操作
-f	指定要解压的 tar 包的包名
-t	只查看 tar 包中有哪些文件或目录，不对 tar 包做解打包操作
-C 目录  指定解打包位置
-v	显示解打包的具体过程
# 常用
tar -xvf tar包 # 解打包到当前目录下
tar -tvf tar包 # 不解打包，只是看包内的文件

# 4、压缩与打包合并操作
tar [选项] 压缩包 源文件或目录
# 选项
-z：压缩和解压缩 ".tar.gz" 格式；
-j：压缩和解压缩 ".tar.bz2"格式；
# 常用
tar -zcvf 包名.tar.gz [目录] # 压缩并打包
tar -zxvf 包名.tar.gz # 解压缩并解打包
```

## 文本编辑

> 一般情况下，执行**vim**文件名，按i键进行编辑，编辑完成按**$ESC$**键退出输入模式，输入**$:wq$**，按回车键保存退出就行！

### 1、vim三种工作模式

**命令模式：**默认，此模式下，可使用方向键（上、下、左、右键）或 k、j、h、i 移动光标的位置，还可以对文件内容进行复制、粘贴、替换、删除等操作；

**输入模式：**使 Vim 进行输入模式的方式是在命令模式状态下输入 i、I、a、A、o、O 等插入命令，当编辑文件完成后按 Esc 键即可返回命令模式；

![image-20241029212933930](image\image-20241029212933930.png)
**编辑模式：**编辑模式用于对文件中的指定内容执行保存、查找或替换等操作。在命令模式状态下按“：”键，此时 Vim 窗口的左下方出现一个“：”符号，这是就可以输入相关指令进行操作了；

### 2、常用命令

```python
# 1、查看文件(非vim)
cat 文件名
 
# 2、打开文件（vim）
vim 文件名 # 如果没有此文件，则创建该文件，并打开
 
# 3、编辑文本[见“插入文本快捷键”]
# 从命令模式进入输入模式进行编辑，可以按下 I、i、O、o、A、a 等键来完成，使用不同的键，光标所处的位置不同
# 按i进入输入模式！
# 随意输入文本！
# 输入完之后，按ESC退出输入模式，回到命令模式，然后输入:wq，最后按回车键，保存退出！输入:qa!，不保存退出

```

### 3、插入文本快捷键

![image-20241029213602894](image\image-20241029213602894.png)

### 4、查找文本快捷键

### ![image-20241029213623828](image\image-20241029213623828.png)5、删除文本快捷键

> 注意，被删除的内容并没有真正删除，都放在了剪贴板中。将光标移动到指定位置处，按下 "p" 键，就可以将刚才删除的内容又粘贴到此处。

![image-20241029214259896](image\image-20241029214259896.png)

### 6、复制和粘贴文本快捷键

![image-20241029214331976](image\image-20241029214331976.png)

### 7、保存退出文本命令

![image-20241029214352657](image\image-20241029214352657.png)

Ubuntu启用



## make/makefile/cmake/nmake区别

![image-20250222164307490](D:/tool/typora/image/image-20250222164307490.png)

### gcc

它是[GNU Compiler Collection](https://zhida.zhihu.com/search?content_id=112821805&content_type=Article&match_order=1&q=GNU+Compiler+Collection&zhida_source=entity)（就是GNU编译器套件），也可以简单认为是**编译器**，它可以编译很多种编程语言（括C、C++、Objective-C、Fortran、Java等等）。

我们的程序**只有一个**源文件时，直接就可以用gcc命令编译它。

### make

make工具可以看成是一个智能的**批处理**工具，它本身并没有编译和链接的功能，而是用类似于批处理的方式—通过调用**makefile文件**中用户指定的命令来进行编译和链接的。

### makefile

make工具就根据makefile中的命令进行编译和链接的。makefile命令中就包含了调用gcc（也可以是别的编译器）去编译某个源文件的命令。

makefile在一些简单的工程完全可以人工拿下，但是当工程非常大的时候，手写makefile也是非常麻烦的，如果换了个平台makefile又要重新修改，这时候就出现了下面的Cmake这个工具。

### cmake

cmake就可以更加简单的生成makefile文件给上面那个make用。当然cmake还有其他更牛X功能，就是可以**跨平台**生成对应平台能用的makefile

cmake根据一个叫CMakeLists.txt文件（学名：组态档）去生成makefile。

### CMakeList.txt

自己手写

### nmake

nmake是[Microsoft Visual Studio](https://zhida.zhihu.com/search?content_id=112821805&content_type=Article&match_order=1&q=Microsoft+Visual+Studio&zhida_source=entity)中的附带命令，需要安装VS，实际上可以说相当于linux的make，





## 其他

**ctrl+c**			停止下载

**ctrl+alt+t**	 打开终端

VScode远控时：   **ctrl+j**			打开终端

**常用的安装软件**

```python
sudo apt-get install vim
sudo apt install git

sudo apt-get update
sudo apt-get install python3.6
sudo apt-get install python3-pip
sudo update-alternatives --install /usr/bin/python python /usr/bin/python2 100
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 150
# 可用以下命令切换python版本
sudo update-alternatives --config python
# 开始安装一堆子依赖
sudo apt-get install git wget flex bison gperf python3 python3-pip python3-setuptools cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
```

### 更改默认源

```c
sudo cp /etc/apt/sources.list /etc/apt/sources.list.old //先把源文件复制到sources.list.old
sudo gedit /etc/apt/sources.list //修改sources文件，文件内容删掉，替换成清华源
sudo apt-get update
```

### 查看磁盘空间

`df`以磁盘分区为单位查看文件系统，可以获取硬盘被占用了多少空间，目前还剩下多少空间等信息。 -h 选项为根据大小适当显示。

```c
df -h
```

`du` (disk usage)，含义为显示磁盘空间的使用情况，用于查看当前目录的大小

```shell
#查看当前目录大小
du -sh
#返回该目录/文件的大小
du -sh [目录/文件]
```

### SSH远程

SSH一般采用22端口

查看ip信息

```c++
ifconfig
ip addr show | grep inet
```

查看SSH服务是否开启

```bash
systemctl status sshd
```

启动SSH服务

```bash
sudo systemctl start sshd
sudo service ssh restart		#重启
```





## 常用包

#### wget

`wget` 是 Linux 系统中一个非常常用的**命令行下载工具** ，它可以用来从网络上（通常是 HTTP、HTTPS 或 FTP 协议）下载文件。

```c++
sudo apt update
sudo apt install wget
wget https://example.com/file.zip 
```

