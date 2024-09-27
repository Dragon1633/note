**操作**

> 打开编辑界面输入**i**键保存
>
> INSERT状态下按**Esc**是退出编辑界面
>
> 先按 **:** 再输入**wqa**按回车为保存
>
> 先按 **:** 再输入**q**按回车为取消此次提交

# 配置

## 配置并查看初始信息

1. 打开Git Bash
2. 设置用户信息

```cmd
git config --global user.name "kurisu"
git config --global user.email "lzyoung123@163.com"
```

查看配置信息

```cmd
git config --global user.name
git config --global user.email
```

## 设置指令别名

1. 在用户目录下创建 `.bashrc` 配置文件

   ```cmd
   touch .bashrc
   ```

2. 在`bashrc`文件中输入：

   ```cmd
   # 单行输出日志并且只取前几位数
   alias git-log='git log --pretty=oneline --all --graph --abbrev-commit'
   
   alias ll='ls -al'
   ```

# 常用命令

## 获取本地仓库

在想要创建本地仓库的目录下执行命令`git init`

## 基础操作指令

1. `git add`（工作区文件移动至暂存区）
2. `git commit`（将工作区或暂存区文件提交到版本库）
3. `git status`（查看状态）
4. `git log [option]`（查看提交记录）
5. `git reset --hard commitID`（版本切换）
6. `git reflog`（查看已经删除的提交记录）

## 添加文件至忽略列表

在仓库中创建名为`.gitignore`文件

```cmd
touch .gitignore
```

在`.gitignore`中输入想要忽略的文件

```cmd
# 忽略所有后缀为.txt的文件
*.txt
# 忽略build文件夹下的所有文件
build/
# 忽略build文件夹下的后缀为.txt的文件
build/*.txt
```

## 分支指令

1. `git branch`（查看本地分支）
2. `git branch 分支名`（创建本地分支）
3. `git checkout 分支名`（切换分支）
4. `git checkout -b 分支名`（创建并切换分支）
5. `git branch -d 分支名`（删除分支，更改为-D可以强制删除）
6. `git merge 分支名`（合并分支）

当合并的两个分支同时修改了一个分支，merge时会发生冲突，直接处理文件中冲突的地方，再将修改后的结果暂存并提交即可。

# Git远程仓库

## 创建远程仓库

1. 注册GitHub账号
2. 在GitHub中创建仓库

## 配置SSH公钥（命令行方式连接GitHub账号）

1. `rsa`方法生成SSH公钥

   ```cmd
   ssh-keygen -t rsa
   ```

2. 获取公钥内容

   ```cmd
   cat ~/.ssh/id_rsa.pub
   ```

3. 验证是否配置成功

   ```cmd
   ssh -T git@github.com
   ```

## 操作远程仓库

1. 添加远程仓库

   ```cmd
   git remote add <远端名称> <仓库路径>
   # example（仓库路径从Github仓库的SSH地址获得）
   git remote add origin git@github.com:IHateStr/bachelor_graduation_design.git
   ```

2. 查看远程仓库

   ```cmd
   git remote
   ```

3. 推送到远程仓库

   ```cmd
   # --set-upstream 推送到远端的同时建立起和远端分支的关联关系
   git push [-f] [--set-upstream] [远端名称] [本地分支名][:远端分支名]
   # 若远程分支名和本地分支名名称相同，则可以只写本地分支
   git push origin master
   # 如果当前分支已经和远端分支关联，则可以省略分支名和远端名
   git push
   ```

4. 查询本地分支和远程分支的关联关系

   ```cmd
   git branch -vv
   ```

5. 从远程仓库克隆（如果已经有一个远端仓库，我们可以直接clone到本地）

   ```cmd
   git clone <仓库路径> [本地目录]
   # example
   git clone https://github.com/IHateStr/bachelor_graduation_design.git ~/demo
   ```

6. 从远程仓库中抓取和拉取

   ```cmd
   # 抓取
   git fetch [remote name] [branch name]
   
   #拉取，拉取指令会执行抓取和合并两步，同样可能发生合并冲突
   git pull [remote name] [branch name]
   ```



**git commit**

```python
git commit –a –m "commit messeages"
```

- `-a`参数会把当前暂存区里所有的修改（包括删除操作）都提交，`-m`参数不需要进入编辑界面即可备注此次更改的信息。
- 未输入-m会打开了一个`vim`编辑界面，敲入`i`键后保存，输入要添加的`message`后，输入“`ESC`”按键退出编辑界面，然后再敲入“`:wqa`”后会保存`message`内容，并且提交此次修改，如果敲入“`:q`”会取消这次提交。

```python
git commit --amend
```

- 此操作会把此次提交追加到上一次的`commit`内容里

**合并**：合并多个`commit`备注信息为一个。

使用`git rebase –i 4cbeb4248f7`，`-i`后面的参数表示不要合并的`commit`的`hash`值。

<img src="D:/tool/typora/image/image-20240926211910517.png" alt="image-20240926211910517" style="zoom:50%;" />

`pick` 和 `squash`的意思如下：

- `pick` 的意思是要会执行这个 `commit`。
- `squash` 的意思是这个 `commit` 会被合并到前一个`commit`。

将 `ad777ea`和`a271901`这两个`commit`前方的命令改成 `squash` 或 `s`，然后输入`:wq`以保存并退出。

退出后会弹出界面，即需要重新编辑合并后的`commit`信息，未注释掉的是会包含在`commit message`里的，按”`wqa`”保存后即完成了此次`commit`的合并。



**删除某个文件**

```python
git rm file.txt
# git将自动提示：rm "file.txt"
```

提交更改。使用以下命令提交文件的删除操作

```python
git commit -m "Remove file.txt"
```

推送更改到GitHub仓库。

```python
git push origin master
```

