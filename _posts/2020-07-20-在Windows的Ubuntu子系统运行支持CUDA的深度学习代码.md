---
layout: post
comments: true
title: 在Windows的Ubuntu子系统运行支持CUDA的深度学习代码
published: true
---

2020年6月，微软公布了Windows Subsystem for Linux 2的最新更新，全面支持CUDA和N卡GPU。在Windows上跑Ubuntu子系统并在其中运行GPU加速的深度学习代码成为现实，开发者终于不用特意为了熟悉的Linux环境而在自己的开发机上安装Windows与Ubuntu的双系统（以及Windows10之后繁琐的boot manager调试设置过程），同时又可以让Windows和Ubuntu共享同一个文件系统。

笔者新买了Workstation，在各种尝试安装Windows和Ubuntu双系统还是安装Windows的Ubuntu子系统两个选择中游走踩坑之后，终于成功在Windows 10中安装了最新的WSL2、Ubuntu系统及NVIDIA Driver，成功在Windows的Ubuntu子系统中运行深度学习代码，GPU资源全部跑满！

![](/images/202007/1.png)

# 设置Windows Insider并安装更新

首先要确保电脑的BIOS选项中，Virtualization虚拟化功能是打开的。

BIOS设置好之后，我们需要在Windows中安装微软在2020年6月17日最新开放的Windows Insider Build。我们要首先注册为[Windows Insider](https://insider.windows.com/en-us/)，加入Windows的Dev Channel，然后更新Windows为build 20150或者以上。

![](https://admin.insights.ubuntu.com/wp-content/uploads/1270/image.png)

# 设置Windows Subsystem Linux (WSL) 2

未来微软将WSL 2变为稳定版以后，我们只需要输入下面的命令来设置WSL 2：

```
wsl --install
```

现在WSL2的功能还是测试版，我们需要用管理员权限打开PowerShell。

首先设置WSL 1：

```
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

![](https://admin.insights.ubuntu.com/wp-content/uploads/e722/image.png)

然后设置WSL 2：

```
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

![](https://admin.insights.ubuntu.com/wp-content/uploads/e930/image.png)

重启Windows 10:

```
Restart-Computer
```

之后WSL 2成为默认之后下面的步骤就可以省略，不过现在我们还需要打开PowerShell将WSL 2设置为默认选项：

```
wsl.exe --set-default-version 2
```

![](https://admin.insights.ubuntu.com/wp-content/uploads/a203/Annotation-2020-05-27-173213.png)

# 在WSL上安装Ubuntu

在微软商店中安装[Ubuntu](https://www.microsoft.com/store/productId/9NBLGGH4MSV6):

![](https://admin.insights.ubuntu.com/wp-content/uploads/cae0/image.png)

# 安装Windows Terminal

在微软商店中安装[Windows Terminal](https://www.microsoft.com/store/productId/9N0DX20HK701)。Windows Terminal的主要好处是未来可以在同一个窗口中一键打开多个PowerShell和Ubuntu Terminal的tab，非常方便。

![](https://admin.insights.ubuntu.com/wp-content/uploads/e1d4/image.png)

# 设置WSL上面的Ubuntu

在Windows开始菜单中打开Ubuntu，第一次打开需要设置Ubuntu系统的用户名和密码，这个账户是和Windows账户分开的。

![](https://admin.insights.ubuntu.com/wp-content/uploads/b549/image.png)

设置完毕后关掉原先的窗口，然后打开Windows Terminal，在下拉菜单中选择Ubuntu开启一个新的Ubuntu Terminal。

![](https://admin.insights.ubuntu.com/wp-content/uploads/f1cc/image.png)

下面这一步很关键，我们要检查确认我们运行的是正确的WSL 2 Linux内核。在Ubuntu中输入：

```
uname -r
```

![](https://admin.insights.ubuntu.com/wp-content/uploads/1f26/image.png)

内核版本一定是 **4.19.121** 或更高。如果不是的话先在Windows的PowerShell中试下：

```
wsl.exe --update
```

如果还是不行，检查一下Windows升级设定中"Receive updates for other Microsoft products when you update Windows"这个选项是打开的：

![](https://admin.insights.ubuntu.com/wp-content/uploads/92f6/image.png)

然后再检查一下Windows更新，看看有没有最新的Windows Subsystem for Linux Update。

![](https://admin.insights.ubuntu.com/wp-content/uploads/26b3/image.png)

# 在Windows 10上面安装Nvidia的WSL2驱动

针对不同显卡安装相应的[驱动](https://devblogs.nvidia.com/announcing-cuda-on-windows-subsystem-for-linux-2/)。

未来Nvidia的驱动会自动集成在Windows Update中，但现在支持WSL2的Nvidia驱动还是开发者测试版，用户需要加入[Nvidia Developer Program](https://developer.nvidia.com/developer-program)来获取最新驱动的下载权限。

![](https://admin.insights.ubuntu.com/wp-content/uploads/7e60/image.png)

# 在WSL中安装Docker

在Ubuntu Terminal中：

```
sudo apt -y install docker.io
```

![](https://admin.insights.ubuntu.com/wp-content/uploads/909f/image.png)

## 安装Nvidia Container Toolkit

设置版本的变量，导入Nvidia库的GPG Key，把Nvidia的repo加入到Ubuntu的apt安装源中。在Ubuntu Terminal中：

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container-experimental.list | sudo tee /etc/apt/sources.list.d/libnvidia-container-experimental.list
```

![](https://admin.insights.ubuntu.com/wp-content/uploads/0bb3/image.png)

更新Ubuntu的apt安装源然后安装Nvidia运行环境：

```
sudo apt update && sudo apt install -y nvidia-docker2
```

![](https://admin.insights.ubuntu.com/wp-content/uploads/9d7e/image.png)

关闭所有的Ubuntu terminal，打开PowerShell terminal，手动关闭掉Ubuntu内核：

```
wsl.exe --shutdown Ubuntu
```

![](https://admin.insights.ubuntu.com/wp-content/uploads/b079/image.png)

# 测试GPU计算环境

打开一个新的Ubuntu terminal并开启Docker:

```
sudo dockerd
```

在另一个新的Ubuntu terminal中运行：

```
sudo docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

如果一切设置没问题的话，输出应该和下面的类似：

![](https://admin.insights.ubuntu.com/wp-content/uploads/7875/image.png)

## 测试Tensorflow-GPU容器

在另一个新的Ubuntu terminal中运行：

```
docker run -u $(id -u):$(id -g) -it --gpus all -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3-jupyter
```

![](https://admin.insights.ubuntu.com/wp-content/uploads/81bb/image.png)

一切正常的话，Terminal最后会给出一个带token的jupter notebook地址。将其复制并在浏览器中打开，我们就成功开启了一个GPU加速的运行Tensorflow的Jupyter notebook:

![](https://admin.insights.ubuntu.com/wp-content/uploads/0cb0/image.png)

现在我们就可以在这个Windows的Ubuntu子系统环境中编写、测试和运行支持CUDA的Tensorflow了！



Reference: https://ubuntu.com/blog/getting-started-with-cuda-on-ubuntu-on-wsl-2
