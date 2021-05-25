# Report For MSRA-USTC Innovation Project 2021: AI-System
## Environment 
### Hardware
* Intel Core i7-9750H CPU with 6 Core, 12 Threads
* NVIDIA GeForce GTX1650 with 4GB of GDDR5
### Software
* Operating System: Ubuntu 18.04
* DL Arch: Pytorch
* Python version: 3.8
* CUDA version: 11.0
## Step 1. Install Docker
run following command:
```bash
 sudo apt-get update
 sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```
Add Docker’s official GPG key:
` curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg`

Set up the stable repository:
` echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null`

  Update the apt package index, and install the latest version of Docker Engine and containerd:
  ```shell 
  sudo apt-get update
 sudo apt-get install docker-ce docker-ce-cli containerd.io
  ```


 Verify that Docker Engine is installed correctly by running the hello-world image:

 ` sudo docker run hello-world`

 we can get the following result:
 ![hello_docker](/Lab5/images/hello_docker.png)

This means the docker has been installed successfully.

## Step 2 Run your first container

run `docker pull alpine` to pull an alpine Linux image into your computer.

after you successfully pulled it, you can list all your docker images using `docker images`.

![image-20210513192219661](/Lab5/images/alpine.png)

you can run alpine image using command `docker run alpine pwd`: start the alpine container and  execute command `pwd` in that container.

Add argument `-it` to run docker container interactively.

After exiting the container you can list all your execution history using command `docker ps -a`.

The output of the commands above is something like this:

![image-20210513193740683](/Lab5/images/alpineexecution.png)

if you run `top` in the container, the output will be something like:

![image-20210513194106664](/Lab5/images/alpinetop.png)

## Step 3 Deploy PyTorch Training Program on Docker

First build your Dockerfile.

main.py is originated from [PyTorch Official example](https://github.com/pytorch/examples/blob/master/mnist/main.py).

```dockerfile
# What image do you want to start building on?
FROM nvidia/cuda:10.1-cudnn7-devel

# Make a folder in your image where your app's source code can live
RUN mkdir -p /src/app

# Tell your container where your app's source code will live
WORKDIR /src/app

# What source code do you what to copy, and where to put it?
COPY main.py /src/app

# Does your app have any dependencies that should be installed?
# Changing Source to TUNA and Aliyun to fix invalid: BADSIG F60F4B3D7FA2AF80 cudatools <cudatools@nvidia.com>
RUN echo "deb https://mirrors.aliyun.com/nvidia-cuda/ubuntu1804/x86_64/ ./" > /etc/apt/sources.list.d/cuda.list
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-security main restricted universe multiverse" >> /etc/apt/sources.list&& \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty main restricted universe multiverse" >> /etc/apt/sources.list&& \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list 
RUN apt-get update && apt-get install wget -y
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda
ENV PATH /opt/conda/bin:$PATH



RUN conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
RUN conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
RUN conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
RUN conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/bioconda
RUN conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/msys2
RUN conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge
RUN conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch
RUN conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/menpo
RUN conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/simpleitk
RUN conda clean -i
RUN conda config --set show_channel_urls yes
RUN conda install pytorch torchvision cudatoolkit=10.1 pytorch
# How do you start your app?
CMD [ "python", "main.py" ]# What image do you want to start building on?
FROM nvidia/cuda:10.1-cudnn7-devel

# Make a folder in your image where your app's source code can live
RUN mkdir -p /src/app

# Tell your container where your app's source code will live
WORKDIR /src/app

# What source code do you what to copy, and where to put it?
COPY main.py /src/app

# Does your app have any dependencies that should be installed?
# Changing Source to TUNA and Aliyun to fix invalid: BADSIG F60F4B3D7FA2AF80 cudatools <cudatools@nvidia.com>.
RUN echo "deb https://mirrors.aliyun.com/nvidia-cuda/ubuntu1804/x86_64/ ./" > /etc/apt/sources.list.d/cuda.list
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-security main restricted universe multiverse" >> /etc/apt/sources.list&& \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty main restricted universe multiverse" >> /etc/apt/sources.list&& \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list 
RUN apt-get update && apt-get install wget -y
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda
ENV PATH /opt/conda/bin:$PATH

RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# How do you start your app?
CMD [ "python", "main.py" ]
```

run command `docker build -t msra_bcli_train .` to build docker image.

The docker image is built successfully and the output is shown below:

Add argument to build tagged docker image: `docker build -f Dockerfile.gpu -t msra_bcli_train_gpu .`

The docker image is built successfully and the output is shown below:

![image-20210513221459126](/Lab5/images/trainbuild.png)

use `docker images` to list all the local docker images:

![image-20210513222023298](/Lab5/images/trainbuild_docker_images.png)

 Use command `docker run --name training msra_bcli_train_gpu ` to start the image we recently built.

the output will be something like this:

![image-20210513230936851](/Lab5/images/trainresult.png)

## Step 4 Deploy PyTorch Inference Program on Docker

Prepare source code of TorchServe:

```bash
$ BRANCH_NAME=v0.1.0
$ rm -rf serve
$ git clone https://github.com/pytorch/serve.git
$ cd serve
$ git checkout -b $BRANCH_NAME
$ cd ..
```

build GPU images for docker:

```bash
$ docker build --file Dockerfile.infer.gpu -t msra-bcli-torchserve:0.1-gpu .
```

Dockerfile is shown below:

```dockerfile
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV PYTHONUNBUFFERED TRUE
RUN echo "deb https://mirrors.aliyun.com/nvidia-cuda/ubuntu1804/x86_64/ ./" > /etc/apt/sources.list.d/cuda.list

RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-security main restricted universe multiverse" >> /etc/apt/sources.list&& \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty main restricted universe multiverse" >> /etc/apt/sources.list&& \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list 
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    ca-certificates \
    dpkg-dev \
    g++ \
    python3-dev \
    openjdk-8-jdk-headless \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp
RUN curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
RUN export USE_CUDA=1
RUN pip install psutil future torch torchvision torchtext
RUN pip install --no-cache-dir psutil
RUN pip install --no-cache-dir captum


# RUN pip --no-cache-dir install psutil future torch torchvision torchtext
# RUN pip install --no-cache-dir psutil
# RUN pip install --no-cache-dir captum

ADD serve serve
RUN pip install ../serve/

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server

COPY config.properties /home/model-server/config.properties
RUN mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store

EXPOSE 8080 8081

WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]
```



the result is shown below:

![image-20210517183130196](/Lab5/images/tsbuildgpu.png)

Start the container using TorchServe Image.

```bash
$ docker run --rm -it -p 8080:8080 -p 8081:8081 msra-bcli-torchserve:0.1-gpu
```

the output is shown below:

![image-20210513211746450](/Lab5/images/torchservestart.png)

In the same host run command

```bash
$ curl http://localhost:8080/ping
```

The output is shown below:

![image-20210513211908285](/Lab5/images/checktorchservestatus.png)

If we check the container running we can get the following result:

![image-20210517183525327](/Lab5/images/dockerpsgpu.png)

now we go into the container:

```ba
$ docker exec -it 99c28e8dc03b  /bin/bash
```

run `docker port containerid`

get the following output:

![image-20210517183716872](/Lab5/images/dockerportgpu.png)

go to the model folder

```ba
$ cd /home/model-server/model-store
```

get a model:

```bash
$ apt-get update  
$ apt-get install wget
$ wget https://download.pytorch.org/models/densenet161-8d451a50.pth
```

then we get the following result:

![image-20210517183912534](/Lab5/images/dockerdownloadmodel.png)

in the container we install model archiver:

```bash
$ cd /serve/model-archiver
$ pip install .
```

run 

```ba
$ torch-model-archiver --model-name densenet161 --version 1.0 --model-file /serve/examples/image_classifier/densenet_161/model.py --serialized-file /home/model-server/model-store/densenet161-8d451a50.pth --export-path /home/model-server/model-store --extra-files /serve/examples/image_classifier/index_to_name.json --handler image_classifier
```

we get the .mar file.

run 

```shell
$ ls  /home/model-server/model-store/
```

To  get the files in the directory:

![image-20210513233137638](/Lab5/images/tsls.png)

Stop the TorchServe if started:

```shell
$ torchserve --stop
```

if succeed then get the following output:

```bash
TorchServe has stopped.
```



run

``` shell
$ cd /home/model-server/
$ torchserve --start --ncs --model-store model-store --models densenet161.mar
```

to start the service.

Output will be like this:

![image-20210517184154542](/Lab5/images/workermodelgpu.png)

Open new terminal in local host and execute the following command:

```shell
$ curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
$ curl -X POST http://127.0.0.1:8080/predictions/densenet161 -T kitten.jpg
```

the result will be like this:

![image-20210513233505402](/Lab5/images/success.png)

# Pull Requests
Below are my fixes to some errors encountered when doing the lab 5.

I pulled them to the original repository after I finished my report.

PRs can be seen at https://github.com/microsoft/AI-System/pull/27 and https://github.com/microsoft/AI-System/pull/28.

## 1. Fix invalid: BADSIG F60F4B3D7FA2AF80 error in basic lab 5 #28

## Environment
* OS: Ubuntu 18.04
* Python 3.6
## Command
`docker build -f Dockerfile.gpu -t train_dl .`
## Error
When the dockerfile.gpu runs the command `RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`to install `cudatookit`, because the CDN cached a newer signature Release.gpg but did NOT refresh Release's cache at the same time(see https://github.com/NVIDIA/nvidia-docker/issues/613), the installation will return an error like
```bash
W: GPG error: https://developer.download.nvidia.cn/compute/machine-learning/repos/ubuntu1804/x86_64  Release: The following signatures were invalid: BADSIG F60F4B3D7FA2AF80 cudatools <cudatools@nvidia.com>
E: The repository 'https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release' is not signed.
```
 (see also https://github.com/NVIDIA/nvidia-docker/issues/969)
The installation may fail 1000 times unless adding a proxy. (but this error may happen somewhere else according to the comments)
## Solution
I tried to change the downloading source from Nvidia to Aliyun and TUNA, (like https://github.com/NVIDIA/nvidia-docker/issues/969#issuecomment-703186192) and removed the **adding pubkey** step on my second try(to reproduce) to build the image and it worked for me.
```dockerfile
RUN curl -fsSL https://mirrors.aliyun.com/nvidia-cuda/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - // the command I removed
```
>  At the first time I added all the commands in https://github.com/NVIDIA/nvidia-docker/issues/969#issuecomment-703186192 and resolved the problem, but at the second time I removed the command **above** and it still works. However I didn't know whether it is because I already added before. So if the "removed" version cause some problems, please consider readding the command above.
## 2. Fix subprocess.CalledProcessError when building TorchServe GPU image in basic lab 5 #27
## Environment
* OS: Ubuntu 18.04
* CUDA 11.0
* Python 3.6
## Command
`docker build --file Dockerfile.infer.gpu -t torchserve:0.1-gpu .`
## Error log
```bash
  Building wheel for torchserve (setup.py): started
  Building wheel for torchserve (setup.py): still running...
  Building wheel for torchserve (setup.py): finished with status 'error'
  ERROR: Command errored out with exit status 1:
   command: /usr/bin/python3 -u -c 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-req-build-8zsfe839/setup.py'"'"'; __file__='"'"'/tmp/pip-req-build-8zsfe839/setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' bdist_wheel -d /tmp/pip-wheel-gr9trt96

```
## Gradle output
```bash
FAILURE: Build failed with an exception.
  
  * What went wrong:
  Execution failed for task ':server:compileJava'.
  > Compilation failed; see the compiler error output for details.
  
  * Try:
  Run with --stacktrace option to get the stack trace. Run with --info or --debug option to get more log output. Run with --scan to get full insights.
  
  * Get more help at https://help.gradle.org

 BUILD FAILED in 1s
    9 actionable tasks: 8 executed, 1 up-to-date
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/tmp/pip-req-build-8zsfe839/setup.py", line 164, in <module>
        license='Apache License Version 2.0'
      File "/usr/local/lib/python3.6/dist-packages/setuptools/__init__.py", line 153, in setup
        return distutils.core.setup(**attrs)
      File "/usr/lib/python3.6/distutils/core.py", line 148, in setup
        dist.run_commands()
      File "/usr/lib/python3.6/distutils/dist.py", line 955, in run_commands
        self.run_command(cmd)
      File "/usr/lib/python3.6/distutils/dist.py", line 974, in run_command
        cmd_obj.run()
      File "/usr/local/lib/python3.6/dist-packages/setuptools/command/install.py", line 61, in run
        return orig.install.run(self)
      File "/usr/lib/python3.6/distutils/command/install.py", line 589, in run
        self.run_command('build')
      File "/usr/lib/python3.6/distutils/cmd.py", line 313, in run_command
        self.distribution.run_command(command)
      File "/usr/lib/python3.6/distutils/dist.py", line 974, in run_command
        cmd_obj.run()
      File "/usr/lib/python3.6/distutils/command/build.py", line 135, in run
        self.run_command(cmd_name)
      File "/usr/lib/python3.6/distutils/cmd.py", line 313, in run_command
        self.distribution.run_command(command)
      File "/usr/lib/python3.6/distutils/dist.py", line 974, in run_command
        cmd_obj.run()
      File "/tmp/pip-req-build-8zsfe839/setup.py", line 103, in run
        self.run_command('build_frontend')
      File "/usr/lib/python3.6/distutils/cmd.py", line 313, in run_command
        self.distribution.run_command(command)
      File "/usr/lib/python3.6/distutils/dist.py", line 974, in run_command
        cmd_obj.run()
      File "/tmp/pip-req-build-8zsfe839/setup.py", line 90, in run
        subprocess.check_call(build_frontend_command[platform.system()], shell=True)
      File "/usr/lib/python3.6/subprocess.py", line 311, in check_call
        raise CalledProcessError(retcode, cmd)
    subprocess.CalledProcessError: Command 'frontend/gradlew -p frontend clean assemble' returned non-zero exit status 1.
```
## Solution
I figured out the jdk didn't build the image properly. So I managed to change the jdk version in the dockerfile from **openjdk-8-jdk-headless** to **openjdk-11-jdk**, after that I retried, and it worked perfectly.