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