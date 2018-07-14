FROM ubuntu:16.04

MAINTAINER osanai

ENV TZ Asia/Tokyo

RUN rm -rf /var/lib/apt/lists/* && \
  apt-get clean && \
  apt-get update -y && \
  apt-get install -y sudo tzdata && \
  useradd -m app-user && \
  echo 'app-user:password' | chpasswd && \
  echo "app-user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
  echo "${TZ}" > /etc/timezone && \
  rm /etc/localtime && \
  ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
  dpkg-reconfigure -f noninteractive tzdata
USER app-user

ENV DEBCONF_NOWARNINGS yes
ENV PATH $PATH:/usr/local/bin

RUN sudo apt-get -y install \
  git \
  make \
  build-essential \
  python-dev \
  python-pip \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  libglib2.0-0 \
  libfontconfig1 \
  libxrender1 \
  libsm6 \
  libxext6 \
  vim \
  curl \
  wget \
  tar 
ENV PYTHONIOENCODING "utf-8"
ENV DOCKERIZE_VERSION v0.6.0

COPY . /home/app-user/work
WORKDIR /home/app-user/work

ENV HOME  /home/app-user
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN sudo apt-get install -y locales && \
  sudo locale-gen en_US.UTF-8 && \
  git clone https://github.com/yyuu/pyenv.git ~/.pyenv  && \
  pyenv install 3.6.3 && \
  pyenv global 3.6.3 && \
  pip install -r requirement.txt && \
  jupyter contrib nbextension install --user && \
  jupyter nbextension enable codefolding/main && \
  jupyter contrib nbextensions migrate && \
  jupyter nbextension enable --py widgetsnbextension --sys-prefix
ENV LANG 'en_US.UTF-8'
ENV LC_LANG 'en_US.UTF-8'
ENV LC_ALL 'en_US.UTF-8'
WORKDIR /home/app-user/work/notebook
ENTRYPOINT [ "jupyter", "notebook", "--ip=0.0.0.0", "--port=8080", "--NotebookApp.token='token'" ]