FROM nvcr.io/nvidia/pytorch:24.01-py3
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "zsh"]
RUN curl -s  https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN ["apt-get", "install", "-y", "git-lfs"]
#RUN ["apt-get", "install", "-y", "git"]
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --upgrade
WORKDIR /workspace/
