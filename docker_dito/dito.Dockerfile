FROM fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023

# install baseline agent requirements
RUN /bin/bash -c "\
    . activate home-robot \
    && cd home-robot \
    && git submodule update --init --recursive src/third_party/detectron2 \
    src/home_robot/home_robot/perception/detection/detic/Detic \
    src/third_party/contact_graspnet \
    && pip install -e src/third_party/detectron2 \
    && pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt \
    && pip install -e src/home_robot \
    "

# download pretrained Detic checkpoint
RUN mkdir -p home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models && \
    wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
    -O home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
    --no-check-certificate

# download pretrained skills
RUN /bin/bash -c "\
    mkdir -p home-robot/data/checkpoints \
    && cd home-robot/data/checkpoints \
    && wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ovmm_baseline_home_robot_challenge_2023.zip \
    -O ovmm_baseline_home_robot_challenge_2023.zip \
    --no-check-certificate \
    && unzip ovmm_baseline_home_robot_challenge_2023.zip \
    && rm ovmm_baseline_home_robot_challenge_2023.zip \
    "

#Change Work directory, so commands execute from within the folder
WORKDIR /home-robot

RUN /bin/bash -c "\
    . activate home-robot \
    && conda update -n base conda \
    && conda install -n base conda-libmamba-solver \
    && conda config --set solver libmamba \   
    && conda config --add channels conda-forge \
    "
# cuda 11.7
RUN /bin/bash -c "\
    . activate home-robot \ 
    && conda env config vars set CUDA_HOME=/usr/local/cuda-12.1 \
    && conda install flax=0.6.1 \
    && conda install jax=0.4.13 \
    "
RUN /bin/bash -c "\
    . activate home-robot \
    && CONDA_OVERRIDE_CUDA="12.1" conda install -y cudnn=8.8.0.121=h838ba91_4\
    && CONDA_OVERRIDE_CUDA="12.1" conda install -y tensorflow=2.14.0=cuda118py39hc3a5e0e_0 \
    && pip install tensorflow-text==2.14.0 \
    "

RUN /bin/bash -c "\
    . activate home-robot \ 
    && CONDA_OVERRIDE_CUDA="12.1" conda install -y pytorch3d=0.7.5=cuda112py39hcd8cc04_0\
    "

RUN /bin/bash -c "\
    . activate home-robot \
    && pip install --force-reinstall torch_geometric \
    && pip install --force-reinstall torch_cluster \
    "

COPY /storage-01/ml-stud-share/ml-stud21/google-research/fvlm /home-robot/src/home_robot/home_robot/perception/detection/fvlm

ADD /storage-01/ml-stud-share/ml-stud21/docker_dito/dito_perception.py /home-robot/src/home_robot/home_robot/perception/detection/fvlm/dito_perception.py

#For navigation via bert embeddings
#RUN /bin/bash -c "\
#    . activate home-robot \
#    && pip install transformers \
#    "

# # add baseline agent code
# ADD projects/habitat_ovmm/eval_baselines_agent.py /home-robot/projects/habitat_ovmm/agent.py
# ADD projects/habitat_ovmm/extra_code/esc_agent.py /home-robot/src/home_robot/home_robot/agent/ovmm_agent/esc_agent.py
# ADD projects/habitat_ovmm/extra_code/heuristic_agent_24.yaml /home-robot/projects/habitat_ovmm/configs/agent/heuristic_agent.yaml
# ADD projects/habitat_ovmm/extra_code/ovmm_perception.py /home-robot/src/home_robot/home_robot/agent/ovmm_agent/ovmm_perception.py
# ADD projects/habitat_ovmm/evaluator.py /home-robot/projects/habitat_ovmm/evaluator.py

# #add floor height check 
# ADD projects/habitat_ovmm/extra_code/objectnav_agent_module.py /home-robot/src/home_robot/home_robot/agent/objectnav_agent/objectnav_agent_module.py
# ADD projects/habitat_ovmm/extra_code/categorical_2d_semantic_map_module.py /home-robot/src/home_robot/home_robot/mapping/semantic/categorical_2d_semantic_map_module.py

# #Fix to have consistend episodes every run
# ADD projects/habitat_ovmm/extra_code/dataset.py /home-robot/src/third_party/habitat-lab/habitat-lab/habitat/core/dataset.py

# # add submission script
# ADD projects/habitat_ovmm/scripts/submission.sh /home-robot/submission.sh

# # set number of max steps to 1000
# ADD projects/habitat_ovmm/extra_code/ovmm.yaml /home-robot/src/third_party/habitat-lab/habitat-lab/habitat/config/benchmark/ovmm/ovmm.yaml

# # add new navigation policy
# ADD projects/habitat_ovmm/extra_code/esc_frontier_exploration_policy.py /home-robot/src/home_robot/home_robot/navigation_policy/object_navigation/esc_frontier_exploration_policy.py

# #Run with correct cudnn version
# ENV LD_LIBRARY_PATH /opt/conda/envs/home-robot/lib:$LD_LIBRARY_PATH

# # set evaluation type to remote
# ENV AGENT_EVALUATION_TYPE remote

# # additional command line arguments for local evaluations (empty for remote evaluation)
# ENV LOCAL_ARGS ""

# # run submission script
# CMD /bin/bash -c "\
#     . activate home-robot \
#     && cd /home-robot \
#     && export PYTHONPATH=/evalai_remote_evaluation:$PYTHONPATH \
#     && bash submission.sh $LOCAL_ARGS \
#     "
