FROM registry.access.redhat.com/ubi9

RUN dnf -y install git pip python3-devel

#Install the cuda-toolkit
ADD https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-rhel9-12-2-local-12.2.2_535.104.05-1.x86_64.rpm /tmp
RUN rpm -i /tmp/cuda-repo-rhel9-12-2-local-12.2.2_535.104.05-1.x86_64.rpm
RUN dnf -y install cuda-toolkit
RUN rm -f /tmp/cuda-repo-rhel9-12-2-local-12.2.2_535.104.05-1.x86_64.rpm

#clone the Microsoft Trellis repository
WORKDIR /
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
WORKDIR /TRELLIS

RUN pip install wheel
ENV PATH="/usr/local/cuda-12.2/bin/:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.9/site-packages/nvidia/cudnn/lib/:/usr/local/lib/python3.9/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
# NOTE: Next line is because setup.sh does not currently install kaolin for torch-2.5.0
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu121.html
RUN . ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
RUN . ./setup.sh --demo
# Download the models: 
RUN echo -e 'from trellis.pipelines import TrellisImageTo3DPipeline \nTrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")' | python

EXPOSE 7860
CMD ["python", "app.py"]
