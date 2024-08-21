# Start from PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install COLMAP dependencies and gsplat requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository universe \
    && add-apt-repository multiverse \
    && apt-get update && apt-get install -y --no-install-recommends \
    imagemagick \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    libglm-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA-related environment variables
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Print CUDA version to confirm
RUN nvcc --version

# Clone and build COLMAP
# WORKDIR /
# RUN git clone https://github.com/colmap/colmap.git
# WORKDIR /colmap
# RUN mkdir build
# WORKDIR /colmap/build
# RUN cmake .. \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_INSTALL_PREFIX=/usr/local \
#     -DCUDA_ENABLED=ON \
#     -DCMAKE_CUDA_ARCHITECTURES="75;86"
# RUN make -j$(nproc)
# RUN make install

# # Clone and install gsplat with specific CUDA architecture flags
# Copy the current directory to /gsplat
COPY . /gsplat
WORKDIR /gsplat

# Set TORCH_CUDA_ARCH_LIST for compatibility
ENV TORCH_CUDA_ARCH_LIST="7.5;8.6"

# Install the current directory
RUN pip install --no-cache-dir -v .

# Install additional dependencies
RUN pip install matplotlib plyfile

# Install requirements from examples directory
RUN cd examples && pip install -r requirements.txt

# Set working directory to root
WORKDIR /gsplat

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash"]