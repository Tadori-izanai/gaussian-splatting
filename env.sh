# conda create -n ags python=3.9
# conda activate ags

conda install -y pytorch=2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install tqdm
pip install plyfile
pip install scipy
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
