# conda create -n ags python=3.9
# conda activate ags

conda install -y pytorch=2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install tqdm
pip install plyfile
pip install scipy
pip install opencv-python einops kornia yacs
pip install pykeops geomloss
pip install open3d
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/RaDe-GS/submodules/diff-gaussian-rasterization
