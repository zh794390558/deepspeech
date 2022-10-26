cd /media/data/jpiasecki/Paddle/build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_PYTHON=ON -DPY_VERSION=3.9 -DWITH_GPU=OFF -DWITH_MKLDNN=ON -DWITH_MKL=ON -DWITH_AVX=ON -DWITH_TESTING=OFF -DON_INFER=ON ..
#cp -rf /media/data/jpiasecki/gflags_tmp/* /media/data/jpiasecki/Paddle/build/third_party/
make -j192
#mkdir -p /media/data/jpiasecki/gflags_tmp/gflags
#mkdir -p /media/data/jpiasecki/gflags_tmp/install/gflags
#cp -rf /media/data/jpiasecki/Paddle/build/third_party/gflags/* /media/data/jpiasecki/gflags_tmp/gflags/
#cp -rf /media/data/jpiasecki/Paddle/build/third_party/install/gflags/* /media/data/jpiasecki/gflags_tmp/install/gflags/
rm -rf /media/data/jpiasecki/Paddle/build/third_party/gflags /media/data/jpiasecki/Paddle/build/third_party/install/gflags
pip install --force-reinstall /media/data/jpiasecki/Paddle/build/python/dist/*
cp /home/jpiasecki/paddle_build/test/u2/sysconfig.py.bak /media/data/jpiasecki/venv_39/lib/python3.9/site-packages/paddle/sysconfig.py
