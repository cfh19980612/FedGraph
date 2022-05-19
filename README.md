# ML-Distribution
FedGraph: Federated Graph Learning with Intelligent Sampling

## Install requests
```
pip install -r requirements.txt
```
## Install dgl from source
### Step 1: Unzip dgl source files
```
tar -xvf dgl.tar
cd dgl-2/
```
### Step 2: Install latest cmake from source
(ref: https://www.cnblogs.com/yanqingyang/p/12731855.html)
```
apt-get remove cmake
tar -zxv -f cmake-3.20.tar.gz
./bootstrap
make
make install
```
### Step 3: CUDA build
```
mkdir build
cd build
cmake -DUSE_CUDA=ON ..
make -j4
```
### Step 4: Install the Python binding
```
cd ../python
python setup.py install
```
## Run FedGraph on Cora dataset using GPU
```
python FedGraph.py --dataset cora --gpu 1
```
