# DISNET: Distributed Micro-Split Deep Learning in Heterogeneous Dynamic IoT
### MICRO-SPLIT DEEP LEARNING
Distributed micro-split deep learning enables flexible partitioning and distributed computing of deep neural networks in heterogeneous dynamic IoT for low-latency and energy-efficient cooperative Deep Learning operations. DISNET iteratively combines vertical (layerwise) and horizontal DNN partitioning for flexible, distributed, and parallel execution of neural networks in IoT systems with heterogenous computing capabilities and network conditions without compromising accuracy.

### Code Structure

The repository contains the source code of DiSNet. The code is organised as: 
The folder `main` contains the files for the core implementation of the DiSNet framework. 
Test deep neural network models are included in the `models` folder.
The file `network.py` contains the implementation of the IoT mesh networks for testing, saved in `networks` folder.
In the file `main/model_convert.py` modify the first part of the model_path
`model_path = "/home/eric/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth"`.

### Running

Create an instance of the Vgg-16 model by running the `model_vgg16.py` file located in the `main` folder:

```
python3 models/model_vgg16.py
```

To generate the parameters of a pre-trained model the `model_convert.py` file.

```
python3 main/model_convert.py
```

Run the file `run_DiSNet.py` to test the inference time, energy consumption and inference accuracy.

```
python3 main/run_DiSNet.py
```
