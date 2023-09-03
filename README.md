# DISNET: Distributed Micro-Split Deep Learning in Heterogeneous Dynamic IoT
### MICRO-SPLIT DEEP LEARNING
Distributed micro-split deep learning enables flexible partitioning and distributed computing of deep neural networks in heterogeneous dynamic IoT for low-latency and energy-efficient cooperative DL. DISNET accelerates inference time and minimizes energy consumption by combining vertical (layer-based) and horizontal DNN partitioning for flexible, distributed, and parallel execution of neural network models. DISNET considers the IoT devices’ computing and communication resources and the network conditions for resource-aware distributed ML.

### Code Structure
model_path = "/home/eric/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth"

### Running
- Use `model_vgg16` to implemet the model of Vgg-16.
python3 models/model_vgg16.py

- Use `model_convert` to generate the parameter of pre-trained model.
python3 main/model_convert.py

- Run `run_DiSNet` to test the inference time and inference accuracy of DiSNet and MoDNN.
python3 main/run_DiSNet.py