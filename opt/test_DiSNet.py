import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import math
from rf_vgg import ReceptiveFieldCalculator
from rf_DiSNet import ReceptiveFieldCalculatorDiSNet
from torchvision import transforms
from models.model_vgg16 import VGG16
from infer import opt_flp,opt_modnn, opt_DiSNet,get_partiton_info,get_partiton_info_DiSNet

# load image
filename = ("data/dog.jpg")
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# setup parameters
trans_rate = [40, 60, 80, 100] # Gbps
# num_sever = [3,4,5,6,7,8,9,10]
num_sever = [3]

# load model
model = VGG16()
model_dict = model.state_dict()
model_dict.update(torch.load("opt/vgg16-modify.pth"))
model.load_state_dict(model_dict)
model.eval()

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda:0')
    model.to('cuda:0')

# # inference HALP
# for i in range(0,len(num_sever)):
#     print("--------------------------------------------------------")
#     print("server:",num_sever[i])
#     with torch.no_grad():
        
#         infer_time, trans_time, output = opt_flp(input_batch, trans_rate[0], num_sever[i],model) 
#         probabilities = torch.nn.functional.softmax(output[0], dim=0)
#         # print(probabilities)
#         print('infer time ', infer_time)
#         print('trans time ', trans_time)


#         with open("opt/imagenet_classes.txt", "r") as f:
#             categories = [s.strip() for s in f.readlines()]
#         # Show top categories per image
#         top5_prob, top5_catid = torch.topk(probabilities, 5)
#         for k in range(top5_prob.size(0)):
#             print(categories[top5_catid[k]], top5_prob[k].item())    


# print("--------------------------------------------------------")

# #inference HALP
# for i in range(0,len(num_sever)):
#     with torch.no_grad():
#         for j in range(0,len(trans_rate)):
#             infer_time, trans_time, output = opt_flp(input_batch, trans_rate[j], num_sever[i],model) 
#             probabilities = torch.nn.functional.softmax(output[0], dim=0)
#             # print(probabilities)
#             print("trans_rate HARP",trans_rate[j])
#             print('infer time ', infer_time)
#             print('trans time ', trans_time)
            
#             with open("opt/imagenet_classes.txt", "r") as f:
#                 categories = [s.strip() for s in f.readlines()]
#             # Show top categories per image
#             top5_prob, top5_catid = torch.topk(probabilities, 5)
#             for k in range(top5_prob.size(0)):
#                 print(categories[top5_catid[k]], top5_prob[k].item())  
#             print("--------------------------------------------------------")            

# inference modnn
for i in range(0,len(num_sever)):
    partition_input = []
    for m in range(0,18):
        partition = get_partiton_info(m,m,num_sever[i]) #how to split each layer
        partition_input.append(partition)
    #     print(partition)
    # print(partition_input)
    with torch.no_grad():
        for j in range(0,len(trans_rate)):
            output,infer_time = opt_modnn(input_batch, partition_input, trans_rate[j], model)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            # print(probabilities)
            print("trans_rate ",trans_rate[j])
            print("--------------------------------------------------------")
            print('infer time ', infer_time)
            print("--------------------------------------------------------")


            with open("opt/imagenet_classes.txt", "r") as f:
                categories = [s.strip() for s in f.readlines()]
            # Show top categories per image
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            for k in range(top5_prob.size(0)):
                print(categories[top5_catid[k]], top5_prob[k].item()) 
            print("--------------------------------------------------------")

# inference DiSNet
for i in range(0,len(num_sever)):
    partition_input = []
    layer_range = []
    for m in range(0,18):
        
        partition = get_partiton_info_DiSNet(m,m,num_sever[i]) #how to split each layer
        partition_input.append(partition)
    #     print(partition)

    layer_range = np.array([[0,3],[3,7],[7,10],[10,15],[15,18]])
    # print(partition_input)
    trans_rate = [40, 60, 80, 100, 120] # Gbps

    output = input_batch
    infer_time = []
    with torch.no_grad():
        for j in range(0,len(trans_rate)):

            print(partition_input[layer_range[j,0]:layer_range[j,1]])
            output,sub_infer_time = opt_DiSNet(output, layer_range[j], partition_input[layer_range[j,0]:layer_range[j,1]], trans_rate[j], model)
            # print("Output",output.shape)
            # print(probabilities)
            print("sub trans_rate ",trans_rate[j])
            print('sub infer time ', sub_infer_time)
            infer_time.append(sub_infer_time)
            print("--------------------------------------------------------")
                    
        print('End to end inference time ', np.sum(infer_time))
        print("--------------------------------------------------------")

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        with open("opt/imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for k in range(top5_prob.size(0)):
            print(categories[top5_catid[k]], top5_prob[k].item()) 