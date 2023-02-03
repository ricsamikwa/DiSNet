import torch
import torch.nn as nn
import math
import time
import sys
import numpy as np
import os
from field_vgg import FieldCalculator
from field_DiSNet import FieldCalculatorDiSNet
import configurations
device_pace_rate = configurations.rate

o_path = os.getcwd()
sys.path.append(o_path)

 #this needs investigation

def infer_block(in_tensor, start_layer, end_layer, model):
    cmp_time = []
    out_tesor = []
    with torch.no_grad():
        for i in range(0,len(in_tensor)):
            start_infer = time.time()
            # for k in range(0,1000):
            #     for j in range(start_layer,end_layer+1):

            #         in_tensor[i] = model([in_tensor[i],j])
            for j in range(start_layer,end_layer+1):

                    in_tensor[i] = model([in_tensor[i],j])        
            end_infer = time.time()


            # cmp_time.append((end_infer-start_infer)/1000)
            cmp_time.append(end_infer-start_infer)
            out_tesor.append(in_tensor[i])
    # print(cmp_time)
    return out_tesor,cmp_time

def infer_layer(in_tensor,model,layer_num):
    start_infer = 0
    end_infer = 0
    
    with torch.no_grad():
        start_infer = time.time()
        # out_tensor = model([in_tensor,layer_num])

        for i in range(0,10):
            out_tensor = model([in_tensor,layer_num])
        end_infer = time.time()
    # infer_time = end_infer-start_infer
    infer_time = (end_infer-start_infer)/10

    return out_tensor, infer_time

def layer_infer(in_tensor,model,layer_num):
    start_infer = time.time()
    with torch.no_grad():
        out_tensor = model([in_tensor,layer_num])

        # for i in range(0,1000):
        #     out_tensor = model([in_tensor,layer_num])
    end_infer = time.time()
    infer_time = end_infer-start_infer
    # infer_time = (end_infer-start_infer)/1000

    return out_tensor, infer_time
def get_partiton_info(start_layer, end_layer, par_num):
    # print(start_layer, end_layer, par_num)
    calculator = FieldCalculator(224, start_layer, end_layer, par_num)
    partition = calculator.input
    return partition

def get_partiton_info_DiSNet(start_layer, end_layer, split_ratio):
    # print(start_layer, end_layer, par_num)
    calculator = FieldCalculatorDiSNet(224, start_layer, end_layer, len(split_ratio), split_ratio)
    partition = calculator.input
    # print(partition)
    return partition

class Opt_Par:
    def __init__(self,trans_rate, num_sever, in_img, model, init_value, flag):
        super(Opt_Par, self).__init__()
        self.trans_rate = trans_rate
        self.num_sever = num_sever
        self.model = model
        self.opt_par = []
        self.init_value = init_value
        self.in_img = in_img
        self.flag = flag
    def getOptInfo(self):
        _ = self.opt_point(0,17)
        opt_partion = self.opt_block(self.opt_par)
        return opt_partion

    def time_exe(self, start_layer, end_layer):
        if start_layer == 0:
            in_tensor = self.in_img
        else:
            in_tensor = self.in_img
            for i in range(0,start_layer):
                in_tensor= self.model([in_tensor,i])
        # input index of each 
        partition,_ = get_partiton_info(start_layer, end_layer, self.num_sever)
        # input sub_tensor of each 
        in_par =[]
        for i in range(0,len(partition)):
            if partition[i]==[0,0]:
                pass
            else:
                in_par.append(in_tensor[:,:,partition[i][0]:partition[i][1]+1,:])
        # transmission time
        in_size = in_tensor.size()
        trans_data = 0
        if start_layer ==0:
            for i in range(0,len(partition)):
                if partition[i]==[0,0]:
                    pass
                else:
                    trans_data = trans_data + 32*in_size[1]*(partition[i][1]+1-partition[i][0])*in_size[3]
        else:
            for i in range(0,len(partition)):
                if partition[i]==[0,0]:
                    pass
                else:
                    trans_data = trans_data + 32*in_size[1]*(partition[i][1]+1-partition[i][0])*in_size[3]
            trans_data = trans_data -32*in_size[1]*in_size[2]*in_size[3] 
        # computing time
        out_sub,t_cmp = infer_block(in_par,start_layer,end_layer,self.model)
        if end_layer == 17:
            dim_sub_out = out_sub[0].size()
            trans_data = trans_data+32*dim_sub_out[1]*(dim_sub_out[3]+1-dim_sub_out[2])*dim_sub_out[3]
        t_com = trans_data/(1024*1024*1024*self.trans_rate)
        infer_time = t_com + max(t_cmp)
        return infer_time

    def opt_block(self,s):
        s.sort(reverse=False)
        s_e = []
        flag = -1
        for i in range(0, len(s)):
            if i == flag:
                continue
            if i == 0:
                s_e.append([0, s[i]])
                continue
            if s[i] != s[i - 1] + 1:
                start = s[i - 1] + 1
                end = s[i]
                s_e.append([start, end])
            else:
                if i != len(s) - 1:
                    start = s[i]
                    if s[i] == s[i + 1] - 1:
                        end = s[i]
                    else:
                        end = s[i + 1]
                        flag = i + 1
                    s_e.append([start, end])
                else:
                    s_e.append([s[i], s[i]])
        start = s[len(s) - 1]
        s_e.append([start + 1, 17])
        return s_e

    def opt_point(self, start_layer, end_layer):
        if start_layer == end_layer:
            return self.time_exe(start_layer, end_layer)
        t_min = sys.maxsize
        opt_point = 18
        for k in range(1, end_layer - start_layer + 1):
            if self.flag[start_layer][start_layer + k] == 0:
                self.init_value[start_layer][start_layer + k] = self.time_exe(start_layer, start_layer + k)
                self.flag[start_layer][start_layer + k] = 1
            temp = self.init_value[start_layer][start_layer + k] + self.opt_point(start_layer + k + 1, end_layer)
            if temp < t_min:
                t_min = temp
                opt_point = start_layer + k
        if opt_point != 18:
            if opt_point not in self.opt_par:
                self.opt_par.append(opt_point)
        return t_min

##################### workig here #######################
def opt_DiSNet(in_img, layer_range, input_index, trans_rate, comp_rate, split_ratio, model):
    # layers = 10
    # layers = 21 

    norm_comp_rate = comp_rate / np.linalg.norm(comp_rate)
    norm_split_ratio = comp_rate / np.linalg.norm(split_ratio)

    # print("norm_comp_rate: ",norm_comp_rate)
    # print("norm_split_ratio: ",norm_split_ratio)

    
    layers = layer_range[1]
    if layer_range[1] == 18:
        layers = 21  
    layers_iter = layers - layer_range[0]

    layers_start = layer_range[0]
    # input size [s1,s2,h1,kernel_size]
    # kerner_size = [3,3,2,3,3,2,3,3,3,2,3,3,3,2,3,3,3,2,0,0,0]
    kerner_size = [3,3,2,3,3,2,3,3,3,2,3,3,3,2,3,2,2,2,0,0,0]

    # Total inference time t_h: host inference time t_p: secondary ES inference time
    t_CLs = 0
    t_com = 0
    t_FLs = 0
    in_tensor = in_img
    # print(in_tensor.shape)
    # print(input_index)
    # print(layers_start)

    with torch.no_grad():

        # Transmission Scheduling
        for i in range(0,layers_iter):
            
            p = layers_start + i
            # print(p)
            if p < layers-3:
                out_tensor = []
                t_sub = []
                t_sub_com = []
                # print(i)
                # print(input_index[i][0])
                for j in range(0,len(input_index[i][0])):
                    # print(len(input_index[i][0]))
                    t_sub_rec = 0
                    t_sub_send = 0

                    if input_index[i][1][j] == 0:
                        break
                    # print(input_index[i][0][j][0],input_index[i][0][j][1]+1)
                    in_sub = in_tensor[:,:,input_index[i][0][j][0]:input_index[i][0][j][1]+1,:]
                    inputsize_sub = in_sub.size()
                    
                    if i == 0:
                        # print('receiving data for partition')
                        t_sub_rec = 32*inputsize_sub[1]*inputsize_sub[2]*inputsize_sub[3]/(1024*1024*trans_rate)
                    
                    # say this is running on diffent devices at different speeds
                    output_sub, t_sub_cmp = infer_layer(in_sub, model, p)
                    # print(output_sub.shape)
                    outputsize_sub = output_sub.size()
                    # print(outputsize_sub)

                    #this ------- shit
                    if kerner_size[p] == 3:
                        
                        if j not in [0,8]:
                            ### sychronication part not per layer!!!
                            # if i == layers_iter - 1:
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-2)*outputsize_sub[3]/(1024*1024*trans_rate)
                            out_tensor.append(output_sub[:,:,1:-1,:])
                        elif j == 0:
                            out_tensor.append(output_sub[:,:,:-1,:])
                            # if i == layers_iter - 1:
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-1)*outputsize_sub[3]/(1024*1024*trans_rate)
                        else:
                            out_tensor.append(output_sub[:,:,1:,:])
                            # if i == layers_iter - 1:
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-1)*outputsize_sub[3]/(1024*1024*trans_rate)
                    elif kerner_size[p] == 2:

                        # if i == layers_iter - 1:
                        t_sub_send = 32*outputsize_sub[1]*outputsize_sub[2]*outputsize_sub[3]/(1024*1024*trans_rate)
                        out_tensor.append(output_sub)
                    
                    #this is per layer 
                    t_sub_cmp_proportional = (1 + (abs(split_ratio[j]-comp_rate[j])/comp_rate[j]))*t_sub_cmp*device_pace_rate#times slowness comparison
                    # print(t_sub_cmp_proportional, t_sub_cmp, t_sub_rec, t_sub_send) # more checks later
                    # t_sub.append(t_sub_rec + t_sub_cmp + t_sub_send)
                    t_sub.append(t_sub_rec + t_sub_cmp_proportional + t_sub_send)

                    t_sub_com.append(t_sub_rec + t_sub_send)
                in_tensor = out_tensor[0]
                for i in range(len(out_tensor)-1):
                    in_tensor = torch.cat([in_tensor,out_tensor[i+1]],dim=2)
                t_com = t_com + max(t_sub_com) # max of the data exchange time for each layer (send and receiving) - not used 
                t_CLs = t_CLs + max(t_sub) # max of the processing time
            else:
                #last layers on the last device 
                output_tensor, t_fl = infer_layer(in_tensor, model, p)
                # print(output_tensor.shape)

                in_tensor = output_tensor
                t_FLs = t_FLs + t_fl
        t = t_CLs + t_FLs
        # print("this is t", t)
    return output_tensor, t/device_pace_rate

# MoDNN
def opt_modnn(in_img, input_index, trans_rate,comp_rate_modnn, model):
    # layers = 10
    layers = 21 
    # input size [s1,s2,h1,kernel_size]
    # kerner_size = [3,3,2,3,3,2,3,3,3,2,3,3,3,2,3,3,3,2,0,0,0]
    kerner_size = [3,3,2,3,3,2,3,3,3,2,3,3,3,2,3,2,2,2,0,0,0]

    # Total inference time t_h: host inference time t_p: secondary ES inference time
    t_CLs = 0
    t_com = 0
    t_FLs = 0
    in_tensor = in_img
    # print(in_tensor.shape)
    # print(input_index)
    with torch.no_grad():
        # Transmission Scheduling
        for i in range(0,layers):
            if i < layers-3:
                out_tensor = []
                t_sub = []
                t_sub_com = []
                # print(input_index[i][0])
                for j in range(0,len(input_index[i][0])):
                    if input_index[i][1][j] == 0:
                        break
                    in_sub = in_tensor[:,:,input_index[i][0][j][0]:input_index[i][0][j][1]+1,:]
                    inputsize_sub = in_sub.size()
                    
                    t_sub_rec = 32*inputsize_sub[1]*inputsize_sub[2]*inputsize_sub[3]/(1024*1024*trans_rate)
                    # print(in_sub.shape, i)

                    output_sub, t_sub_cmp = layer_infer(in_sub, model, i)
                    
                    outputsize_sub = output_sub.size()

                    if kerner_size[i] == 3:
                        if j not in [0,8]:
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-2)*outputsize_sub[3]/(1024*1024*trans_rate)
                            out_tensor.append(output_sub[:,:,1:-1,:])
                        elif j == 0:
                            out_tensor.append(output_sub[:,:,:-1,:])
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-1)*outputsize_sub[3]/(1024*1024*trans_rate)
                        else:
                            out_tensor.append(output_sub[:,:,1:,:])
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-1)*outputsize_sub[3]/(1024*1024*trans_rate)
                    elif kerner_size[i] == 2:
                        t_sub_send = 32*outputsize_sub[1]*outputsize_sub[2]*outputsize_sub[3]/(1024*1024*trans_rate)
                        out_tensor.append(output_sub)
                                        
                    # t_sub_cmp_proportional = (1 + (abs(comp_rate_modnn[j]-comp_rate_modnn[j])/comp_rate_modnn[j]))*t_sub_cmp*DEVICE_PACE_RATE#times slowness comparison

                    t_sub_cmp_proportional = (1 + abs(1/len(comp_rate_modnn) - (comp_rate_modnn[j]/sum(comp_rate_modnn))))*t_sub_cmp*device_pace_rate#times slowness comparison
                    # print(t_sub_cmp_proportional, t_sub_cmp, t_sub_rec, t_sub_send) # more checks later
                    # t_sub.append(t_sub_rec + t_sub_cmp + t_sub_send)
                    t_sub.append(t_sub_rec + t_sub_cmp_proportional + t_sub_send)
                    # t_sub.append(t_sub_rec + t_sub_cmp + t_sub_send)
                    t_sub_com.append(t_sub_rec + t_sub_send)
                in_tensor = out_tensor[0]
                for i in range(len(out_tensor)-1):
                    in_tensor = torch.cat([in_tensor,out_tensor[i+1]],dim=2)
                t_com = t_com + max(t_sub_com)
                t_CLs = t_CLs + max(t_sub)
            else:
                output_tensor, t_fl = infer_layer(in_tensor, model, i)
                # print(output_tensor.shape)

                in_tensor = output_tensor
                t_FLs = t_FLs + t_fl
        t = t_CLs + t_FLs
        # print("this is t", t)
    return output_tensor, t/device_pace_rate, t_com/device_pace_rate


# DeepSlicing
def opt_deepsclicing(in_img, input_index, trans_rate,pos_max_par_partitions,comp_rate_modnn, model):
    # layers = 10
    layers = 21 
    # input size [s1,s2,h1,kernel_size]
    # kerner_size = [3,3,2,3,3,2,3,3,3,2,3,3,3,2,3,3,3,2,0,0,0]
    kerner_size = [3,3,2,3,3,2,3,3,3,2,3,3,3,2,3,2,2,2,0,0,0]

    # Total inference time t_h: host inference time t_p: secondary ES inference time
    t_CLs = 0
    t_com = 0
    t_FLs = 0
    in_tensor = in_img
    # print(in_tensor.shape)
    # print(input_index)
    with torch.no_grad():
        # Transmission Scheduling
       
        pointer = 0
        for i in range(0,layers):
             
            das_flag = False
            t_sub_rec = 0
            t_sub_send = 0
            
            if i < layers-3:
                out_tensor = []
                t_sub = []
                t_sub_com = []
                # print(input_index[i][0])
               
                
                for j in range(0,len(input_index[i][0])):

                    
                    if input_index[i][1][j] == 0:
                        break
                    in_sub = in_tensor[:,:,input_index[i][0][j][0]:input_index[i][0][j][1]+1,:]
                    inputsize_sub = in_sub.size()
                    
                    if i == pos_max_par_partitions[pointer] | i == 0:
                        t_sub_rec = 32*inputsize_sub[1]*inputsize_sub[2]*inputsize_sub[3]/(1024*1024*trans_rate)
                    # print(in_sub.shape, i)

                    output_sub, t_sub_cmp = layer_infer(in_sub, model, i)
                    
                    outputsize_sub = output_sub.size()

                    if kerner_size[i] == 3:
                        if j not in [0,8]:
                            # if i == pos_max_par_partitions[pointer]:
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-2)*outputsize_sub[3]/(1024*1024*trans_rate)
                            out_tensor.append(output_sub[:,:,1:-1,:])
                        elif j == 0:
                            out_tensor.append(output_sub[:,:,:-1,:])
                            # if i == pos_max_par_partitions[pointer]:
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-1)*outputsize_sub[3]/(1024*1024*trans_rate)
                        else:
                            out_tensor.append(output_sub[:,:,1:,:])
                            # if i == pos_max_par_partitions[pointer]:
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-1)*outputsize_sub[3]/(1024*1024*trans_rate)
                    elif kerner_size[i] == 2:
                        # if i == pos_max_par_partitions[pointer]:
                        t_sub_send = 32*outputsize_sub[1]*outputsize_sub[2]*outputsize_sub[3]/(1024*1024*trans_rate)
                        out_tensor.append(output_sub)
                                                                                
                    t_sub_cmp_proportional = (1 + 0)*t_sub_cmp*device_pace_rate#times slowness comparison

                    # t_sub_cmp_proportional = (1 + abs(1/3 - (comp_rate_modnn[j]/sum(comp_rate_modnn))))*t_sub_cmp*DEVICE_PACE_RATE#times slowness comparison
                    
                    # print(t_sub_cmp_proportional, t_sub_cmp, t_sub_rec, t_sub_send) # more checks later
                    # t_sub.append(t_sub_rec + t_sub_cmp + t_sub_send)
                    t_sub.append(t_sub_rec + t_sub_cmp_proportional + t_sub_send)
                    # t_sub.append(t_sub_rec + t_sub_cmp + t_sub_send)
                    t_sub_com.append(t_sub_rec + t_sub_send)
                in_tensor = out_tensor[0]
                for i in range(len(out_tensor)-1):
                    in_tensor = torch.cat([in_tensor,out_tensor[i+1]],dim=2)
                t_com = t_com + max(t_sub_com)
                t_CLs = t_CLs + max(t_sub)
            else:
                output_tensor, t_fl = infer_layer(in_tensor, model, i)
                # print(output_tensor.shape)

                in_tensor = output_tensor
                t_FLs = t_FLs + t_fl

            if i == pos_max_par_partitions[pointer]:
                pointer = pointer + 1
        t = t_CLs + t_FLs
        # print("this is t", t)
    return output_tensor, t/device_pace_rate, t_com/device_pace_rate