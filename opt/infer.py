import torch
import torch.nn as nn
import math
import time
import sys
import os
from rf_vgg import ReceptiveFieldCalculator
from rf_DiSNet import ReceptiveFieldCalculatorDiSNet
o_path = os.getcwd()
sys.path.append(o_path)

DEVICE_PACE_RATE = 1

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
    calculator = ReceptiveFieldCalculator(224, start_layer, end_layer, par_num)
    partition = calculator.input
    return partition

def get_partiton_info_DiSNet(start_layer, end_layer, par_num, split_ratio):
    # print(start_layer, end_layer, par_num)
    calculator = ReceptiveFieldCalculatorDiSNet(224, start_layer, end_layer, par_num, split_ratio)
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
        # input index of each ES
        partition,_ = get_partiton_info(start_layer, end_layer, self.num_sever)
        # input sub_tensor of each ES
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

def opt_flp(in_img,trans_rate,num_sever,model):
    # optimal layer partition
    init_value = [[sys.maxsize for col in range(18)] for row in range(18)]
    flag = [[0 for col in range(18)] for row in range(18)]
    opt_block = Opt_Par(trans_rate, num_sever, in_img, model,init_value, flag).getOptInfo()
    # inference
    infer_time = 0
    sub_output = []
    trans_time = 0
    print(len(opt_block))
    for i in range(0,len(opt_block)):
        start_layer = opt_block[i][0]
        end_layer = opt_block[i][1]
        # input index of each ES
        partition,len_out = get_partiton_info(start_layer, end_layer, num_sever)
        # input sub_tensor of each ES
        if start_layer == 0:
            in_sub = []
            trans_data = 0
            in_size = in_img.size()
            print(len(partition))
            for j in range(0,len(partition)):
                if partition[j]==[0,0]:
                    pass
                in_sub.append(in_img[:,:,partition[j][0]:partition[j][1]+1,:])
                trans_data = trans_data+32*in_size[1]*(partition[j][1]+1-partition[j][0])*in_size[3]
            # computing time
            out_sub,t_cmp = infer_block(in_sub,start_layer,end_layer,model)
            if end_layer == 17:
                dim_sub_out = out_sub[0].size()
                trans_data = trans_data+32*dim_sub_out[1]*(sum(len_out)-len_out[0])*dim_sub_out[3]
            t_com = trans_data/(1024*1024*1024*trans_rate)
            infer_time = infer_time + t_com + max(t_cmp)
            trans_time = trans_time + t_com
            index_start = 0
            index_end = -1
            print(len(out_sub))
            for k in range(0,len(out_sub)):
                out_size = out_sub[k].size()
                if k == 0:
                    out_sub[k] = out_sub[k][:,:,:len_out[k]-out_size[2],:]
                elif k == len(out_sub)-1:
                    out_sub[k] = out_sub[k][:,:,out_size[2]-len_out[k]:,:]
                else:
                    out_sub[k] = out_sub[k][:,:,int((out_size[2]-len_out[k])/2):int((len_out[k]-out_size[2])/2),:]
                index_start = index_end + 1
                index_end = index_start + len_out[k]-1
                sub_output.append([out_sub[k], index_start, index_end])
        else:
            in_sub = []
            trans_data = 0
            for j in range(0,len(partition)):
                if partition[j]==[0,0]:
                    pass
                index_start_in = partition[j][0]
                index_end_in = partition[j][1]
                if j == 0:
                    in_sub_tensor = sub_output[j][0][:,:,0:min(index_end_in,sub_output[j][2])+1,:]
                    if index_end_in>sub_output[j][2]:
                        in_sub_tensor = torch.cat([in_sub_tensor,sub_output[j+1][0][:,:,:index_end_in-sub_output[j][2],:]],dim=2)
                        dim_sub_out = sub_output[j+1][0].size()
                        trans_data = trans_data + 32*dim_sub_out[1]*(index_end_in-sub_output[j][2])*dim_sub_out[3]
                elif j == len(partition)-1:
                    in_sub_tensor = sub_output[j][0][:,:,max(0,index_start_in-sub_output[j][1]):,:]
                    if index_start_in<sub_output[j][1]:
                        in_sub_tensor = torch.cat([in_sub_tensor,sub_output[j-1][0][:,:,index_start_in-sub_output[j][1]:,:]],dim=2)
                        dim_sub_out = sub_output[j-1][0].size()
                        trans_data = trans_data + 32*dim_sub_out[1]*(sub_output[j][1]-index_start_in)*dim_sub_out[3]
                else:
                    in_sub_tensor = sub_output[j][0][:,:,max(0,index_start_in-sub_output[j][1]):min(index_end_in-sub_output[j][1],sub_output[j][2]-sub_output[j][1])+1,:]
                    if index_end_in>sub_output[j][2]:
                        for m in range(j+1,len(sub_output)):
                            if index_end_in<sub_output[m][2]:
                                in_sub_tensor = torch.cat([in_sub_tensor,sub_output[m][0][:,:,:index_end_in-sub_output[j][2],:]],dim=2)
                                dim_sub_out = sub_output[m][0].size()
                                trans_data = trans_data + 32*dim_sub_out[1]*(index_end_in-sub_output[j][2])*dim_sub_out[3]
                                break
                            else:
                                in_sub_tensor = torch.cat([in_sub_tensor,sub_output[m][0][:,:,:,:]],dim=2)
                                dim_sub_out = sub_output[m][0].size()
                                trans_data = trans_data + 32*dim_sub_out[1]*dim_sub_out[2]*dim_sub_out[3]
                    if index_start_in<sub_output[j][1]:
                        for m in range(j-1,-1,-1):
                            if index_start_in > sub_output[m][1]:
                                in_sub_tensor = torch.cat([in_sub_tensor,sub_output[m][0][:,:,index_start_in-sub_output[j][1]:,:]],dim=2)
                                dim_sub_out = sub_output[m][0].size()
                                trans_data = trans_data + 32*dim_sub_out[1]*(sub_output[j][1]-index_start_in)*dim_sub_out[3]
                                break
                            else:
                                in_sub_tensor = torch.cat([in_sub_tensor,sub_output[m][0][:,:,:,:]],dim=2)
                                dim_sub_out = sub_output[m][0].size()
                                trans_data = trans_data + 32*dim_sub_out[1]*dim_sub_out[2]*dim_sub_out[3]
                in_sub.append(in_sub_tensor)
            # computing time
            out_sub,t_cmp = infer_block(in_sub,start_layer,end_layer,model)
            if end_layer == 17:
                dim_sub_out = out_sub[0].size()
                trans_data = trans_data+32*dim_sub_out[1]*(sum(len_out)-len_out[0])*dim_sub_out[3]
            t_com = trans_data/(1024*1024*1024*trans_rate)
            infer_time = infer_time + t_com + max(t_cmp)
            trans_time = trans_time + t_com
            index_start = 0
            index_end = -1
            sub_output = []
            for k in range(0,len(out_sub)):
                out_size = out_sub[k].size()
                if k == 0:
                    if len_out[k]-out_size[2] < 0:
                        out_sub[k] = out_sub[k][:,:,:len_out[k]-out_size[2],:]
                elif k == len(out_sub)-1:
                    out_sub[k] = out_sub[k][:,:,out_size[2]-len_out[k]:,:]
                else:
                    if len_out[k]-out_size[2] < 0:
                        out_sub[k] = out_sub[k][:,:,int((out_size[2]-len_out[k])/2):int((len_out[k]-out_size[2])/2),:]
                index_start = index_end + 1
                index_end = index_start + len_out[k]-1
                sub_output.append([out_sub[k], index_start, index_end])
    # compute the FLs
    in_fls = sub_output[0][0]
    for i in range(1,len(sub_output)):
        in_fls = torch.cat([in_fls,sub_output[i][0]],dim=2)
    start_infer = time.time()
    for k in range(0,10):
        in_tensor = in_fls
        for j in range(18,21):
            in_tensor = model([in_tensor,j])
    end_infer = time.time()
    t_fls = (end_infer-start_infer)/10
    infer_time = infer_time + t_fls
    return infer_time,trans_time,in_tensor

def opt_modnn(in_img, input_index, trans_rate, model):
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
                    
                    t_sub_rec = 32*inputsize_sub[1]*inputsize_sub[2]*inputsize_sub[3]/(1024*1024*1024*trans_rate)
                    # print(in_sub.shape, i)

                    output_sub, t_sub_cmp = infer_layer(in_sub, model, i)
                    
                    outputsize_sub = output_sub.size()

                    if kerner_size[i] == 3:
                        if j not in [0,8]:
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-2)*outputsize_sub[3]/(1024*1024*1024*trans_rate)
                            out_tensor.append(output_sub[:,:,1:-1,:])
                        elif j == 0:
                            out_tensor.append(output_sub[:,:,:-1,:])
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-1)*outputsize_sub[3]/(1024*1024*1024*trans_rate)
                        else:
                            out_tensor.append(output_sub[:,:,1:,:])
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-1)*outputsize_sub[3]/(1024*1024*1024*trans_rate)
                    elif kerner_size[i] == 2:
                        t_sub_send = 32*outputsize_sub[1]*outputsize_sub[2]*outputsize_sub[3]/(1024*1024*1024*trans_rate)
                        out_tensor.append(output_sub)
                    t_sub.append(t_sub_rec + t_sub_cmp + t_sub_send)
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
        print("this is t", t)
    return output_tensor, t

##################### workig here #######################
def opt_DiSNet(in_img, layer_range, input_index, trans_rate, comp_rate, model):
    # layers = 10
    # layers = 21 
    
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
                    # print(input_index[i][1][j])
                    if input_index[i][1][j] == 0:
                        break
                    # print(input_index[i][0][j][0],input_index[i][0][j][1]+1)
                    in_sub = in_tensor[:,:,input_index[i][0][j][0]:input_index[i][0][j][1]+1,:]
                    inputsize_sub = in_sub.size()
                    
                    t_sub_rec = 32*inputsize_sub[1]*inputsize_sub[2]*inputsize_sub[3]/(1024*1024*1024*trans_rate)
                    
                    # say this is running on diffent devices at different speeds
                    output_sub, t_sub_cmp = infer_layer(in_sub, model, p)
                    # print(output_sub.shape)
                    outputsize_sub = output_sub.size()
                    # print(outputsize_sub)

                    #this ------- shit
                    if kerner_size[p] == 3:
                        # problem ?
                        if j not in [0,8]:
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-2)*outputsize_sub[3]/(1024*1024*1024*trans_rate)
                            out_tensor.append(output_sub[:,:,1:-1,:])
                        elif j == 0:
                            out_tensor.append(output_sub[:,:,:-1,:])
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-1)*outputsize_sub[3]/(1024*1024*1024*trans_rate)
                        else:
                            out_tensor.append(output_sub[:,:,1:,:])
                            t_sub_send = 32*outputsize_sub[1]*(outputsize_sub[2]-1)*outputsize_sub[3]/(1024*1024*1024*trans_rate)
                    elif kerner_size[p] == 2:
                        t_sub_send = 32*outputsize_sub[1]*outputsize_sub[2]*outputsize_sub[3]/(1024*1024*1024*trans_rate)
                        out_tensor.append(output_sub)
                    
                    #this is per layer 
                    t_sub_cmp_proportional = (1 - (comp_rate[j]/sum(comp_rate)))*t_sub_cmp*DEVICE_PACE_RATE#times slowness comparison
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
    return output_tensor, t


