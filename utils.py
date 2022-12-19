import math
import time

def trans_time_forward(tensor, trans_rate, layer_range):
    
    out_tensor = tensor.size()
    # print(out_tensor)
    if layer_range[1] == 18:
        t_sub_send = 0
    else:  
        t_sub_send = 32*out_tensor[1]*(out_tensor[2])*out_tensor[3]/(1024*1024*1024*trans_rate)

    return t_sub_send
