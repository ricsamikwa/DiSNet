import math

vgg16 = {
    'conv1_1': [3, 1, 1], #k, s, p
    'conv1_2': [3, 1, 1],
    'pool1': [2, 2, 0],
    'conv2_1': [3, 1, 1],
    'conv2_2': [3, 1, 1],
    'pool2': [2, 2, 0],
    'conv3_1': [3, 1, 1],
    'conv3_2': [3, 1, 1],
    'conv3_3': [3, 1, 1],
    'pool3': [2, 2, 0],
    'conv4_1': [3, 1, 1],
    'conv4_2': [3, 1, 1],
    'conv4_3': [3, 1, 1],
    'pool4': [2, 2, 0],
    'conv5_1': [3, 1, 1],
    'conv5_2': [3, 1, 1],
    'conv5_3': [3, 1, 1],
    'pool5': [2, 2, 0]
}


class FieldCalculator:
    def __init__(self, input_image_size, input_num, output_num, partition_num): #img, start layer, end layer, number of partitions
        super(FieldCalculator, self).__init__()
        self.architecture = vgg16
        self.input_image_size = input_image_size
        self.input_num = input_num
        self.output_num = output_num
        self.partition_num = partition_num
        self.out = ()
        self.i = ()
        layer = -1 #initialising layer to no feasible layer
        input_layer = ('input_layer', self.input_image_size) #input layer is the input size dimension
        for key in self.architecture: #looping through the model layers
            # print(key)
            layer = layer + 1 # start from layer 0 then upwards
            if layer < self.input_num: # if layer is less start layer for the partitioning
                current_layer = self._calculate_layer_output(self.architecture[key], input_layer, key)
                input_layer = current_layer
            elif layer <= self.output_num:
                if layer == self.input_num:
                    input_layer = (input_layer[0], input_layer[1], 1, 1, 0.5)
                    self.i = input_layer
                    
                current_layer = self._calculate_layer_info(self.architecture[key], input_layer, key)
                # print(current_layer)
                input_layer = current_layer
                self.out = current_layer
            else:
                break
        self.input = self.calculate_overlapped_data()


    def _calculate_layer_output(self, current_layer, input_layer, layer_name):
        
        n_in = input_layer[1]
        k = current_layer[0]
        s = current_layer[1]
        p = current_layer[2]
        n_out = math.floor((n_in - k + 2 * p) / s) + 1
        # print('layer output', layer_name, n_out)
        return layer_name, n_out

    def _calculate_layer_info(self, current_layer, input_layer, layer_name):
        n_in = input_layer[1]
        j_in = input_layer[2]
        r_in = input_layer[3]
        start_in = input_layer[4]

        k = current_layer[0]
        s = current_layer[1]
        p = current_layer[2]

        n_out = math.floor((n_in - k + 2 * p) / s) + 1
        padding = (n_out - 1) * s - n_in + k
        p_right = math.ceil(padding / 2)
        p_left = math.floor(padding / 2)

        j_out = j_in * s
        r_out = r_in + (k - 1) * j_in
        start_out = start_in + ((k - 1) / 2 - p_left) * j_in
        # print(layer_name, n_out, j_out, r_out, start_out)
        return layer_name, n_out, j_out, r_out, start_out

    def calculate_overlapped_data(self):

        #horizontal split ratio
        split_ratio = [2,3,5]

        originl_size = self.i[1]
        out_size = self.out[1]
        jump = self.out[2]
        r = self.out[3]
        start = self.out[4]
        p_num = int(out_size/self.partition_num) #split equal sizes
        p_renum = int(out_size%self.partition_num) #the remainder after splitting equal sizes

        partition = []
        p = []
        start_end = []
        #this loop needs an update
        for i in range(0, self.partition_num): #looping for each of the partitions
            p.append([0, 0]) # add [0,0]  to partition divisions 
            start_end.append([0, 0]) 
            if i < p_renum -1 or i == p_renum -1:
                partition.append(p_num+1)
            else:
                partition.append(p_num)
        # print("partition", partition)
        for i in range(len(partition)):
            if r%2 == 0:
                if i == 0:
                    start_end[i] = [int(start),int((partition[i]-1)*jump + start)]
                else:
                    start_end[i] = [int(start_end[i-1][1]+jump), int(start_end[i-1][1]+partition[i]*jump)]
                p[i] = [max(start_end[i][0]-int(r/2), 0), min(start_end[i][1]+int(r/2)-1, originl_size-1)]
            else:            
                if i == 0:
                    start_end[i] = [int(start-0.5),int((partition[i]-1)*jump + start-0.5)]
                else:
                    start_end[i] = [int(start_end[i-1][1]+jump), int(start_end[i-1][1]+partition[i]*jump)]
                p[i] = [max(start_end[i][0]-int(r/2), 0), min(start_end[i][1]+int(r/2), originl_size-1)]
            if partition[i]==0:
                p[i]= [0,0]
        print(p,partition)
        return p,partition
    
# ReceptiveFieldCalculator()