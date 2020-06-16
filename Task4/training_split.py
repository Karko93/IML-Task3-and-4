
import random
import copy


def split_training():

    with open('train_triplets.txt', 'r') as f:
        _data = f.readlines()
    data = []
    for elem in _data:
        data.append(elem.rstrip().split(' '))
  
    
    _val_array_size = len(data) //10
    
    array = copy.deepcopy(data)

    random.shuffle(array)

    _train_set = []
    _val_set = []

    elem_buffer = []

    elem_buffer.append(array.pop())

    n = 0

    while True:
        n += 1
        #print(f'iteration: {n:6d} len(array) = {len(array):6d} len(val) = {len(_val_set):6d} len(buf) = {len(elem_buffer):4d}')

        if len(_val_set) >= _val_array_size:
            break
        
        if len(elem_buffer) == 0:
            elem_buffer.append(array.pop())



        elem = elem_buffer.pop()
        _val_set.append(elem)

        for x in range(len(array) -1, -1, -1):
            if all(i in array[x] for i in elem):
                elem_buffer.append(array.pop(x))

        


    _unused = elem_buffer

    _train_set = array

    return _train_set, _val_set, _unused



