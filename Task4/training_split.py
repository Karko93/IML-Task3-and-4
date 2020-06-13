
import random
import copy


def split_training(_array, _val_array_size):

    array = copy.deepcopy(_array)

    random.shuffle(array)

    _train_set = []
    _val_set = []

    elem_buffer = []

    elem_buffer.append(array.pop())

    n = 0

    while True:
        n += 1
        print(f'iteration: {n:6d} len(array) = {len(array):6d} len(val) = {len(_val_set):6d} len(buf) = {len(elem_buffer):4d}')

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






if __name__ == '__main__':

    random.seed(12341)

    with open('train_triplets.txt', 'r') as f:
        _data = f.readlines()
    data = []
    for elem in _data:
        data.append(elem.rstrip().split(' '))
    
    print(data[:5])
    train_set, val_set, unused = split_training(data, len(data) //10)

    print(f'trainset length = {len(train_set)}; validation length = {len(val_set)}; unused = {len(unused)}')