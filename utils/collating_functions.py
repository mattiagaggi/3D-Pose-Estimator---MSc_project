import numpy as np

from utils.trans_numpy_torch import numpy_to_tensor_float_cpu, numpy_to_long_cpu


def collate_h36m(batch):
    dic_in, dic_out = {}, {}
    segments = []
    for idx, i in enumerate(batch):
        segments += [ idx*2 + 1, idx*2]
        for key in i[0].keys():
            if key not in dic_in.keys():
                dic_in[key] = []
            dic_in[key] += i[0][key]
        for key in i[1].keys():
            if key not in dic_out.keys():
                dic_out[key] = []
            dic_out[key] += i[1][key]
    for key in dic_in.keys():
        dic_in[key] = np.stack(dic_in[key], axis=0)
        dic_in[key] = numpy_to_tensor_float_cpu(dic_in[key])
    for key in dic_out.keys():
        dic_out[key] = np.stack(dic_out[key], axis=0)
        dic_out[key] = numpy_to_tensor_float_cpu(dic_out[key])
    dic_in['invert_segments'] = numpy_to_long_cpu(segments)
    return dic_in, dic_out





def collate_smpl(batch):
    dic = {}
    dic['mask_idx_all'] = []
    for idx,i in enumerate(batch):
        for key in i.keys():
            if key not in dic.keys():
                dic[key] = []
            if "mask" in key:
                dic[key] += i[key]
                count_masks = len(i[key])
            else:
                dic[key].append(i[key])
        dic['mask_idx_all'] += [idx] * count_masks
    for key in dic.keys():
        if 'idx' in key:
            dic[key] = numpy_to_long_cpu(dic[key])
        if 'idx' not in key:
            dic[key] = np.stack(dic[key], axis=0)
            dic[key] = numpy_to_tensor_float_cpu(dic[key])
    return dic





