def create_mask_dic():
    mask_sub_dic = {"image": [],
                    "idx": [],
                    "R": [],
                    "T": [],
                    "f": [],
                    "c": [],
                    "trans_crop": []
                    }
    return mask_sub_dic


def create_dictionary_data():
    dic = {"image": [],
           "joints_im": [],
           "R": [],
           "masks": {1: create_mask_dic(),
                     2: create_mask_dic(),
                     3: create_mask_dic(),
                     4: create_mask_dic()}
           }
    return dic


def collate_SMPL(batch):
    dic = create_dictionary_data()
    for i in batch:
        for key in i.keys():
            if key == "masks":
                for n in i[key].keys():
                    for sub_key in i[key][n].keys():
                        dic[key][n][sub_key].append(i[key][n][sub_key])
            else:
                dic[key].append(i[key])
    #now stack together


