
def create_mask_dic():
    mask_sub_di c ={"image" :[],
                  "idx" :[],
                  "R" :[],
                  "T" :[],
                  "f" :[],
                  "c" :[],
                  "trans_crop" :[]
                  }
    return mask_sub_dic


def create_dictionary_data(self):
    di c= {"image": [],
          "joints_im": [],
          "R": [],
          "masks": {1 : create_mask_dic(),
                    2 : create_mask_dic(),
                    3 : create_mask_dic(),
                    4 : self.create_mask_dic()}
          }
    return dic
def change_data_keys(batch):
    new_dic={}
    for key in batch.keys():
        if "masks" in key:
            split = key.split"_"





