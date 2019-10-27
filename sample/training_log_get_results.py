import numpy
import matplotlib.pyplot as plt
import matplotlib
import os
import matplotlib.ticker as tick
#matplotlib.use('Agg')




from logger.train_hist_logger import TrainingLogger
import pickle as pkl
import numpy as np
from utils.utils_H36M.visualise import Drawer



def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)

    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]

    return new_tick_format

log=TrainingLogger("data/checkpoints/enc_dec_S15678_no_rot_baseSMPL/log_results")

def moving_avr(arr,move=10):
    new=[]
    for i, value in enumerate(arr):
        if i>move:
            if i < len(arr)-move:
                new.append(np.mean(arr[i-move:i+move]))
            else:
                new.append(np.mean(arr[i-move:i]))
        else:
            new.append(value)
    return new




def list_scalars_to_dic(lst):

    dic={}
    for i in lst:
        for key in i.keys():
            if key in dic:
                dic[key]+=i[key]
            else:
                dic[key]=i[key]
    return dic

def get_scalar_data(log):
    path=log.path
    count=1
    scalar_lst=[]
    print(path)
    while os.path.exists(os.path.join(path,'scalars%s.pkl'  % count)): #%s
        scal=pkl.load(open(os.path.join(path,'scalars%s.pkl' % count), "rb"))  #% count), "rb"))
        scalar_lst.append(scal[1]) #[1
        count += 1
    return list_scalars_to_dic(scalar_lst)

scalar = get_scalar_data(log)
print(scalar.keys())
print(len(scalar['train_loss_idx']),len(scalar['train_loss']))
print(len(scalar['test_loss_idx']),len(scalar['test_loss']))



def plot_loss(idx_train,idx_test,train,test):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_xlim([2000, 6000])
    #ax.set_ylim([0, 60])
    ax.plot(idx_train, train, label='verts train loss ')
    ax.plot(idx_test, test, label='verts test loss')
    ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax.legend(fontsize=16)
    ax.tick_params(labelsize=14)
    return fig
#print(scalar['train_loss_idx'])
fig=plot_loss(scalar['train_loss_vert_idx'], scalar['test_loss_vert_idx'], scalar['train_loss_vert'], scalar['test_loss_vert'])
#fig=plot_loss(scalar['test_metrics/MPJ_idx'], scalar['test_loss_idx'],scalar['test_metrics/MPJ'], scalar['test_loss'])
plt.show()

"""
set="train"

inp=log.get_dic_arbitrary(set,"indic_act_5n0")
out = log.get_dic_arbitrary(set,"outdic_act_5n0")
# data/checkpoints/enc_dec_S15678_no_rotfinal3D/log_results_test/training_logger/test/indic_act_15n0
print(inp.keys())
print(out.keys())



for i in range(inp["im_in"].shape[0]//2):
    #print(inp["details"][i])
    plt.axis("off")
    plt.figure()
    im=np.transpose(inp['im_in'][i],(1,2,0))
    im2 = im.copy()
    im2[:, :, 0] = im[:, :, 2]
    im2[:, :, 2] = im[:, :, 0]
    plt.imshow(im2)
    plt.axis("off")
    plt.show()
    plt.figure()
    title="degrees 0 -"+set
    plt.title("degrees 0 -")
    im=np.transpose(out['image_final0'][i],(1,2,0))
    im2 = im.copy()
    im2[:, :, 0] = im[:, :, 2]
    im2[:, :, 2] = im[:, :, 0]
    plt.imshow(im2)
    plt.axis("off")
    plt.figure()
    title = "degrees 30 -" + set
    plt.title(title)
    im=np.transpose(out['image_final30'][i],(1,2,0))
    im2 = im.copy()
    im2[:, :, 0] = im[:, :, 2]
    im2[:, :, 2] = im[:, :, 0]
    plt.imshow(im2)
    plt.axis("off")
    plt.figure()
    title = "degrees 45 -" + set
    plt.title(title)
    im=np.transpose(out['image_final45'][i],(1,2,0))
    im2 = im.copy()
    im2[:, :, 0] = im[:, :, 2]
    im2[:, :, 2] = im[:, :, 0]
    plt.imshow(im2)
    plt.axis("off")
    plt.figure()
    title = "degrees 60 -" + set
    plt.title(title)
    im=np.transpose(out['image_final60'][i],(1,2,0))
    im2 = im.copy()
    im2[:, :, 0] = im[:, :, 2]
    im2[:, :, 2] = im[:, :, 0]
    plt.imshow(im2)
    plt.axis("off")
    plt.figure()
    title = "degrees 90 -" + set
    plt.title(title)
    im = np.transpose(out['image_final90'][i], (1, 2, 0))
    im2 = im.copy()
    im2[:, :, 0] = im[:, :, 2]
    im2[:, :, 2] = im[:, :, 0]
    plt.imshow(im2)
    plt.axis("off")
    plt.figure()
    title = "degrees 180 -" + set
    plt.title(title)
    im = np.transpose(out['image_final180'][i], (1, 2, 0))
    im2 = im.copy()
    im2[:, :, 0] = im[:, :, 2]
    im2[:, :, 2] = im[:, :, 0]
    plt.imshow(im2)
    plt.axis("off")
    plt.show()
    #plt.figure()
    
    im=np.transpose(out['image_final45'][i],(1,2,0))
    im2 = im.copy()
    im2[:, :, 0] = im[:, :, 2]
    im2[:, :, 2] = im[:, :, 0]
    plt.imshow(im2)
    




d=Drawer()
for i in range(inp["im_in"].shape[0]):
    #print(inp["details"][i])
    plt.figure()
    im=np.transpose(inp["im_in"][i],(1,2,0))
    im2 = im.copy()
    im2[:, :, 0] = im[:, :, 2]
    im2[:, :, 2] = im[:, :, 0]
    plt.imshow(im2)
    fig2 = plt.figure()
    fig2 = d.pose_3d(out["pose_final"][i], plot=True, fig=fig2, azim=-90, elev=-80)
    #plt.title("Prediction")
    fig3 = plt.figure()
    out["joints_im"][i]= out["joints_im"][i]-np.reshape(out["joints_im"][i][0],(1,3))
    cam = np.dot(out["joints_im"][i],np.transpose(inp['R_world_im'][i],(1,0)))
    fig3 = d.pose_3d(cam, plot=True, fig=fig3, azim=-90, elev=-80)
    fig4 = plt.figure()
    fig4 = d.pose_3d(out["pose_final"][i], plot=True, fig=fig4, azim=-90, elev=10)
    #plt.title("Prediction")
    fig5 = plt.figure()
    out["joints_im"][i]= out["joints_im"][i]-np.reshape(out["joints_im"][i][0],(1,3))
    cam = np.dot(out["joints_im"][i],np.transpose(inp['R_world_im'][i],(1,0)))
    fig5 = d.pose_3d(cam, plot=True, fig=fig5, azim=-90, elev=10)


    fig6 =plt.figure()
    out["pose_final"][i]= out["pose_final"][i]-np.reshape(out["pose_final"][i][0],(1,3))
    fig6 = d.poses_3d(cam,out["pose_final"][i],plot=True, fig=fig6, azim=-90, elev=-80)
    #plt.title("real")
    fig7 =plt.figure()
    out["pose_final"][i]= out["pose_final"][i]-np.reshape(out["pose_final"][i][0],(1,3))
    fig7 = d.poses_3d(cam,out["pose_final"][i],plot=True, fig=fig7, azim=-90, elev=10)
    #if i==3:
    #    break
    #plt.title("real")

    plt.show()

#scal = get_scalar_data(log)



#plt.plot(scal['train_loss_idx'], scal['train_loss'])
#plt.show()

#dic=log.get_dic("train", 0, extra_str="")
#print(dic.keys())


#dic=log.load_batch_images("train_img_images",0)

"""