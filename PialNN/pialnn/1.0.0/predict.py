import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config import load_config
from data.dataload import load_data, BrainDataset
from model.pialnn import PialNN
from utils import compute_normal, save_mesh_obj, compute_distance


if __name__ == '__main__':
    
    """set device"""
    if torch.cuda.is_available():
        device_name = "cuda:0"
        print('selected gpu')
    else:
        device_name = "cpu"
    device = torch.device(device_name)


    """load configuration"""
    config = load_config()

    """load dataset"""
    print("----------------------------")
    print("Start loading dataset ...")
    test_data = load_data(data_path = config.data_path,
                          hemisphere = config.hemisphere)
    n_data = len(test_data)
    L,W,H = test_data[0].volume[0].shape  # shape of MRI
    LWHmax = max([L,W,H])

    test_set = BrainDataset(test_data)
    testloader = DataLoader(test_set, batch_size=1, shuffle=True)
    print("Finish loading dataset. There are total {} subjects.".format(n_data))
    print("----------------------------")

    
    """load model"""
    print("Start loading model ...")
    model = PialNN(config.nc, config.K, config.n_scale).to(device)
    model.load_state_dict(torch.load(config.model,
                                     map_location=device))
    model.initialize(L, W, H, device)
    print("Finish loading model")
    print("----------------------------")
    
    
    """evaluation"""
    print("Start evaluation ...")
    with torch.no_grad():
        #CD = []
        #AD = []
        #HD = []
        for idx, data in tqdm(enumerate(testloader)):
            volume_in, v_gt, f_gt, v_in, f_in = data

            sub_id = idx
	
            volume_in = volume_in.to(device)
            v_gt = v_gt.to(device)
            f_gt = f_gt.to(device)
            v_in = v_in.to(device)
            f_in = f_in.to(device)

            # set n_smooth > 1 if the mesh quality is not good
            v_pred = model(v=v_in, f=f_in, volume=volume_in,
                           n_smooth=config.n_smooth, lambd=config.lambd)

            v_pred_eval = v_pred[0].cpu().numpy() * LWHmax/2 + [L/2,W/2,H/2]
            f_pred_eval = f_in[0].cpu().numpy()
            v_gt_eval = v_gt[0].cpu().numpy() * LWHmax/2 + [L/2,W/2,H/2]
            f_gt_eval = f_gt[0].cpu().numpy()

            # compute distance-based metrics
            #cd, assd, hd = compute_distance(v_pred_eval, v_gt_eval,
            #                                f_pred_eval, f_gt_eval, config.n_test_pts)
            
            #CD.append(cd)
            #AD.append(assd)
            #HD.append(hd)
            print('sub_id',sub_id)
            if config.save_mesh_eval:
                path_save_mesh = "./pialnn_mesh_eval_"\
                        +config.hemisphere+"_subject_"+str(sub_id)+".obj"

                normal = compute_normal(v_pred, f_in)
                n_pred_eval = normal[0].cpu().numpy()
                save_mesh_obj(v_pred_eval, f_pred_eval, n_pred_eval, path_save_mesh)
                
                ################
                path_save_mesh = "./pialnn_mesh_eval_"\
                        +config.hemisphere+"_subject_"+str(sub_id)+"_gt.obj"

                normal = compute_normal(v_gt, f_gt)
                n_gt_eval = normal[0].cpu().numpy()
                save_mesh_obj(v_gt_eval, f_gt_eval, n_gt_eval, path_save_mesh)

    # print("CD: Mean={}, Std={}".format(np.mean(CD), np.std(CD)))
    # print("AD: Mean={}, Std={}".format(np.mean(AD), np.std(AD)))
    # print("HD: Mean={}, Std={}".format(np.mean(HD), np.std(HD)))
    print("Finish evaluation.")
    print("----------------------------")
