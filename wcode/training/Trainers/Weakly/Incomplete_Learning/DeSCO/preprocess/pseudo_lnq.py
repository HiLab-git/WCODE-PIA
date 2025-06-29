import numpy as np
import shutil
from pathlib import Path
import ants
import SimpleITK as sitk
import json
import multiprocessing
from tqdm import tqdm
from time import sleep
from scipy import ndimage
from wcode.utils.NDarray_operations import get_largest_k_components
from scipy.ndimage import binary_opening, binary_closing,generate_binary_structure

def label_smooth(label_arr):
    label_arr = label_arr == 1
    kernel = generate_binary_structure(2,1)
    mask = binary_opening(label_arr, kernel)
    mask = binary_closing(mask, kernel)
    return mask
def write_json(sorted_dict, out_path):
    with open(str(out_path), 'w+') as fp:
        json.dump(sorted_dict,fp,indent=4)
        
def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float32)
        label_tmp = label_tmp.astype(np.float32)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice

def synra_labeling(image3d, slice_label, slice_loc, out_dir):
    '''
    image3d: D x H x W
    
    '''
    moving_image = ants.from_numpy(image3d[slice_loc])
    moving_label = ants.from_numpy(slice_label)
    pseudo_label = []
    for i in range(slice_loc - 1, -1, -1):
        fixed_image = ants.from_numpy(image3d[i])
        outs = ants.registration(fixed_image, moving_image, type_of_transforme="SyNRA",outprefix=str(out_dir)+"/")
        fixed_label = ants.apply_transforms(fixed_image, moving_label, transformlist=outs['fwdtransforms'],
                                              interpolator='nearestNeighbor')
        pseudo_label.append(label_smooth(fixed_label.numpy()))
        
        moving_image = fixed_image
        moving_label = fixed_label
    
    pseudo_label.reverse()
    moving_image = ants.from_numpy(image3d[slice_loc])
    moving_label = ants.from_numpy(slice_label)
    
    pseudo_label.append(slice_label)
    
    for i in range(slice_loc + 1, image3d.shape[0]):
        fixed_image = ants.from_numpy(image3d[i])
        outs = ants.registration(fixed_image, moving_image, type_of_transforme="SyNRA")
        fixed_label = ants.apply_transforms(fixed_image, moving_label, transformlist=outs['fwdtransforms'],
                                              interpolator='nearestNeighbor')
        pseudo_label.append(label_smooth(fixed_label.numpy()))
        
        moving_image = fixed_image
        moving_label = fixed_label
        
    
    return np.stack(pseudo_label)

def transverse_labeling(image_arr, label_arr, out_log):
    z,*_ = np.where(label_arr)
    slice_loc = (z.max() + z.min()) // 2
    pseudo_label = synra_labeling(image_arr, label_arr[slice_loc], slice_loc, out_log)    
    return pseudo_label,slice_loc


def preprocess_one_volume(image_path,label_path,class_num, out_dir ,tmp_dir):
    out_log = tmp_dir / label_path.name
    out_log.mkdir(exist_ok=True,parents=True)
    save_dir = out_dir / image_path.name[:-10]
    save_dir.mkdir(exist_ok = True)
    save_json_path = save_dir/"desc.json"
    if save_json_path.exists():
        return
    
    image_obj = sitk.ReadImage(image_path)    
    image = sitk.GetArrayFromImage(image_obj)
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
    
    componets = get_largest_k_components(label,k=100)
    
    desc = {"components_num": len(componets), "componet_detail":[]}
    fused_label = np.zeros_like(label)
    for i, comp in enumerate(componets):
        comp_dir = trans_label_path = save_dir/f"component_{i}"
        comp_dir.mkdir(exist_ok = True)
        comp_desc = {"component id":i,"frontal plane":{}, "transverse plane":{}}
        # transverse plane
        pseudo_label,slice_loc = transverse_labeling(image, comp,out_log)
        pseudo_label = pseudo_label.astype(np.uint8)
        pseudo_label_obj = sitk.GetImageFromArray(pseudo_label)
        pseudo_label_obj.CopyInformation(image_obj)
        
        trans_label_path = comp_dir/(image_path.name[:-10] + f"_trans.nrrd")
        sitk.WriteImage(pseudo_label_obj, trans_label_path)
        
        transverse_dice = cal_dice(pseudo_label, comp,num=class_num)
        comp_desc['transverse plane']['component dice'] = transverse_dice.tolist()
        comp_desc['transverse plane']['labeled slice'] = int(slice_loc)
        
        fused_label = fused_label | pseudo_label
        
        # frontal plane
        transposed_image = np.transpose(image,(1,0,2))
        transposed_label = np.transpose(comp,(1,0,2))
        pseudo_label,slice_loc = transverse_labeling(transposed_image, transposed_label,out_log)
        pseudo_label = pseudo_label.astype(np.uint8)
        pseudo_label = np.transpose(pseudo_label, (1,0,2))
        pseudo_label_obj = sitk.GetImageFromArray(pseudo_label)
        pseudo_label_obj.CopyInformation(image_obj)
        front_label_path = comp_dir/(image_path.name[:-10] + "_front.nrrd")
        sitk.WriteImage(pseudo_label_obj, front_label_path)
        
        frontal_dice = cal_dice(pseudo_label, comp,num=class_num)
    
        comp_desc['frontal plane']['component dice'] = frontal_dice.tolist()
        comp_desc['frontal plane']['labeled slice'] = int(slice_loc)
        
        desc['componet_detail'].append(comp_desc)
        fused_label = fused_label | pseudo_label
    
    fused_dice = cal_dice(fused_label,label,num=class_num)
    desc["dice"] = fused_dice.tolist()
    pseudo_label_obj = sitk.GetImageFromArray(fused_label)
    pseudo_label_obj.CopyInformation(image_obj)
    front_label_path = save_dir/(image_path.name[:-10] + "_fused.nrrd")
    sitk.WriteImage(pseudo_label_obj, front_label_path)
    
    
    
    if not save_json_path.exists():
        write_json(desc, save_json_path)
    shutil.rmtree(str(out_log))

def mp_launch(func, parameter_list , num_p, context="spawn"):
    with multiprocessing.get_context(context).Pool(num_p) as pool:
        r = []
        for p in parameter_list:
            r.append(pool.starmap_async(func, (p, )))
        workers = [j for j in pool._pool]
        
        remaining = list(range(len(parameter_list)))
        with tqdm(
            desc=None, total=len(r), disable=False
        ) as pbar:
            while len(remaining) > 0:
                all_alive = all([j.is_alive() for j in workers])
                if not all_alive:
                    raise RuntimeError(
                        "Some background worker is 6 feet under. Yuck."
                    )
                done = [i for i in remaining if r[i].ready()]
                for _ in done:
                    pbar.update()
                remaining = [i for i in remaining if i not in done]
                sleep(0.1)

    return [i.get() for i in r]


if __name__ == "__main__":
    tmp_dir = Path("/media/ssd/wlty/LNQ/log")
    image_dir, label_dir = Path("/media/ssd/wlty/LNQ/Dataset/CTLymphNodes02/images"), Path("/media/ssd/wlty/LNQ/Dataset/CTLymphNodes02/labels")
    out_dir = Path("/media/ssd/wlty/LNQ/Dataset/CTLymphNodes02_pseudo_label/labelsTr")
    out_dir.mkdir(exist_ok=True)
    class_num = 2
    num_p = 4
    images = list(image_dir.iterdir())
    param_list = []
    for image_path in images:
        label_path = label_path = label_dir / (image_path.name[:-12] + ".nii.gz")
        param_list.append((image_path,label_path,class_num, out_dir ,tmp_dir))
    mp_launch(func=preprocess_one_volume,parameter_list=param_list,num_p=num_p)