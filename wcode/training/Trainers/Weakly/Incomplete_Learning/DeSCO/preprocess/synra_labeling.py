import ants
import numpy as np
import shutil
from pathlib import Path
import SimpleITK as sitk
import json
import multiprocessing
from tqdm import tqdm
from time import sleep
# def empty_tmpdir():
#     shutil.rmtree(tmp_dir)
#     tmp_dir.mkdir()

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
        pseudo_label.append(fixed_label.numpy())
        
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
        pseudo_label.append(fixed_label.numpy())
        
        moving_image = fixed_image
        moving_label = fixed_label
        
    
    return np.stack(pseudo_label)

def preprocess_one_volume(image_path,label_dir,class_num, out_dir ,tmp_dir):
    label_name = image_path.name[:-10] + ".nrrd"
    out_log = tmp_dir / label_name
    out_log.mkdir(exist_ok=True)
    save_dir = out_dir / image_path.name[:-10]
    save_dir.mkdir(exist_ok = True)
    desc = {"frontal plane":{}, "transverse plane":{}}
    
    
    trans_label_path = save_dir/(image_path.name[:-10] + f"_trans.nrrd")
    if not trans_label_path.exists():
        label_path = label_dir / label_name
        image_obj = sitk.ReadImage(image_path)    
        image = sitk.GetArrayFromImage(image_obj)
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        # transverse plane
        z,*_ = np.where(label)
        slice_loc = (z.max() + z.min()) // 2
        pseudo_label = synra_labeling(image, label[slice_loc], slice_loc,out_log)    
        pseudo_label = pseudo_label.astype(np.uint8)
        pseudo_label_obj = sitk.GetImageFromArray(pseudo_label)
        pseudo_label_obj.CopyInformation(image_obj)
        sitk.WriteImage(pseudo_label_obj, trans_label_path)
        transverse_dice = cal_dice(pseudo_label, label,num=class_num)
        desc['transverse plane']['dice'] = transverse_dice.tolist()
        desc['transverse plane']['labeled slice']=int(slice_loc)

    front_label_path = save_dir/(image_path.name[:-10] + "_front.nrrd")
    if not (save_dir/(image_path.name[:-10] + "_front.nrrd")).exists():
        label_path = label_dir / label_name
        image_obj = sitk.ReadImage(image_path)    
        image = sitk.GetArrayFromImage(image_obj)
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        # frontal plane
        transposed_image= np.transpose(image,(1,0,2))
        transposed_label= np.transpose(label,(1,0,2))
        z,*_ = np.where(transposed_label)
        slice_loc = (z.max() + z.min()) // 2
        pseudo_label = synra_labeling(transposed_image, transposed_label[slice_loc], slice_loc,out_log)    
        pseudo_label = pseudo_label.astype(np.uint8)
        pseudo_label = np.transpose(pseudo_label, (1,0,2))
        pseudo_label_obj = sitk.GetImageFromArray(pseudo_label)
        pseudo_label_obj.CopyInformation(image_obj)
        sitk.WriteImage(pseudo_label_obj, front_label_path)
        
        frontal_dice = cal_dice(pseudo_label, label,num=class_num)
        
        desc['frontal plane']['dice'] = frontal_dice.tolist()
        desc['frontal plane']['labeled slice'] = int(slice_loc)
    save_json_path = save_dir/"desc.json"
    if not save_json_path.exists():
        write_json(desc, save_json_path)
    shutil.rmtree(str(out_log))

    
def cmp(x):
    if len(x[1]) == 0:
        return -1
    elif len(x[1]['transverse plane']) == 0 and x[1]['frontal plane']!= 0:
        return x[1]['frontal plane']
    elif len(x[1]['transverse plane']) != 0 and x[1]['frontal plane']== 0:
        return x[1]['transverse plane']
    else:
        return x[1]['transverse plane'] + x[1]['frontal plane']
        
if __name__ == "__main__":

    tmp_dir = Path("/media/x/Wlty/LNQ/Dataset/log/")
    
    # image_dir, label_dir = Path("/media/x/Wlty/LNQ/test_set/image"), Path("/media/x/Wlty/LNQ/test_set/label")
    # front_our_dir, transvers_out_dir = Path("/media/x/Wlty/LNQ/test_set/out/frontalplane/labelsTr"),Path("/media/x/Wlty/LNQ/test_set/out/transverseplane/labelsTr")
    # image_dir, label_dir = Path("/media/x/Wlty/LNQ/Dataset/LNQ2023/imagesTr"), Path("/media/x/Wlty/LNQ/Dataset/LNQ2023/labelsTr")
    # out_dir = Path("/media/x/Wlty/LNQ/Dataset/LNQ2023_pseudo_label/labelsTr")
    
    image_dir, label_dir = Path("/media/x/Wlty/LNQ/test_set/image"), Path("/media/x/Wlty/LNQ/test_set/label")
    out_dir = Path("/media/x/Wlty/LNQ/test_set/out")
    
    if not out_dir.exists():
        out_dir.mkdir()
    class_num = 2
    num_p = 4
    images = list(image_dir.iterdir())
    
    preprocess_one_volume(images[0], label_dir,2, out_dir, tmp_dir)
    # with multiprocessing.get_context("spawn").Pool(num_p) as pool:
    #     r = []
        
    #     for i in images:
    #         r.append(pool.starmap_async(preprocess_one_volume, ((i, label_dir,2,out_dir,tmp_dir), )))
    #     workers = [j for j in pool._pool]
    #     remaining = list(range(len(images)))
    #     with tqdm(
    #         desc=None, total=len(r), disable=False
    #     ) as pbar:
    #         while len(remaining) > 0:
    #             all_alive = all([j.is_alive() for j in workers])
    #             if not all_alive:
    #                 raise RuntimeError(
    #                     "Some background worker is 6 feet under. Yuck."
    #                 )
    #             done = [i for i in remaining if r[i].ready()]
    #             for _ in done:
    #                 pbar.update()
    #             remaining = [i for i in remaining if i not in done]
    #             sleep(0.1)



        
    
    # for image_path in image_dir.iterdir():
    #     print(f"preprocessing {str(image_path)}...")
    #     label_name = image_path.name[:-10] + ".nrrd"
    #     if not (transvers_out_dir/label_name).exists():
    #         label_path = label_dir / label_name
    #         image_obj = sitk.ReadImage(image_path)    
    #         image = sitk.GetArrayFromImage(image_obj)
    #         label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
    #         # transverse plane
    #         mid = image.shape[0] // 2
    #         pseudo_label = synra_labeling(image, label[mid], mid)    
    #         pseudo_label = pseudo_label.astype(np.uint8)
    #         pseudo_label_obj = sitk.GetImageFromArray(pseudo_label)
    #         pseudo_label_obj.CopyInformation(image_obj)
    #         sitk.WriteImage(pseudo_label_obj, transvers_out_dir/label_name)
    #         trans_dice = cal_dice(pseudo_label, label,num=class_num)
    #         pseudo_dice[label_name] = {
    #             "transverse plane":trans_dice.tolist()
    #         }
        
    #     if not (front_our_dir/label_name).exists():
    #         label_path = label_dir / label_name
    #         image_obj = sitk.ReadImage(image_path)    
    #         image = sitk.GetArrayFromImage(image_obj)
    #         label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
    #         # frontal plane
    #         transposed_image= np.transpose(image,(1,0,2))
    #         transposed_label= np.transpose(label,(1,0,2))
    #         mid = image.shape[0] // 2
    #         pseudo_label = synra_labeling(transposed_image, transposed_label[mid], mid)    
    #         pseudo_label = pseudo_label.astype(np.uint8)
    #         pseudo_label = np.transpose(pseudo_label, (1,0,2))
    #         pseudo_label_obj = sitk.GetImageFromArray(pseudo_label)
    #         pseudo_label_obj.CopyInformation(image_obj)
    #         sitk.WriteImage(pseudo_label_obj, front_our_dir/label_name)
    #         frontal_dice = cal_dice(pseudo_label, label,num=class_num)
    #         if 'transverse plane' in pseudo_dice[label_name]:
    #             pseudo_dice[label_name]["frontal plane"] = frontal_dice.tolist()
    #         else:
    #             pseudo_dice[label_name] = {"frontal plane": frontal_dice.tolist()}
            
    #     empty_tmpdir()
    #     sorted_dict = dict(sorted(pseudo_dice.items(), key=lambda x : x[1]['transverse plane'] + x[1]['frontal plane'], reverse=True))
    #     write_json(sorted_dict,front_our_dir.parent.parent / "dice.json")


    
    # # image_obj = sitk.ReadImage("/mnt/c/Users/yeep/Desktop/partial instance/registration/case/LNQ2023_0002_0000.nrrd")
    # # image = sitk.GetArrayFromImage(image_obj)
    # # label = sitk.GetArrayFromImage(sitk.ReadImage("/mnt/c/Users/yeep/Desktop/partial instance/registration/case/LNQ2023_0002.nrrd"))
    # # print(label.shape)
    # # pseudo_label = synra_labeling(image, label[31], 31)
    # # pseudo_label = pseudo_label.astype(np.uint8)
    # # pseudo_label_obj = sitk.GetImageFromArray(pseudo_label)
    # # pseudo_label_obj.CopyInformation(image_obj)
    # # sitk.WriteImage(pseudo_label_obj,"pseudo_label.nii.gz")
    # # dice = cal_dice(pseudo_label, label,num=2)
    # # print(dice)
        
    
    
    
    