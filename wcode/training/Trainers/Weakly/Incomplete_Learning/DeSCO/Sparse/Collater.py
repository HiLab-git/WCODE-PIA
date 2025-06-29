import numpy as np
import torch
from threadpoolctl import threadpool_limits

from wcode.training.dataloader.Patchsampler import Patchsampler

class LabelWeightedCollater(object):
    def __init__(
            self,
            slice_weight,
            batchsize,
            sample_patch_size,
            final_patch_size,
            oversample_foreground_percent: float,
            probabilistic_oversampling: bool,
            transform=None,
        ):  
            self.batchsize = batchsize
            self.transform = transform
            self.Patchsampler = Patchsampler(
                sample_patch_size,
                final_patch_size,
                oversample_foreground_percent,
                probabilistic_oversampling,
            )
            self.slice_weight = np.array(slice_weight)
        
    def add_weights(self, label):
        coronal_slice,trans_slice,reg_coronal_label,reg_trans_label = label
        _, cor_labeled_idx, _ = np.where(coronal_slice)
        cor_labeled_idx = np.unique(cor_labeled_idx)
        coronal_slice_w = np.asarray([1])
        for index in cor_labeled_idx:
            coronal_slice_w = coronal_slice_w * self.slice_weight ** np.abs(np.arange(0, coronal_slice.shape[1]) - index)
        coronal_slice_w = coronal_slice_w[None, : ,None] * np.ones_like(coronal_slice)
                
        trans_labeled_idx,_, _ = np.where(trans_slice)
        trans_labeled_idx = np.unique(trans_labeled_idx)
        trans_slice_w = np.asarray([1])
        for index in trans_labeled_idx:
            trans_slice_w = trans_slice_w * self.slice_weight ** np.abs(np.arange(0, coronal_slice.shape[0]) - index)
        trans_slice_w = trans_slice_w[:,None, None] * np.ones_like(trans_slice)

        coronal_slice_w[trans_labeled_idx] = 1
        trans_slice_w[:,cor_labeled_idx] = 1

        label = np.stack([reg_coronal_label, reg_trans_label, coronal_slice_w,trans_slice_w,coronal_slice,trans_slice])
        return label


    def __call__(self, data):
        """
        Dataloader will give a list object to collate_fn method. Each element in it is the return of the __getitem__
        method in your dataset.
        image and label is (c, z, y, x), so it is needed to add one more channel to make the shape is (b, c, z, y, x)
        """
        # We manually pass in batchsize separately, because if the number of batchsize larger than the size of dataset,
        # the length of data here is smaller than the batchsize we want.
        # You may meet this situation when we do a patch-based training on Whole Slide Image (WSI). There are 30 WSIs, for
        # example, we set batchsize to 32. So in Patchsampler, we need to sample two more patches.
        output_dict = {key: [] for key in data[0].keys()}
        image_lst = []
        label_lst = []
        oversample_lst = []
        for i in range(len(data)):
            image_lst.append(data[i]["image"])
            label_lst.append(self.add_weights(data[i]['label']))
            oversample_lst.append(data[i]["property"]["oversample"])
        image, label, idx_lst = self.Patchsampler.run(
            image_lst, label_lst, oversample_lst, self.batchsize
        )
        
        if self.transform is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    image_all = torch.from_numpy(image).float()
                    label_all = torch.from_numpy(label).to(torch.float32)
                    image_lst = []
                    label_lst = []
                    for i, idx in zip(list(range(image.shape[0])), idx_lst):

                        sample_data = self.transform(
                            **{"image": image_all[i], "label": label_all[i]}
                        )
                        image_lst.append(sample_data["image"])
                        label_lst.append(sample_data["label"])
                        output_dict["idx"].append(data[idx]["idx"])
                        output_dict["property"].append(data[idx]["property"])
                    image_all = torch.stack(image_lst)
                    if isinstance(label_lst[0], list):
                        label_all = [
                            torch.stack([s[i] for s in label_lst])
                            for i in range(len(label_lst[0]))
                        ]
                    else:
                        label_all = torch.stack(label_lst)
                    del image_lst, label_lst

        # we use the vstack method in torch here because the image and label
        # are Tensors after transform.
        output_dict["image"] = image_all
        output_dict["label"] = label_all
        # print("Augmentation: {}s".format(time.time() - start))
        return output_dict

        
            
        
            
            
            
           