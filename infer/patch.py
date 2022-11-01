import multiprocessing
from multiprocessing import Lock, Pool

multiprocessing.set_start_method("spawn", True)  # ! must be at top for VScode debugging
import multiprocessing as mp
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait
from multiprocessing import Lock, Pool

import numpy as np
import torch
import torch.utils.data as data
import tqdm

from loader.infer_loader import PatchDataset2
from misc.utils import recur_find_ext

from . import base


class InferManager(base.InferManager):
    """Run inference on tiles."""

    def process_file_list(self, run_args, run_paramset):
        """
        Process all patches from the input data.
        """
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        
        # class_names = run_paramset["class_names"] 
        class_names = {
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8"
        }
        nr_classes = len(list(class_names.keys()))

        file_path_list = recur_find_ext(self.input_dir, [".dat"])
        dataset = PatchDataset2(file_path_list, self.patch_input_shape)
        dataloader = data.DataLoader(
            dataset,
            num_workers=self.nr_inference_workers,
            batch_size=self.batch_size,
            drop_last=False,
        )

        pbar = tqdm.tqdm(
            desc="Process Patches",
            leave=True,
            total=int(len(dataloader)),
            ncols=80,
            ascii=True,
            position=0,
        )

        prob_list = []
        true_list = []
        for batch_idx, batch_data in enumerate(dataloader):
            pdat_list, plab_list = batch_data
            plab_list = np.squeeze(plab_list.detach().numpy())
            poutput_list = self.run_step(pdat_list, None)
            prob_list.extend([poutput_list])
            true_list.extend(plab_list)
            pbar.update()
        pbar.close()

        prob_list = np.concatenate(prob_list, axis=0)
        true_list = np.array(true_list)

        # give the metrics
        from sklearn import metrics
        all_ap = []
        for idx in range(nr_classes):
            true_oneclass = true_list == idx
            true_oneclass = true_oneclass.astype('int') # binary array
            prob_oneclass = prob_list[..., idx]
            ap_oneclass = metrics.average_precision_score(true_oneclass, prob_oneclass)
            all_ap.append(ap_oneclass)
            print("%s-AP" % class_names[idx+1], ap_oneclass)
        print('='*40)

        pred_list = np.argmax(np.array(prob_list), -1)

        correct_all = np.sum(true_list == pred_list)
        acc_all = correct_all / true_list.shape[0]

        all_acc = []
        for idx in range(nr_classes):
            true_subset = true_list == idx 
            pred_oneclass = pred_list[true_subset]
            true_oneclass = true_list[true_subset]
            correct_oneclass = np.sum(true_oneclass == pred_oneclass)
            acc_oneclass = correct_oneclass / true_oneclass.shape[0]
            print("%s-accu" % class_names[idx+1], acc_oneclass)
            all_acc.append(acc_oneclass)
        print('='*40)

        f1_score = metrics.f1_score(true_list, pred_list, average=None)
        for idx in range(nr_classes):
            print("%s-F1" % class_names[idx+1], f1_score[idx])
        avg_f1_score = np.mean(f1_score)

        print('='*40)
        print("ALL-accu", acc_all)
        print("AVG-accu", sum(all_acc) / len(all_acc))
        print("AVG-AP", sum(all_ap) / len(all_ap))
        f1_score = metrics.f1_score(true_list, pred_list, average='macro')
        print("AVG-F1", avg_f1_score)
        print('='*40)
        conf_mat = metrics.confusion_matrix(true_list, pred_list, labels=np.arange(nr_classes), normalize='true')
        print(conf_mat)
    
    
        return
