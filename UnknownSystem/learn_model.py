from data_model import FullData
from trainer_model import HypernetTrainerConvNeXt
from deepset_convNeXt import HyperNetwork
from matplotlib_init import init_matplotlib

from multiprocessing import freeze_support
import argparse

def main():
        init_matplotlib()
        
        d = {10, 14, 20, 23, 41, 72, 96}

        val_path = [
            f"COMPLETE"
            for u in d
        ]
        
        learn_path = []
        
        i = 0
        target = 70
        
        while len(learn_path) < target:
            if i not in d:
                learn_path.append(
                    f"COMPLETE"
                )
            i += 1
        
        systems_path = learn_path + val_path
        
        print(systems_path)
        

        
        
        parser = argparse.ArgumentParser()

        parser.add_argument("--save_root", type=str, required=True)
        parser.add_argument("--pastwindow", type=int, default=48)
        parser.add_argument("--depth", type=int, default=3)
        parser.add_argument("--wide", type=int, default=32)
        parser.add_argument("--patch_len", type=int, default=7)
        parser.add_argument("--patch_padding", type=int, default=3)
        parser.add_argument("--gradient_clip_val", type=float, default=0.1)
        parser.add_argument("--time_invariance", type=int, default=0)
        parser.add_argument("--max_epochs", type=int, default=5)
        parser.add_argument("--Ddepth", type=int, default=3)
        parser.add_argument("--Dwide", type=int, default=32)
        parser.add_argument("--Dpatch_len", type=int, default=7)
        parser.add_argument("--Dpatch_padding", type=int, default=3)
        parser.add_argument("--Dpatch_stride", type=int, default=7)
        parser.add_argument("--factor", type=int, default=2)

        args = parser.parse_args()
        args.time_invariance = args.time_invariance != 0
        
        


        dat = FullData(
        batch_size=512,
        root_dirs= systems_path,
        trajectory_dim=5,
        past_window=args.pastwindow,
        num_workers=4,
        num_train=10,
        num_val=1,
    )


        hypernetwork = HyperNetwork(lr= 1e-3,
                            input_channels=1,
                            channels=args.wide,
                            patch_len=args.patch_len,
                            patch_stride=1,
                            patch_padding=args.patch_padding,
                            depth=args.depth,
                            node_dim=(2, 400, 1),
                            channels_Deepset=args.Dwide,
                            patch_len_Deepset=args.Dpatch_len,
                            patch_stride_Deepset=args.Dpatch_stride,
                            patch_padding_Deepset=args.Dpatch_padding,
                            depth_Deepset=args.Ddepth,
                            aggregation="max",
                            input_dim=400,
                            factor=args.factor)
        
        trainer = HypernetTrainerConvNeXt(
                max_epochs = args.max_epochs,
                num_gpus=1,
                gradient_clip_val=args.gradient_clip_val,
                time_invariance = False,
                save_model_folder=args.save_root,
                save_every=1000, 
                samples_every=1000, 
                project_name="TFE",
                model_name="HypernetConvNeXt")

        trainer.fit(hypernetwork, dat)




if __name__ == "__main__":
    freeze_support()
    main()
