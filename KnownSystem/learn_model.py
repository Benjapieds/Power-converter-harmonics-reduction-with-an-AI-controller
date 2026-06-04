from data_model import PlantDataBis

from convNextBlocks import ConvNeXtUnet
from trainer_model import UNetTrainerConvNeXt
from matplotlib_init import init_matplotlib

from multiprocessing import freeze_support
import argparse

def main():
        init_matplotlib()
       
        parser = argparse.ArgumentParser()

        parser.add_argument("--root", type=str, required=True)
        parser.add_argument("--save_root", type=str, required=True)
        parser.add_argument("--pastwindow", type=int, default=48)
        parser.add_argument("--depth", type=int, default=3)
        parser.add_argument("--wide", type=int, default=64)
        parser.add_argument("--patch_len", type=int, default=7)
        parser.add_argument("--patch_padding", type=int, default=3)
        parser.add_argument("--gradient_clip_val", type=float, default=0.1)
        parser.add_argument("--time_invariance", type=int, default=0)
        parser.add_argument("--max_epochs", type=int, default=5)

        args = parser.parse_args()
        args.time_invariance = args.time_invariance != 0

        data = PlantDataBis(
                 batch_size=512,
                 pastwindow=args.pastwindow,
                 root=args.root,
         )

        kernel_size, padding, activation = 3, 1, "gelu"

        model = ConvNeXtUnet(
                lr=0.001,
                depth=args.depth,
                channels=args.wide,
                patch_len=args.patch_len,
                patch_stride=1,
                patch_padding=args.patch_padding,
        )

        trainer = UNetTrainerConvNeXt(
                max_epochs=args.max_epochs,
                gradient_clip_val=args.gradient_clip_val,
                time_invariance=args.time_invariance,
                num_gpus=1,
                save_model_folder=args.save_root,
                save_every=250,
                samples_every=250
        )

        trainer.fit(model, data)


if __name__ == "__main__":
    freeze_support()
    main()
