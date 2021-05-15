from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import torch


class ImagesDataset(Dataset):
    def __init__(
        self,
        source_root,
        target_root,
        opts,
        latents_root=None,
        target_transform=None,
        source_transform=None,
    ):

        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.latent_paths = None
        if latents_root is not None:
            self.latent_paths = sorted(
                data_utils.make_latents_dataset(latents_root)
            )

        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = (
            from_im.convert("RGB")
            if self.opts.label_nc == 0
            else from_im.convert("L")
        )

        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert("RGB")

        if self.target_transform:
            to_im = self.target_transform(to_im)
        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im
        latent = None
        if self.latent_paths is not None:
            latent_path = self.latent_paths[index]
            latent = torch.load(latent_path)

        return from_im, to_im, latent
