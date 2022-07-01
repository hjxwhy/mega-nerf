from typing import Tuple, Optional

import torch

from mega_nerf.image_metadata import ImageMetadata


def get_rgb_index_mask(metadata: ImageMetadata) -> Optional[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    rgbs = metadata.load_image().view(-1, 3)

    keep_mask = metadata.load_mask() # 这里训练是每个region单独训练的，所以数据加载的时候也是单独加载的每个region的mask，

    if metadata.is_val:
        if keep_mask is None:
            keep_mask = torch.ones(metadata.H, metadata.W, dtype=torch.bool)
        else:
            # Get how many pixels we're discarding that would otherwise be added
            discard_half = keep_mask[:, metadata.W // 2:]
            discard_pos_count = discard_half[discard_half == True].shape[0]

            candidates_to_add = torch.arange(metadata.H * metadata.W).view(metadata.H, metadata.W)[:, :metadata.W // 2]
            keep_half = keep_mask[:, :metadata.W // 2]
            candidates_to_add = candidates_to_add[keep_half == False].reshape(-1)
            to_add = candidates_to_add[torch.randperm(candidates_to_add.shape[0])[:discard_pos_count]]

            keep_mask.view(-1).scatter_(0, to_add, torch.ones_like(to_add, dtype=torch.bool))

        keep_mask[:, metadata.W // 2:] = False

    if keep_mask is not None:
        if keep_mask[keep_mask == True].shape[0] == 0: # 如果这个的所有像素和这个region完全没有关系，这个mask就都是false，就返回None
            return None

        keep_mask = keep_mask.view(-1)
        rgbs = rgbs[keep_mask == True]

    assert metadata.image_index <= torch.iinfo(torch.int32).max
    return rgbs, metadata.image_index * torch.ones(rgbs.shape[0], dtype=torch.int32), keep_mask
