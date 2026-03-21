from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPScore
from torchmetrics.regression import MeanSquaredError
from torchvision import transforms
from torchvision.transforms import Resize


class DinoStructureDistance:
    def __init__(self, device: str) -> None:
        self.layer_num = 11
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(
            device
        )
        self.model.eval()
        self.transform = transforms.Compose(
            [
                Resize(224, max_size=480),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

    def _capture_qkv(self, image_tensor: torch.Tensor) -> torch.Tensor:
        outputs = []

        def hook(_, __, output):
            outputs.append(output)

        handler = self.model.blocks[self.layer_num].attn.qkv.register_forward_hook(hook)
        try:
            self.model(image_tensor)
        finally:
            handler.remove()
        return outputs[0]

    def _extract_keys(self, qkv: torch.Tensor) -> torch.Tensor:
        if qkv.dim() != 3:
            raise ValueError(f"Unexpected qkv shape: {tuple(qkv.shape)}")

        batch_size, token_count, total_dim = qkv.shape
        embed_dim = total_dim // 3
        num_heads = self.model.blocks[self.layer_num].attn.num_heads
        head_dim = embed_dim // num_heads
        qkv = qkv.reshape(batch_size, token_count, 3, num_heads, head_dim).permute(
            2, 0, 3, 1, 4
        )
        return qkv[1]

    def _keys_self_similarity(self, image_tensor: torch.Tensor) -> torch.Tensor:
        qkv = self._capture_qkv(image_tensor)
        keys = self._extract_keys(qkv)[0]
        token_count = keys.shape[1]
        concatenated_keys = keys.permute(1, 0, 2).reshape(token_count, -1)
        return self._cosine_similarity(concatenated_keys[None, None, ...])

    def _cosine_similarity(self, tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        tensor = tensor[0]
        norm = tensor.norm(dim=2, keepdim=True)
        denom = torch.clamp(norm @ norm.permute(0, 2, 1), min=eps)
        return (tensor @ tensor.permute(0, 2, 1)) / denom

    def calculate(self, reference: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        with torch.no_grad():
            for reference_image, prediction_image in zip(reference, prediction):
                reference_image = self.transform(reference_image).unsqueeze(0)
                prediction_image = self.transform(prediction_image).unsqueeze(0)
                reference_sim = self._keys_self_similarity(reference_image)
                prediction_sim = self._keys_self_similarity(prediction_image)
                loss += F.mse_loss(prediction_sim, reference_sim)
        return loss


class MetricsCalculator:
    def __init__(self, device) -> None:
        self.device = device
        self.clip_metric_calculator = CLIPScore(
            model_name_or_path="openai/clip-vit-large-patch14"
        ).to(device)
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(
            net_type="squeeze"
        ).to(device)
        self.mse_metric_calculator = MeanSquaredError().to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(device)
        self.structure_distance_metric_calculator = DinoStructureDistance(device=device)

    def _apply_mask(self, image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None:
            return image
        return image * np.array(mask).astype(np.float32)

    def _to_tensor(
        self, image: np.ndarray, normalize: bool, batched: bool
    ) -> torch.Tensor:
        if normalize:
            image = image / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(
            self.device
        )
        if batched:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _prepare_image_pair(
        self,
        img_pred,
        img_gt,
        mask_pred=None,
        mask_gt=None,
        normalize: bool = True,
        batched: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_pred = np.array(img_pred).astype(np.float32)
        img_gt = np.array(img_gt).astype(np.float32)
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        img_pred = self._apply_mask(img_pred, mask_pred)
        img_gt = self._apply_mask(img_gt, mask_gt)
        return (
            self._to_tensor(img_pred, normalize=normalize, batched=batched),
            self._to_tensor(img_gt, normalize=normalize, batched=batched),
        )

    def calculate_clip_similarity(self, img, txt, mask=None):
        image = np.array(img)
        if mask is not None:
            image = np.uint8(image * np.array(mask))
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).to(self.device)
        score = self.clip_metric_calculator(image_tensor, txt)
        return score.cpu().item()

    def calculate_psnr(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred_tensor, img_gt_tensor = self._prepare_image_pair(
            img_pred, img_gt, mask_pred, mask_gt, normalize=True, batched=True
        )
        score = self.psnr_metric_calculator(img_pred_tensor, img_gt_tensor)
        return score.cpu().item()

    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred_tensor, img_gt_tensor = self._prepare_image_pair(
            img_pred, img_gt, mask_pred, mask_gt, normalize=True, batched=True
        )
        score = self.lpips_metric_calculator(
            img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1
        )
        return score.cpu().item()

    def calculate_mse(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred_tensor, img_gt_tensor = self._prepare_image_pair(
            img_pred, img_gt, mask_pred, mask_gt, normalize=True, batched=False
        )
        score = self.mse_metric_calculator(
            img_pred_tensor.contiguous(), img_gt_tensor.contiguous()
        )
        return score.cpu().item()

    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred_tensor, img_gt_tensor = self._prepare_image_pair(
            img_pred, img_gt, mask_pred, mask_gt, normalize=True, batched=True
        )
        score = self.ssim_metric_calculator(img_pred_tensor, img_gt_tensor)
        return score.cpu().item()

    def calculate_structure_distance(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred_tensor, img_gt_tensor = self._prepare_image_pair(
            img_pred, img_gt, mask_pred, mask_gt, normalize=False, batched=True
        )
        score = self.structure_distance_metric_calculator.calculate(
            img_gt_tensor, img_pred_tensor
        )
        return score.cpu().item()
