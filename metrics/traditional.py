# Structure distance and metric settings follow the PIE-Bench evaluation (https://github.com/cure-lab/PnPInversion).
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPScore
from torchmetrics.regression import MeanSquaredError
from torchvision import transforms


class DinoStructureDistance:
    """MSE between the self-similarity of DINO ViT-B/8 keys (last block) of two images."""

    def __init__(self, device):
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(device).eval()
        self.transform = transforms.Compose(
            [transforms.Resize(224, max_size=480), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )

    def _key_self_similarity(self, image):
        attn = self.model.blocks[11].attn
        outputs = []
        handle = attn.qkv.register_forward_hook(lambda module, inputs, output: outputs.append(output))
        self.model(image)
        handle.remove()
        qkv = outputs[0]
        batch, tokens, dim = qkv.shape
        keys = qkv.reshape(batch, tokens, 3, attn.num_heads, dim // 3 // attn.num_heads).permute(2, 0, 3, 1, 4)[1][0]
        keys = keys.permute(1, 0, 2).reshape(tokens, -1)[None]
        norm = keys.norm(dim=2, keepdim=True)
        return (keys @ keys.permute(0, 2, 1)) / torch.clamp(norm @ norm.permute(0, 2, 1), min=1e-8)

    @torch.no_grad()
    def __call__(self, reference, prediction):
        loss = 0.0
        for ref, pred in zip(reference, prediction):
            ref_sim = self._key_self_similarity(self.transform(ref).unsqueeze(0))
            pred_sim = self._key_self_similarity(self.transform(pred).unsqueeze(0))
            loss += F.mse_loss(pred_sim, ref_sim)
        return loss


class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        self.clip = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)
        self.mse = MeanSquaredError().to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.structure = DinoStructureDistance(device)

    def _tensors(self, img_pred, img_gt, mask=None, normalize=True, batched=True):
        tensors = []
        for image in (img_pred, img_gt):
            image = np.array(image).astype(np.float32)
            if mask is not None:
                image = image * mask.astype(np.float32)
            if normalize:
                image = image / 255.0
            tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(self.device)
            tensors.append(tensor.unsqueeze(0) if batched else tensor)
        return tensors

    def clip_similarity(self, image, text, mask=None):
        image = np.array(image)
        if mask is not None:
            image = np.uint8(image * mask)
        return self.clip(torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(self.device), text).item()

    def psnr_score(self, img_pred, img_gt, mask=None):
        return self.psnr(*self._tensors(img_pred, img_gt, mask)).item()

    def lpips_score(self, img_pred, img_gt, mask=None):
        pred, gt = self._tensors(img_pred, img_gt, mask)
        return self.lpips(pred * 2 - 1, gt * 2 - 1).item()

    def mse_score(self, img_pred, img_gt, mask=None):
        pred, gt = self._tensors(img_pred, img_gt, mask, batched=False)
        return self.mse(pred.contiguous(), gt.contiguous()).item()

    def ssim_score(self, img_pred, img_gt, mask=None):
        return self.ssim(*self._tensors(img_pred, img_gt, mask)).item()

    def structure_distance(self, img_pred, img_gt):
        pred, gt = self._tensors(img_pred, img_gt, normalize=False)
        return self.structure(gt, pred).item()
