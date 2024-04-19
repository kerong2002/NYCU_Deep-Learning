def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    import torch
    eps = 1e-6
    pred_mask[pred_mask > 0.5] = torch.tensor(1.0)
    pred_mask[pred_mask <= 0.5] = torch.tensor(0.0)
    common_pix = (abs(pred_mask - gt_mask) < eps).sum()
    total_pix = pred_mask.reshape(-1).shape[0] + gt_mask.reshape(-1).shape[0]
    dice_score = 2 * common_pix / total_pix
    return dice_score

def focal_loss(pred_mask, gt_mask, alpha=0.25, gamma=2):
    import torch
    eps = 1e-6
    loss = -alpha * gt_mask * torch.pow(1 - pred_mask, gamma) * torch.log(pred_mask + eps) - (1 - alpha) * (1 - gt_mask) * torch.pow(pred_mask, gamma) * torch.log(1 - pred_mask + eps)
    return loss.mean()

def dice_loss(pred_mask, gt_mask, eps=1e-8):
    import torch
    intersection = torch.sum(gt_mask * pred_mask) + eps
    union = torch.sum(gt_mask) + torch.sum(pred_mask) + eps
    loss = 1 - (2 * intersection / union)
    return loss

def plot_img(image, pred_mask, mask, filename):
    import torch
    from torchvision.utils import save_image
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    image[:, 0] = image[:, 0] * mean[0] + std[0]
    image[:, 1] = image[:, 1] * mean[1] + std[1]
    image[:, 2] = image[:, 2] * mean[2] + std[2]
    gray_mask = torch.zeros_like(image)
    gray_pred_mask = torch.zeros_like(image)
    for _ in range(gray_pred_mask.shape[0]):
        gray_pred_mask[_] = pred_mask
        gray_mask[_] = mask
    cat = torch.concat((image, gray_pred_mask, gray_mask, image * gray_pred_mask, image * gray_mask), dim=3)
    save_image(cat, f"test_img/{filename}.png")