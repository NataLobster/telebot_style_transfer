from torch import nn
import torch
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GramMatrix(nn.Module):
    def forward(self, inp):
        b, c, h, w = inp.size()
        features = inp.view(b, c, h*w)
        gram_matrix = torch.bmm(features, features.transpose(1,2))
        gram_matrix.div_(h*w)
        return gram_matrix


class StyleLoss(nn.Module):
    def forward(self, inputs, targets):
        out = nn.MSELoss()(GramMatrix()(inputs), targets)
        return out


class LayerActivations():
    features = []

    def __init__(self, model, layer_nums):
        self.hooks = []
        for layer_num in layer_nums:
            self.hooks.append(model[layer_num].register_forward_hook(self.hook_fn))

    def hook_fn(self, module, inp, outp):
        self.features.append(outp)

    def remove(self):
        for hook in self.hooks:
            hook.remove()


def extract_layers(layers, img, model=None):
    la = LayerActivations(model, layers)
    la.features = []
    model = model.to(DEVICE)
    img = img.to(DEVICE)
    out = model(img)
    la.remove()
    return la.features


def transfer(content_image, style_image, content_layers, style_layers, loss_layers, weights):

    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)

    opt_image = content_image.data.clone().requires_grad_(True)

    if DEVICE == 'cuda':
        optimizer = torch.optim.LBFGS([opt_image], max_iter=4)
        max_iter = 200

    else:
        optimizer = torch.optim.Adam([opt_image], lr=8)
        max_iter = 250

    # optimizer = torch.optim.Adam([opt_image], lr=5)
    # max_iter = 150

    content_target = extract_layers(content_layers, content_image, model=vgg)
    style_target = extract_layers(style_layers, style_image, model=vgg)
    content_target = [t.detach() for t in content_target]
    style_target = [GramMatrix()(t).detach() for t in style_target]
    target = style_target + content_target

    loss_fn = [StyleLoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)

    for j in range(max_iter):
        def closure():
            optimizer.zero_grad()
            print(j)
            out = extract_layers(loss_layers, opt_image, model=vgg)
            layer_losses = [weights[i] * loss_fn[i](item, target[i]) for i, item
                            in enumerate(out)]
            loss = sum(layer_losses)
            loss.backward()

            return loss

        optimizer.step(closure)

    opt_img = opt_image.data.clone().squeeze(0)

    return opt_img
