from torchvision import transforms
import torch
from PIL import Image
def image_loader(image_name, imsize):
    prep = transforms.Compose([transforms.Resize(imsize),
                          transforms.ToTensor(),


                          transforms.Lambda(lambda x:
                                            x[torch.LongTensor([2,1,0])]),
                              transforms.Normalize([0.485, 0.456,
                                                    0.406], [1, 1, 1]),
                              transforms.Lambda(lambda x: x.mul_(255))])
    image = Image.open(image_name)
    image = prep(image)
    image = image.unsqueeze(0)
    return image


# постобработка
def postb(tensor):
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1/255)),
                             transforms.Normalize([-0.485, -0.456,
                                                    -0.406], [1, 1, 1]),
                             transforms.Lambda(lambda x:
                                            x[torch.LongTensor([2,1,0])]),])
    postpb = transforms.Compose([transforms.ToPILImage()])
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    im = postpb(t)
    return im