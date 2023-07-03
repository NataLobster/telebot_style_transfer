from os import getenv, remove, path
from telebot.async_telebot import AsyncTeleBot
import asyncio
import telebot
from telebot import types

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import datasets, models, transforms

from PIL import Image
from tqdm.notebook import tqdm

from torchvision import transforms


TOKEN = getenv('TELEGRAM_BOT_TOKEN') # получаем ТГ токен из системной переменной
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
imsize = 480
style_layers = [1, 6, 11, 20, 25]
content_layers = [21]
loss_layers = style_layers + content_layers
style_weights = [10**3/n**2 for n in [64, 128, 256, 512, 512]]
#style_weights = [0.05, 0.05, 0.2, 0.3, 0.4]
#style_weights = [0.5]*5
conten_weights = [1]
weights = style_weights + conten_weights


'''Секция для модели'''


# загрузка с предобработкой
def image_loader(image_name):
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


def transfer(content_image, style_image):

    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)

    opt_image = content_image.data.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([opt_image], lr=5)
    #optimizer = torch.optim.LBFGS([opt_image], lr=2)

    content_target = extract_layers(content_layers, content_image, model=vgg)
    style_target = extract_layers(style_layers, style_image, model=vgg)
    content_target = [t.detach() for t in content_target]
    style_target = [GramMatrix()(t).detach() for t in style_target]
    target = style_target + content_target

    loss_fn = [StyleLoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)

    max_iter = 120

    for j in range(max_iter):
        def closure():
            optimizer.zero_grad()
            out = extract_layers(loss_layers, opt_image, model=vgg)
            layer_losses = [weights[i] * loss_fn[i](item, target[i]) for i, item
                            in enumerate(out)]
            loss = sum(layer_losses)
            loss.backward()


            return loss

        optimizer.step(closure)

    opt_img = opt_image.data.clone().squeeze(0)
    opt_img = postb(opt_img)

    return opt_img

'''конец секции'''


bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=["help", "start", "main"])
def bot_help(message):
    bot.send_message(message.chat.id, f'Hi, {message.from_user.first_name}!'
                                            f' Бот предназначен для переноса стиля'
                                            ' с одного изображения на другое, '
                                            ' пожалуйста, загрузите фото для обработки')


@bot.message_handler(content_types=['photo'])
def bot_get_photo(message):

    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        print(file_info, type(file_info), file_info.file_path)
        downloaded_file = bot.download_file(file_info.file_path)

        if path.exists('.\images' + '\content_' + str(message.chat.id) + '.jpg'):
            src = '.\own_styles' + '\style_' + str(message.chat.id) + '.jpg'
            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)

            style_image = image_loader('.\own_styles' + '\style_' + str(message.chat.id) + '.jpg')
            content_image = image_loader('.\images' + '\content_' + str(message.chat.id) + '.jpg')
            bot.send_message(message.chat.id, 'Ожидайте, я работаю')
            response = transfer(content_image, style_image)
            bot.send_photo(message.chat.id, response)
            bot.send_message(message.chat.id, "Если понравилось, можете загрузить еще фото")
            remove('.\images' + '\content_' + str(message.chat.id) + '.jpg')
            return

        else:
            src = '.\images' + '\content_' + str(message.chat.id) + '.jpg'
            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)
            style = True

    except Exception as e:
        bot.reply_to(message,e )

    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Выбрать стиль", callback_data='choose'))
    markup.add(types.InlineKeyboardButton("Загрyзить стиль", callback_data='load'))
    bot.reply_to(message, 'Супер! Теперь необходимо изображение для стиля', reply_markup = markup)


@bot.callback_query_handler(func=lambda callback: callback.data == 'choose')
def callback_message(callback):
    markup = types.InlineKeyboardMarkup()
    btn1 = types.InlineKeyboardButton("Ван-Гог", callback_data='van_gogh')
    btn2 = types.InlineKeyboardButton("Моне", callback_data='monet')
    btn3 = types.InlineKeyboardButton("Пикассо", callback_data='picaso')
    markup.row(btn1, btn2, btn3)
    btn4 = types.InlineKeyboardButton("Леа Рош", callback_data='roche')
    btn5 = types.InlineKeyboardButton("Уорхол", callback_data='warhol')
    btn6 = types.InlineKeyboardButton("Матисс", callback_data='matiss')
    markup.row(btn4, btn5, btn6)
    bot.send_message(callback.message.chat.id, 'Доступные стили', reply_markup=markup)


@bot.callback_query_handler(func=lambda callback: callback.data == 'load')
def callback_message(callback):
    bot.send_message(callback.message.chat.id, 'Загрузите фото со стилем')


@bot.callback_query_handler(func=lambda callback: True)
def callback_message(callback):
    match callback.data:

        case 'van_gogh':
            style_image = image_loader('.\styles\Van_Gogh.jpg')
        case 'monet':
            style_image = image_loader('.\styles\Monet.jpg')
        case 'picaso':
            style_image = image_loader('.\styles\Picaso.jpg')
        case 'roche':
            style_image = image_loader('.\styles\Roche.jpg')
        case 'warhol':
            style_image = image_loader('.\styles\Warhol.jpg')
        case 'matiss':
            style_image = image_loader('.\styles\matiss.jpg')
        case _:
            bot.send_message(callback.message.chat.id, 'Что-то пошло не так, попробуйте еще раз')
            if path.exists('.\images' + '\content_' + str(callback.message.chat.id) + '.jpg'):
                remove('.\images' + '\content_' + str(callback.message.chat.id) + '.jpg')


    try:
        content_image = image_loader('.\images' + '\content_' + str(callback.message.chat.id) + '.jpg')
    except Exception as e:
        bot.reply_to(callback.message, e)
    bot.send_message(callback.message.chat.id, 'Ожидайте, я работаю')
    response = transfer(content_image, style_image)
    bot.send_photo(callback.message.chat.id, response)
    bot.send_message(callback.message.chat.id, "Если понравилось, можете загрузить еще фото")
    remove('.\images' + '\content_' + str(callback.message.chat.id) + '.jpg')


# bot.delete_webhook()

bot.polling(none_stop=True)
