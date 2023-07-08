from os import getenv, remove, path

import telebot
from telebot import types
import configparser

import torch
from torch import nn
from torchvision import models

import preprocessing
import calculating


#TOKEN = getenv('TELEGRAM_BOT_TOKEN') # получаем ТГ токен из системной переменной
config = configparser.ConfigParser()  # создаём объекта парсера
config.read("settings.ini")

TOKEN = config["Telegram"]["token"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
imsize = 480
style_layers = [1, 6, 11, 20, 25]
content_layers = [21]
loss_layers = style_layers + content_layers
#style_weights = [10**3/n**2 for n in [64, 128, 256, 512, 512]]
#style_weights = [0.4, 0.3, 0.2, 0.05, 0.05]
style_weights = [0.6, 0.4, 0.2, 0.1, 0.05]
conten_weights = [1]
weights = style_weights + conten_weights


bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=["help", "start", "main"])
def bot_help(message):
    bot.send_message(message.chat.id, f'Hi, {message.from_user.first_name}!'
                                            f' Бот предназначен для переноса стиля'
                                            ' с одного изображения на другое, '
                                            ' пожалуйста, загрузите фото для обработки')

@bot.message_handler(content_types= ['text', 'audio', 'document', 'sticker', 'video', 'video_note', 'voice', 'location',
                                     'contact', 'left_chat_member',' new_chat_title', 'new_chat_photo', 'supergroup_chat_created',
                                     'channel_chat_created', 'migrate_to_chat_id', 'migrate_from_chat_id', 'pinned_message', 'web_app_data'])
def bot_get_content(message):
    bot.reply_to(message, 'Я не понимаю никаких сообщений кроме фото. Загрузите фото для обработки или дождитесь ответа на предыдущее')

@bot.message_handler(content_types=['photo'])
def bot_get_photo(message):

    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        print(file_info, type(file_info), file_info.file_path)
        downloaded_file = bot.download_file(file_info.file_path)

        if path.exists('./images' + '/content_' + str(message.chat.id) + '.jpg'):
            src = './own_styles' + '/style_' + str(message.chat.id) + '.jpg'
            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)

            style_image = preprocessing.image_loader('./own_styles' + '/style_' + str(message.chat.id) + '.jpg', imsize)
            content_image = preprocessing.image_loader('./images' + '/content_' + str(message.chat.id) + '.jpg', imsize)
            bot.send_message(message.chat.id, 'Ожидайте, я работаю')
            response = calculating.transfer(content_image, style_image,content_layers, style_layers, loss_layers, weights)
            response = preprocessing.postb(response)
            bot.send_photo(message.chat.id, response)
            bot.send_message(message.chat.id, "Если понравилось, можете загрузить еще фото")
            remove('./images' + '/content_' + str(message.chat.id) + '.jpg')
            return

        else:
            src = './images' + '/content_' + str(message.chat.id) + '.jpg'
            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)


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
            style_image = preprocessing.image_loader('./styles/Van_Gogh.jpg', imsize)
        case 'monet':
            style_image = preprocessing.image_loader('./styles/Monet.jpg', imsize)
        case 'picaso':
            style_image = preprocessing.image_loader('./styles/Picaso.jpg', imsize)
        case 'roche':
            style_image = preprocessing.image_loader('./styles/Roche.jpg', imsize)
        case 'warhol':
            style_image = preprocessing.image_loader('./styles/Warhol.jpg', imsize)
        case 'matiss':
            style_image = preprocessing.image_loader('./styles/matiss.jpg', imsize)
        case _:
            bot.send_message(callback.message.chat.id, 'Что-то пошло не так, попробуйте еще раз')
            if path.exists('./images' + '/content_' + str(callback.message.chat.id) + '.jpg'):
                remove('./images' + '/content_' + str(callback.message.chat.id) + '.jpg')


    try:
        content_image = preprocessing.image_loader('./images' + '/content_' + str(callback.message.chat.id) + '.jpg', imsize)
    except Exception as e:
        bot.reply_to(callback.message, 'Что-то пошло не так, попробуйте еще раз')
    bot.send_message(callback.message.chat.id, 'Ожидайте, я работаю')
    response = calculating.transfer(content_image, style_image,content_layers, style_layers, loss_layers, weights)
    response = preprocessing.postb(response)
    bot.send_photo(callback.message.chat.id, response)
    bot.send_message(callback.message.chat.id, "Если понравилось, можете загрузить еще фото")
    remove('./images' + '/content_' + str(callback.message.chat.id) + '.jpg')


# bot.delete_webhook()

bot.polling(none_stop=True)
