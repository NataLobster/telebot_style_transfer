Данный проект представляет собой реализацию "медленного" попиксельного алгоритма переноса стиля с использованием предобученной vgg19, завернутую в телеграм бот.
В работе использовались следующие источники https://proproprogs.ru/neural_network/delaem-perenos-stiley-izobrazheniy-s-pomoshchyu-keras-i-tensorflow, https://russianblogs.com/article/52493331862/

В качестве виртуальной среди использовался poetry

для запуска приложения необходимо первоначально установить необходимые библиотеки, запустив из каталога проекта

poetry install

дополнительно, для работы с CUDA потребуется установить pytorch в соответсвии с инструкциями https://pytorch.org/ с помощью poetry, например для Linux

poetry run pip3 install torch torchvision torchaudio

запуск приложения осуществляется с помощью 

poetry run app.py

при этом файл settings.ini должен содержать ваш токен для телеграм-бота

пример работы

исходное изображение
![photo_2023-07-08_20-55-02](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/8c18bcc3-2de4-4dbe-aeda-18842b8bcd6f)
+ ![Van_Gogh](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/d5fbfb46-31cc-4ea1-ae6c-d5e04770c329) = ![photo_2023-07-08_20-54-33](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/9b034116-3ec4-4f11-b63e-f0ec37acc063)
+ ![Monet](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/7a549909-7998-442c-be2f-b409b4f4ae67) = ![photo_2023-07-08_20-54-38](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/441e63a2-2ba0-4931-83ec-7c88a49dd2f8)
+ ![Picaso](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/bc9f73e1-e463-4479-81df-3f6dcd4f7173) = ![photo_2023-07-08_20-54-43](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/ca0198c4-70e1-4ff2-8759-de56c5db47a7)
+ ![Roche](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/474fc5c1-c1f4-44a9-ad06-34667e902ca9) = ![photo_2023-07-08_20-54-48](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/2e6ae240-1423-4389-97e8-f286bab75d96)
+ ![Warhol](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/3acbaac7-e86e-4d23-b845-6b642c3975b0) = ![photo_2023-07-08_20-54-53](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/5ff0f5b9-b3e7-4f66-890e-d92efa7f7620)
+ ![matiss](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/7eda25cd-5dbb-42c0-8edd-2439233b67d3) = ![photo_2023-07-08_20-54-57](https://github.com/NataLobster/telebot_style_transfer/assets/70448060/b3ea7056-301a-4cdc-b505-6a9c2d11d801)















