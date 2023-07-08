Данный проект представляет собой реализацию "медленного" попиксельного алгоритма переноса стиля с использованием предобученной vgg19, завернутую в телеграм бот.
В работе использовались следующие источники 
- https://proproprogs.ru/neural_network/delaem-perenos-stiley-izobrazheniy-s-pomoshchyu-keras-i-tensorflow,
- https://russianblogs.com/article/52493331862/

В качестве виртуальной среди использовался poetry

Для запуска приложения необходимо первоначально установить необходимые библиотеки, запустив из каталога проекта
```sh
poetry install
````
Дополнительно, для работы с CUDA потребуется установить pytorch в соответсвии с инструкциями https://pytorch.org/ с помощью poetry, например для Linux
```sh
poetry run pip3 install torch torchvision torchaudio
````
Запуск приложения осуществляется с помощью 
```sh
poetry run app.py
````
При этом файл settings.ini должен содержать ваш токен для телеграм-бота

***Пример работы***

Исходное изображение + Стиль = Результат

<img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/8c18bcc3-2de4-4dbe-aeda-18842b8bcd6f" width="300"> + <img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/d5fbfb46-31cc-4ea1-ae6c-d5e04770c329" width="300"> = <img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/9b034116-3ec4-4f11-b63e-f0ec37acc063" width="300">

<img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/8c18bcc3-2de4-4dbe-aeda-18842b8bcd6f" width="300"> + <img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/7a549909-7998-442c-be2f-b409b4f4ae67" width="300"> = <img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/441e63a2-2ba0-4931-83ec-7c88a49dd2f8" width="300">

<img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/8c18bcc3-2de4-4dbe-aeda-18842b8bcd6f" width="300"> + <img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/bc9f73e1-e463-4479-81df-3f6dcd4f7173" width="300"> = <img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/ca0198c4-70e1-4ff2-8759-de56c5db47a7" width="300">

<img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/8c18bcc3-2de4-4dbe-aeda-18842b8bcd6f" width="300"> + <img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/474fc5c1-c1f4-44a9-ad06-34667e902ca9" width="300"> = <img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/2e6ae240-1423-4389-97e8-f286bab75d96" width="300">

<img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/8c18bcc3-2de4-4dbe-aeda-18842b8bcd6f" width="300"> + <img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/3acbaac7-e86e-4d23-b845-6b642c3975b0" width="300"> = <img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/5ff0f5b9-b3e7-4f66-890e-d92efa7f7620" width="300">

<img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/8c18bcc3-2de4-4dbe-aeda-18842b8bcd6f" width="300"> + <img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/7eda25cd-5dbb-42c0-8edd-2439233b67d3" width="300"> = <img src="https://github.com/NataLobster/telebot_style_transfer/assets/70448060/b3ea7056-301a-4cdc-b505-6a9c2d11d801" width="300">















