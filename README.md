# Детекция штрихкодов на изображении

Детекция производится с использованием модели **yolov5s**.

## Подготовка данных

1. Скачать [архив](https://disk.yandex.ru/d/nk-h0vv20EZvzg) с данными (gorai);


2. Распаковать его в папку `data`;


3. Подготовить датасет (разделить на train / val / test, преобразовать аннотации к нужному формату):

```shell
python src/data/prepare --args
```
или (с дефолтными аргументами)
```commandline
make prepare_data
```

## Запуск обучения

1. Скачать предобученную модель YoloV5 [здесь](https://github.com/ultralytics/yolov5/releases/tag/v6.1). Я использую YoloV5s;

2. Подготовить виртуальное окружение (Python >= 3.9):

```commandline
make install
clearml-init
```

3. Отредактировать конфиги:
   - `configs/data.yml` -- для данных;
   - `configs/hyps.yml` -- для гиперпараметров;
   - `configs/yolov5s.yml` -- для модели.


4. Выполнить (`python yolov5/train.py --help` для вывода списка возможных аргументов):

```commandline
python yolov5/train.py --args
```

или (с дефолтными параметрами)

```commandline
make train
```

