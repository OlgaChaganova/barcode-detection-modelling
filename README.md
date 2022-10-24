# Детекция штрихкодов на изображении

Детекция производится с использованием модели **yolov5s**.

## 1 - Подготовка данных

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

## 2 - Запуск обучения

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

## 3 - Эксперименты

- Ссылка на [эксперимент](https://app.clear.ml/projects/851ea528d07a4d20b08779aec07d8d0a/experiments/2f0176e6d09044d1b233355a2e7a81f6/output/execution) в ClearML

