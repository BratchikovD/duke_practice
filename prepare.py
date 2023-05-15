import os
from shutil import copyfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'DukeMTMC-reID')
PARSED_DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.isdir(DATA_DIR):
    print("Отсутствет папка с данными.")

TRAIN_DATA = os.path.join(DATA_DIR, 'bounding_box_train')
TEST_DATA = os.path.join(DATA_DIR, 'bounding_box_test')
QUERY_DATA = os.path.join(DATA_DIR, 'query')

TRAIN_DIR = os.path.join(PARSED_DATA_DIR, 'train')
TEST_DIR = os.path.join(PARSED_DATA_DIR, 'test')
VAL_DIR = os.path.join(PARSED_DATA_DIR, 'val')
QUERY_DIR = os.path.join(PARSED_DATA_DIR, 'query_data')

if not os.path.isdir(PARSED_DATA_DIR):
    os.mkdir(PARSED_DATA_DIR)

if not os.path.isdir(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)

if not os.path.isdir(TEST_DIR):
    os.mkdir(TEST_DIR)

if not os.path.isdir(VAL_DIR):
    os.mkdir(VAL_DIR)

if not os.path.isdir(QUERY_DIR):
    os.mkdir(QUERY_DIR)

for root, dirs, files in os.walk(TRAIN_DATA, topdown=True):
    print("Обрабатываем изображения в bounding_box_train")
    val_images = 0
    for name in files:
        ID = name.split('_')[0]
        TRAIN_ID_DIR = os.path.join(TRAIN_DIR, ID)
        VAL_ID_DIR = os.path.join(VAL_DIR, ID)
        if not os.path.isdir(TRAIN_ID_DIR):
            val_images = 0
            os.mkdir(TRAIN_ID_DIR)
            os.mkdir(VAL_ID_DIR)
        if val_images < 5:
            copyfile(os.path.join(TRAIN_DATA, name), os.path.join(VAL_ID_DIR, name))
            val_images += 1
        else:
            copyfile(os.path.join(TRAIN_DATA, name), os.path.join(TRAIN_ID_DIR, name))

for root, dirs, files in os.walk(TEST_DATA, topdown=True):
    print("Обрабатываем изображения в bounding_box_test")
    for name in files:
        ID = name.split('_')[0]
        TEST_ID_DIR = os.path.join(TEST_DIR, ID)
        if not os.path.isdir(TEST_ID_DIR):
            os.mkdir(TEST_ID_DIR)
        copyfile(os.path.join(TEST_DATA, name), os.path.join(TEST_ID_DIR, name))

for root, dirs, files in os.walk(QUERY_DATA, topdown=True):
    print("Обрабатываем изображения в query")
    for name in files:
        ID = name.split('_')[0]
        QUERY_ID_DIR = os.path.join(QUERY_DIR, ID)
        if not os.path.isdir(QUERY_ID_DIR):
            os.mkdir(QUERY_ID_DIR)
        copyfile(os.path.join(QUERY_DATA, name), os.path.join(QUERY_ID_DIR, name))
