import argparse
import subprocess
import os

# Парсим аргументы командной строки
parser = argparse.ArgumentParser(description="Run a specific experiment")
parser.add_argument("--experiment", type=str, help="Name of the experiment to run")
args = parser.parse_args()

# Создаем путь до файла triplet_train.py в указанном эксперименте
train_script = os.path.join('experiments', args.experiment, 'triplet_train.py')

# Запускаем скрипт
subprocess.run(['python', train_script])
