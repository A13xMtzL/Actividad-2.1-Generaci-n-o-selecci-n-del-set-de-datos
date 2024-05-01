import os
import shutil
import random


def get_all_files_in_directory(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


source_dir = "A:/Escuela/Octavos_Semestre/archive_Lego/LEGO brick images v1"

train_dir = "A:/Escuela/Octavos_Semestre/M2_IA/new__/_train"
test_dir = "A:/Escuela/Octavos_Semestre/M2_IA/new__/_test"
validation_dir = (
    "A:/Escuela/Octavos_Semestre/M2_IA/new__/_validation"
)

# Number of times to shuffle the list
num_shuffles = 7

all_files = list(get_all_files_in_directory(source_dir))

# Shuffle the list multiple times
for _ in range(num_shuffles):
    random.shuffle(all_files)

train_split_index = int(len(all_files) * 0.7)
test_split_index = int(len(all_files) * 0.85)  # This should be an index, not an offset

for i, file_path in enumerate(all_files):
    if i < train_split_index:
        destination_dir = train_dir
    elif i < test_split_index:
        destination_dir = test_dir
    else:
        destination_dir = validation_dir

    relative_path = os.path.relpath(file_path, source_dir)
    new_path = os.path.join(destination_dir, relative_path)

    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    shutil.copy2(file_path, new_path)