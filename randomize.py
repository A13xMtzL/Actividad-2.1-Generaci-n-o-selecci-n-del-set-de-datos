import os
import shutil
import random

def get_all_files_in_directory(directory):
  for dirpath, _, filenames in os.walk(directory):
    for f in filenames:
      yield os.path.abspath(os.path.join(dirpath, f))

source_dir = 'cd ../../archive_Lego/LEGO brick images v1'
destination_dir_1 = './test'
destination_dir_2 = './train'

all_files = list(get_all_files_in_directory(source_dir))
random.shuffle(all_files)

split_index = int(len(all_files) * 0.7)

for i, file_path in enumerate(all_files):
  if i < split_index:
    destination_dir = destination_dir_1
  else:
    destination_dir = destination_dir_2

  relative_path = os.path.relpath(file_path, source_dir)
  new_path = os.path.join(destination_dir, relative_path)

  os.makedirs(os.path.dirname(new_path), exist_ok=True)
  shutil.copy2(file_path, new_path)