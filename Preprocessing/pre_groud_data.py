import os
import pandas as pd 
import numpy as np 
import glob 
from tqdm import tqdm

# Setting 
class setting:
    directory_data = '../Data/Ground_data'
    num_files = len([name for name in os.listdir(directory_data) if os.path.isfile(name)])


# Function read the data for a directory (format: csv)
def read_data():
    dict_data = {}
    print(setting.num_files)
    with tqdm(total=len(glob.glob(os.path.join(setting.directory_data, '*.csv')))) as bar:
        for filename in glob.glob(os.path.join(setting.directory_data, '*.csv')):
            station = filename.split('_')[3]
            data = pd.read_csv(filename)
            dict_data.update({station: data})
            bar.set_description(f'Reading data from {station} station')
            bar.update(1)
    return dict_data

if __name__ == '__main__':
    dict_data = read_data()
    print([k for k, v in dict_data.items()])

