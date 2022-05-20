# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe / luisernesto.200892@gmail.coom
# Description: Get data from MODIS (require USER KEY and URLs given by MODIS LAADS)
# ****************************************************************************************** (
import os 
import sys
import subprocess

try:
    DATASOURCE = sys.argv[1]
    if DATASOURCE == 'BE':
        if not os.path.isdir(f'../Data/Beijing'):
            os.mkdir(f'../Data/Beijing')
        DATA_DIR = '../Data/Beijing/Satellite_data/'
    elif DATASOURCE == 'SP':
        if not os.path.isdir(f'../Data/SaoPaulo'):
            os.mkdir(f'../Data/SaoPaulo')
        DATA_DIR = '../Data/SaoPaulo/Satellite_data/'
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)
    else:
        print('Input SP: Sao paulo Data or BE: Beijing Data')
except Exception as e:
    raise ValueError('Input SP: Sao paulo Data or BE: Beijing Data')

def create_directory(type_product):
    print(DATA_DIR+type_product)
    if not (os.path.isdir(DATA_DIR+type_product)):
        os.mkdir(DATA_DIR+type_product)
    else:
        print(type_product+' directory exists.')

    HDF_SAVE_DIR = DATA_DIR+type_product+'/HDF'

    if not (os.path.isdir(HDF_SAVE_DIR)):
        os.mkdir(HDF_SAVE_DIR)
    else:
        print(type_product + ' HDF directory exists.')
    return HDF_SAVE_DIR

# API-KEY of LAADS DAAC user
API_KEY = ''

# Sources sent to your email, after placing the order in https://ladsweb.modaps.eosdis.nasa.gov/search/
# Orders for download VNP95 product 
# ex: https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501670881/
VNP46A1_source = ['https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501751886/']
VNP46A1_destination = create_directory('VNP46A1')

# Orders for download MOD13A2 product
MOD13A2_source = ['https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501751885/']
MOD13A2_destination = create_directory('MOD13A2')

# Exec download_laads.py script
for source in VNP46A1_source:
    subprocess.call(["python3", 'download_laads.py', '-s',
                    source, '-d', VNP46A1_destination, '-t', API_KEY])
    print(source + ' downloaded')
print('VNP46A1 downloaded') 


for source in MOD13A2_source:
    subprocess.call(["python3", 'download_laads.py', '-s',
                     source, '-d', MOD13A2_destination, '-t', API_KEY])
    print(source + ' downloaded')
print('MOD13A2 downloaded')