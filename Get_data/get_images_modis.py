#********************************************************
# GEt Data from Modis (required URLs sent by MODIS LAADS)
#********************************************************
import os 
import subprocess

DATA_DIR = '../Data/Satellite_data/'

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

# Information for download VNP95 product
VNP46A1_source = ['https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501670881/',
                  'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501670883/',
                  'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501670886/']
VNP46A1_destination = create_directory('VNP46A1')

# Information for download MOD13A2 product
MOD13A2_source = ['https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501670633/']

MOD13A2_destination = create_directory('MOD13A2')

# Exec download_laads.py script
for source in VNP46A1_source[1:]:
    subprocess.call(["python3", 'download_laads.py', '-s',
                    source, '-d', VNP46A1_destination, '-t', API_KEY])
    print(source + ' downloaded')
print('VNP46A1 downloaded')

for source in MOD13A2_source:
    subprocess.call(["python3", 'download_laads.py', '-s',
                     source, '-d', MOD13A2_destination, '-t', API_KEY])
    print(source + ' downloaded')
print('MOD13A2 downloaded')
##
