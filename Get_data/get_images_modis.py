import os 
import subprocess

PROJECT_ROOT_DIR = '.'
#os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
SP_DIR = PROJECT_ROOT_DIR+'/Data'

def create_directory(type_product):
    print(type(type_product))
    if not (os.path.isdir(SP_DIR+'/'+type_product)):
        os.mkdir(SP_DIR+'/'+type_product)
    else:
        print(type_product+' directory exists.')

    HDF_SAVE_DIR = SP_DIR+'/'+type_product+'/HDF'

    if not (os.path.isdir(HDF_SAVE_DIR)):
        os.mkdir(HDF_SAVE_DIR)
    else:
        print(type_product + ' HDF directory exists.')
    return HDF_SAVE_DIR

# API-KEY of LAADS DAAC user
API_KEY = 'E4B114B6-9987-11EA-A18A-C5A9BE72AAFA'

# Sources sent to your email, after placing the order in https://ladsweb.modaps.eosdis.nasa.gov/search/
# Information for download MCD19A2 product
MCD19A2_source = ['https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501485422/',
                  'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501485423/',
                  'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501485424/']
MCD19A2_destination = create_directory('MCD19A2')

# Information for download VNP95 product
VNP46A1_source = ['https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501485632/',
                  'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501485633/',
                  'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501485634/']
VNP46A1_destination = create_directory('VNP46A1')

# Information for download MOD13A2 product
MOD13A2_source = ['https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501485635/',
                  'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501485636/',
                  'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501485637/']

MOD13A2_destination = create_directory('MOD13A2')

# Exec download_laads.py script
for source in MCD19A2_source:
    subprocess.call(["python3", PROJECT_ROOT_DIR+'/Download_images/download_laads.py', '-s',
                     source, '-d', MCD19A2_destination, '-t', API_KEY])
    print(source+' downloaded')
print('MCD19A2 downloaded')

for source in VNP46A1_source[1:]:
    subprocess.call(["python3", PROJECT_ROOT_DIR+'/Download_images/download_laads.py', '-s',
                    source, '-d', VNP46A1_destination, '-t', API_KEY])
    print(source + ' downloaded')
print('VNP46A1 downloaded')

for source in MOD13A2_source:
    subprocess.call(["python3", PROJECT_ROOT_DIR+'/Download_images/download_laads.py', '-s',
                     source, '-d', MOD13A2_destination, '-t', API_KEY])
    print(source + ' downloaded')
print('MOD13A2 downloaded')

##
