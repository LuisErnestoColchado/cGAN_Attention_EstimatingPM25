import sys
import  matplotlib.pyplot as plt
import pandas as pd 
from pandas.plotting import scatter_matrix
import math  

try: 
    DATASOURCE = sys.argv[1]
    if not DATASOURCE in ['BE', 'SP']:
        raise 'Invalid parameters: 1: BE or SP'
except Exception as e:
    raise 'Not found parameters: input DATASOURCE: SP or BE'

class setting:
    if DATASOURCE == 'BE':
        DIR_DATA = '../Preprocessing/Results/data_train.csv'
    elif DATASOURCE == 'SP':
        DIR_DATA = '../Preprocessing/Results/data_train_sp_.csv'

if __name__ == '__main__':
    data = pd.read_csv(setting.DIR_DATA)
    data_labeled = data[~data['PM25'].isna()].reset_index(drop=True)
    #print(data_labeled)
    data_analysis = pd.DataFrame(columns=data_labeled.station.unique().tolist())
    
    nrows, ncols = math.ceil(len(data_labeled.station.unique())/ 2), 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(48, 28))
    j = 0
    r = 0
    plt.rcParams.update({'font.size': 25})
    #!r_text, c_text = 0.1, 0.5

    for i, station in enumerate(data_labeled.station.unique()):
        data_analysis.loc[:, station] = data_labeled[data_labeled['station']==station]['PM25'].reset_index(drop=True)
        ax[r, j].plot(data_analysis.loc[:, station], alpha=0.3)
        ax[r, j].set_xlabel('day', fontsize=30)
        ax[r, j].set_ylabel('PM2.5 \n $(\mu g m^{-3})$', fontsize=30)
        stats = data_analysis[station].describe()
        for k, v in stats.items():
            stats[k] = round(v, 4)
        #!ax[r, j].text(0, 0.8, station, horizontalalignment='left', verticalalignment='bottom', transform=ax[r, j].transAxes, fontsize=40)     
        ax[r, j].text(0, 0, f"{station}\n{stats[['min', 'max', 'mean']].to_string()}", horizontalalignment='left', verticalalignment='bottom', transform=ax[r, j].transAxes, fontsize=40)        #plt.figtext(0.75, 0.5, data_analysis.describe())
        r += 1
        if r == ax.shape[0]:
            j += 1
            r = 0
    plt.savefig(f'Graphics/stats_stations_{DATASOURCE}.png')
    
    plt.rcParams.update({'font.size': 13})
    data_analysis.columns = [str(col).split(' ')[-1][:11] for col in data_analysis.columns]
    axes = scatter_matrix(data_analysis, alpha=0.2, diagonal='hist', figsize=(17, 17))#, diagonal='kde')
    corr = data_analysis.corr().values
    for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
        axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center', fontsize=30)
    plt.savefig(f'Graphics/corr_stations_{DATASOURCE}.png')