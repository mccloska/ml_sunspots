"""
Script to process NOAA SWPC SRS & Event files 
Creates a csv file for each year containing AR properties & all GOES 1.8A X-ray flares to their associated NOAA AR

Input:  "YYYYMMDDSRS.txt"
        "YYYYMMDDevents.txt"

Output: CSV files containing all NOAA ARs with all associated X-Ray Flares (>=B-Class) 
        "YYYYsrs_df.csv"

Author: Aoife McCloskey (March 2021;aoife.mccloskey@dlr.de) 

"""


#%% 
import numpy as numpy
import os
from os import path
import pandas as pd
import datetime
from swpc_proc import read_events
from swpc_proc import read_srs
#%%
srs_dir = r"C:/Users/mccl_ao/Documents/DLR/Codes/ml_sunspots/data/SRS/"
event_dir = r"C:/Users/mccl_ao/Documents/DLR/Codes/ml_sunspots/data/Events/"
year_list=['2019']


for year in year_list:
    df_total = pd.DataFrame()
    srs_dir_year = srs_dir+year+"_SRS/"
    
    
    for srs in os.listdir(srs_dir_year):
        
        current_date= pd.to_datetime(srs[0:8])
        next_date= current_date+datetime.timedelta(days=1)
        prev_date = current_date-datetime.timedelta(days=1)
        txt_next_date=next_date.strftime("%Y%m%d")
        txt_prev_date=prev_date.strftime("%Y%m%d")
        txt_prev_year=prev_date.strftime("%Y")
        txt_next_year= next_date.strftime("%Y")

        df_srs = read_srs(srs,srs_dir_year)
        df_srs.insert(loc = 0, column = 'Date', value = current_date) 
   
        print(df_srs)

        event_file= txt_prev_date+'events.txt'
        event_dir_year = event_dir+txt_prev_year+"_events/"
        df_xray= read_events(event_file,event_dir_year)

        df_xray['Flare_class']= df_xray['Particulars'].str[0]
        count_flare=df_xray.value_counts(subset=['Reg_No','Flare_class'])
        df_count= count_flare.to_frame(name='n_flares').reset_index()
        print(df_xray)
        for regions in df_srs['NOAA No']:
            print(str(regions))
            n_bflare = df_count[(df_count['Reg_No']==str(regions)) & (df_count['Flare_class']=='B')]['n_flares']
            n_cflare = df_count[(df_count['Reg_No']==str(regions)) & (df_count['Flare_class']=='C')]['n_flares']
            n_mflare = df_count[(df_count['Reg_No']==str(regions)) & (df_count['Flare_class']=='M')]['n_flares']
            n_xflare = df_count[(df_count['Reg_No']==str(regions)) & (df_count['Flare_class']=='X')]['n_flares']

            if n_bflare.size > 0:
                n_bflare = n_bflare.values[0]
            else:
                n_bflare = 0
            if n_cflare.size > 0:
                n_cflare = n_cflare.values[0]
            else:
                n_cflare = 0
            if n_mflare.size > 0:
                n_mflare = n_mflare.values[0]
            else:
                n_mflare = 0
            if n_xflare.size > 0:
                n_xflare = n_xflare.values[0]
            else:
                n_xflare = 0
            all_flares = [n_bflare,n_cflare,n_mflare,n_xflare]
            

            rowIndex = df_srs.index[df_srs['NOAA No']==regions]
            df_srs.loc[rowIndex, 'N_Bflare'] = n_bflare
            df_srs.loc[rowIndex, 'N_Cflare'] = n_cflare
            df_srs.loc[rowIndex, 'N_Mflare'] = n_mflare
            df_srs.loc[rowIndex, 'N_Xflare'] = n_xflare
        print(df_srs)
        df_total=df_total.append(df_srs)
    df_total.to_csv(year+"srs_df.csv")
            
            

       

        
      


# %%
