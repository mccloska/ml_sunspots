# swpc_proc

import pandas as pd
from os import path

def stop_read_srs(srs,stop_line):
    
    with open(srs, "r") as fp:
        for num, line in enumerate(fp, 1):
            if stop_line in line:
                return num

def read_srs(srs_file,srs_dir_year):
    stop_line='IA. H-alpha Plages without Spots.'
    line_stop = stop_read_srs(srs_dir_year+srs_file,stop_line)
    n_read = line_stop-11
    cols = ["NOAA No","Location","Lo","Area","Zpc","LL","NN","Mag"]
    df_srs = pd.read_csv(srs_dir_year+srs_file,names=cols,skiprows=10,sep='\s+',nrows=n_read)
    return df_srs

def read_events(event_file,event_dir_year):
    event_path = event_dir_year+event_file
    check_epath = path.exists(event_path)
    cols = ["Event No","Begin","Max","End","Obs","Q","Type","Loc/Frq","Particulars","W/m^2","Reg_No"]
    if check_epath:
            #d_types={"Event No":int64,"Begin":datetime64,"Max":datetime64,"End":datetime64,"Obs":str,"Q":int64,"Type":str,"Loc/Frq":str,"Particulars":str,"??":float64,"Reg_No":str}
        df_events = pd.read_csv(event_path,names=cols,skiprows=12,sep='\s{2,}',
                        dtype={"Event No":str,"Begin":str,"Max":str,"End":str,"Obs":str,"Q":str,"Type":str,"Loc/Frq":str,"Particulars":str,"??":str,"Reg_No":str})
        df_xray = df_events[df_events['Type']=='XRA']
    else:
        df_xray = pd.DataFrame(columns=cols)
    return df_xray