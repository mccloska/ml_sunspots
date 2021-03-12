from ftplib import FTP

ftp = FTP('ftp.swpc.noaa.gov')
ftp.login()

ftp.retrlines('LIST') 

year_list=['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']

def get_srs():
    for year in year_list:
        ftp.cwd('/pub/warehouse/'+year)
        file_n = year+"_SRS.tar.gz"
        ftp.retrbinary('RETR '+file_n,open("data/"+file_n, 'wb').write)


    ftp.close()


def get_events():
    for year in year_list:
        ftp.cwd('/pub/warehouse/'+year)
        file_n = year+"_events.tar.gz"
        ftp.retrbinary('RETR '+file_n,open("data/"+file_n, 'wb').write)


    ftp.close()