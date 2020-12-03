import urllib.request
from urllib.request import urlretrieve
import numpy as np

dates = np.arange('2017-09','2017-10',dtype='datetime64[D]',format='%y%m%d')
"""f
or date in dates:
    year = date[0:4]
    month = date[5:7]
    day = date[8:10]
    url_name = 'https://www.solarmonitor.org/data/'+year+'/'+month+'/'+day+'/fits/chmi/chmi_06173_ar_'+ar_no+'_'+20170904_004642.fts.gz'
    file = urlretrieve(url_name, 'test.fts.gz')
"""