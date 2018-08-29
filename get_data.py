import urllib.request
import urllib.parse
import json
import time


symbols = ['AIRT', 'ATSG', 'ALK', 'ALGT', 'AAL', 'ARCB', 'ASC', 'AAWW', 'AVH',
           'AZUL', 'BSTI', 'BCO', 'BRS', 'CHRW', 'CNI', 'CP', 'CPLP', 'CEA',
           'ZNH', 'VLRS', 'CPA', 'CAAP', 'CMRE', 'CVTI', 'CYRX', 'CYRXW',
           'CSX', 'DAC', 'DAL', 'DHT', 'DCIX', 'DSX', 'DSXN', 'LPG', 'DRYS',
           'EGLE', 'ECHO', 'ERA', 'EURN', 'ESEA', 'EXPD', 'FDX', 'FWRD',
           'FRO', 'GNK', 'GWR', 'GSL', 'GLBS', 'GOL', 'GRIN', 'OMAB', 'PAC',
           'ASR', 'GSH', 'HA', 'HTLD', 'HUBG', 'HUNT', 'HUNTU', 'HUNTW',
           'JBHT', 'JBLU', 'KSU', 'KSU', 'KNX', 'LSTR', 'LTM', 'MRTN',
           'NVGS', 'NNA', 'NM', 'NMM', 'NAO', 'NSC', 'ODFL', 'OSG', 'PTSI',
           'PANL', 'PATI', 'PHII', 'PHIIK', 'PXS', 'RLGT', 'RRTS', 'RYAAY',
           'SB', 'SAIA', 'SNDR', 'SALT', 'SLTB', 'SBBC', 'SBNA', 'STNG',
           'CKH', 'SMHI', 'SHIP', 'SHIPW', 'SSW', 'SSWA', 'SSWN', 'SFL',
           'SINO', 'SKYW', 'LUV', 'SAVE', 'SBLK', 'SBLKZ', 'GASS', 'TK',
           'TOPS', 'TRMD', 'TNP', 'USX', 'UNP', 'UAL', 'UPS', 'ULH', 'USAK',
           'USDP', 'WERN', 'YRCW', 'ZTO']


url_param = {
    'function': 'TIME_SERIES_DAILY',
    'outputsize': 'full',
    'datatype': 'csv'
}
with open('./Resource/secret.json') as f:
    url_param['apikey'] = json.load(f)['apikey']
stock_keys = ['open', 'high', 'low', 'close', 'volume']
stock_metadata = [float, float, float, float, int]


def get_stock(symbol):
    global url_param
    url_param['symbol'] = symbol
    url_ext = urllib.parse.urlencode(url_param)
    url = 'https://www.alphavantage.co/query?{}'.format(url_ext)

    print('preparing to retrieve {}...'.format(symbol))
    data = urllib.request.urlopen(url).read().decode('ascii')
    print('loaded data')

    if data[0] == '{':
        return None
    return data


def store_data(symbol, data):
    print('preparing to write in file...')
    with open('./Data/{}.csv'.format(symbol), 'w') as f:
        f.write(data)
    print('written in file')


for i in symbols:
    data = get_stock(i)
    while data is None:
        print('request failed, trying again...')
        time.sleep(10)
        data = get_stock(i)
    store_data(i, data)
