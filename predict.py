import os.path

from HKSA import HKSA

data_name = 'toutiao_cat_data.txt'
model = HKSA()
if os.path.exists(os.path.join('model', data_name + '.json')):
    print('Loading model...')
    model.load(os.path.join('model', data_name + '.json'))
else:
    raise FileNotFoundError(data_name + "'s model no found.")
model.predict(['中国网红竟红到美国？不多说了，连小编都心动了'])
