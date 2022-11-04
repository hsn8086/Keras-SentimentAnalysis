import os.path

from HKSA import HKSA, Formatter

data_name = 'weibo_senti_100k.csv.txt'
formatter = Formatter.lists
formatter_args = ('\n', '==', 0, 1)

print('Loading data...')
text = open(os.path.join('data', data_name), 'r', encoding='utf8').read()

print('Formatting data...')
train_data = formatter(text, *formatter_args)

model = HKSA()
if os.path.exists(os.path.join('model', data_name + '.json')):
    print('Loading model...')
    model.load(os.path.join('model', data_name + '.json'))
else:
    print('Creating model...')
    model.load_from_data(train_data)
print('Dictionary size: ' + str(len(model.word_dict)))
print('Dataset size: ' + str(len(train_data)))
print('Keys size: ' + str(len(model.keys)))
print('Keys: ' + str(model.keys))

print('Training...')
model.train(train_data, batch_size=5000, epochs=4)

print('Saving...')
model.save(os.path.join('model', data_name + '.json'))
