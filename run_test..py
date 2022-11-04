import os.path

from HKSA import HKSA, Formatter

data_name = 'toutiao_cat_data.txt'
formatter = Formatter.lists
formatter_args = ('\n', '_!_', 3, 2)

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

print('Training...')
model.train(model.train_data, batch_size=5000, epochs=10)
print('Saving...')
model.save(os.path.join('model', data_name + '.json'))
