import os.path

from HKSA import HKSA

model_name = 'mc_qq_group.txt'
model = HKSA()
if os.path.exists(os.path.join('model', model_name + '.json')):
    print('Loading model...')
    model.load(os.path.join('model', model_name + '.json'))
else:
    raise FileNotFoundError(model_name + "'s model no found.")

result = model.predict(predict_list)
for i in range(len(predict_list)):
    print(result[i], predict_list[i])
