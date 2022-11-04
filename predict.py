import os.path

from HKSA import HKSA

model_name = 'weibo_senti_100k.csv.txt'
model = HKSA()
if os.path.exists(os.path.join('model', model_name + '.json')):
    print('Loading model...')
    model.load(os.path.join('model', model_name + '.json'))
else:
    raise FileNotFoundError(model_name + "'s model no found.")
predict_list = ['hsn涨rks啦!  ', 'cpu跑的,有显卡那台家长在用', '@hsn涨rks啦! colab', '我看看', '？', '开作', 'Raid10  ',
                '@Raid10 一眼msi', '不是微星还能啥', 'ED HIROSHI  ', '@ED HIROSHI 这啥', 'x', 'Btmy  ',
                '@ED HIROSHI 这啥', '@Btmy ', '虚拟机？', 'woo', '这光威条子真能', 'Btmy  ', '虚拟机？', '@Btmy 实体',
                '直接3200c16了', '/哦哟', '不懂', '虚拟机我还远程干啥', 'ED HIROSHI  ', '虚拟机我还远程干啥',
                '您是不是要找', '小达', '6', '云储么', '@Raid10 ', '咕谷酱  ', '@hsn涨rks啦! colab',
                '@咕谷酱 挂了梯子还是慢', '狗子和暴雷一对了', '哇 是咕谷酱', '[打call]请使用最新版手机QQ体验新功能',
                '[舔屏]请使用最新版手机QQ体验新功能']
predict_list=['哎,这几把什么ai啊,真无语,怎么什么都评正面啊']
result = model.predict(predict_list)
for i in range(len(predict_list)):
    print(result[i], predict_list[i])
