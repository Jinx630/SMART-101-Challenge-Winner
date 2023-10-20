import json
from tqdm import tqdm

data = json.load(open('dataset/VLAR-test.json','r'))['VLAR']
val = []
qid2type = json.load(open('ckpts/id2type.json','r'))

cls2id = {'biking':0,
 'umbrella':1,
 'feline':2,
 'phone':3,
 'arrow directions':4,
 'hand':5,
 'building':6,
 'vehicle':7,
 'cloud':8,
 'family':9,
 'books':10,
 'cartwheeling':11,
 'bunny ears':12,
 'hat':13,
 'moon':14,
 'star':15,
 'water polo':16,
 'lock':17,
 'holding hands':18,
 'boat':19,
 'ball':20,
 'fast train':21,
 'heart':22,
 'numbers':23,
 'envelope':24,
 'drinks':25,
 'prohibit sign':26,
 'flag':27,
 'writing utensil':28,
 'worker':29,
 'bird':30,
 'blade':31,
 'emotion face':32,
 'airplane':33,
 'clock':34,
 'footwear':35,
 'kiss':36,
 'money':37,
 'japanese ideograph':38,
 'monkey':39,
 'marine animals':40,
 'mailbox':41,
 'tree':42,
 'flower':43,
 'surfing':44,
 'medal':45,
 'mountain':46,
 'golfing':47,
 'disk':48,
 'wrestling':49}

for r,row in tqdm(enumerate(data[1:])):
    # print(row[8])
    iid = row['Id']
    question = row['Question']
    img = row['Image']
    # if not question.endswith('?'):
    #     question+='?'
    
    objs = []
    try:
        for obj in open(f"yolov7/test/exp/labels/{img.rstrip('.png')}.txt",'r').readlines():
            cdd,x1,y1,x2,y2 = obj.rstrip('\n').split()
            objs.append([cid2cls[int(cdd)],round(float(x1),3),round(float(y1),3),round(float(x2),3),round(float(y2),3)])
    except:
        pass

    a,b,c,d,e = row['A'],row['B'],row['C'],row['D'],row['E']
    answer = []

    meta = {
        "question_id": iid,
        "question": f"question: {question} options: [{a},{b},{c},{d},{e},], answer: ",
        "answer": [str(answer)],
        "dataset": 'vqa',
        "image": f"dataset/test-images/{img}",
        "qtype":qid2type[iid],
        "objs":objs,
    }

    val.append(meta)

print("test file: save to ckpts/test.json")
json.dump(val, open('ckpts/test.json','w'))
print("====================================================================================================")