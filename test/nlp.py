import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertEmbeddings

import os

#device = torch.device("cuda:0")
bertmodel, vocab = get_pytorch_kobert_model()

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


## Setting parameters.
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

#model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
model = BERTClassifier(bertmodel,  dr_rate=0.5)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


os.chdir('./models/')

model1 = torch.load('7emotions_model.pt')                                       # 전체 모델을 통째로 불러옴, 클래스 선언 필수.
model1.load_state_dict(torch.load('7emotions_model_state_dict.pt'))             # state_dict를 불러 온 후, 모델에 저장.

checkpoint = torch.load('7emotions_all.tar')                                    # dict 불러오기.
model1.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def split_text(non_splited_text):
    splited_text = non_splited_text.split('.')                                    # .(마침표)단위로 끊어낸 뒤 자동으로 리스트 형 변환.
    return splited_text


def predict(predict_sentence):                                                  #참고 : predict_sentence는 \n 없이 입력받아야 합니다.

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model1.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        #token_ids = token_ids.long().to(device)
        token_ids = token_ids.long()

        #segment_ids = segment_ids.long().to(device)
        segment_ids = segment_ids.long()

        valid_length= valid_length
        #label = label.long().to(device)
        label = label.long()

        out = model1(token_ids, valid_length, segment_ids)


        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                eval = '공포'
            elif np.argmax(logits) == 1:
                eval = '놀람'
            elif np.argmax(logits) == 2:
                eval = '분노'
            elif np.argmax(logits) == 3:
                eval = '슬픔'
            elif np.argmax(logits) == 4:
                eval = '중립'
            elif np.argmax(logits) == 5:
                eval = '행복'
            elif np.argmax(logits) == 6:
                eval = '혐오'
        return eval


def SolveTheResult(text):                                                       #들어가는 text는 str형이며, \n이 없도록 주의해주세요.
    emotions = {'공포':0,'놀람':0,'분노':0,'슬픔':0,'중립':0,'행복':0,'혐오':0}   #초기값은 모두 0.
    text = text.split('.')                                                        #마침표 단위로 문장을 끊습니다.
    for splited_text in text:
        if predict(splited_text)=='공포':
            emotions['공포']+=1
        elif predict(splited_text)=='놀람':
            emotions['놀람']+=1
        elif predict(splited_text)=='분노':
            emotions['분노']+=1
        elif predict(splited_text)=='슬픔':
            emotions['슬픔']+=1
        elif predict(splited_text)=='중립':
            emotions['중립']+=1
        elif predict(splited_text)=='행복':
            emotions['행복']+=1
        elif predict(splited_text)=='혐오':
            emotions['혐오']+=1

    emotions = sorted(emotions.items(), key = lambda item: item[1], reverse=True) #emotions를 오름차순으로 정렬
                                                                                    #list 내부에 tuple이 있는 형태가 됨 --> tuple을 바꿀 수 없으므로 내부까지 list로 변환.
    emotions = [list(emotions[x]) for x in range(len(emotions))]                  #내부까지 list형으로 변환.

    summary = sum(int(j) for i, j in emotions)                                    #emotions의 모든 감정의 개수

    for i in range(len(emotions)):                                                #emotions의 각각의 감정을 총합과 나누어 %로 나타낸다.
        try:                                                                        #감정의 값이 0일 경우를 위한 예외처리
            emotions[i][1] = emotions[i][1] / summary
        except ZeroDivisionError:
            pass
    emotions = emotions[0:3]                                                      #emotions의 상위 3개값만 출력
    return emotions


test_text = '수요일은 특식 데이이다. 학교 급식에서 잔반 없는 날 이라고 하여 맛있는 급식을 주는 날이다. 바로 오늘이 수요일이라 아침부터 기대가 되었다. 오늘의 반찬은 치킨 샐러드와 스파게티, 옥수수 수프 그리고 고구마튀김이 나와서 친구들 모두 좋아했다. 그중에서 스파게티는 내 짝 정연이가 좋아하는 음식이다. 정연이는 돈가스와 스파게티를 제일 좋아한다. 내가 제일 좋아하는 만두는 없어서 아쉬웠지만, 정연이가 만두가 맛있는 분식집을 알려주어서 아쉬운 마음이 사라졌다. 주말에 엄마 아빠에게 그 분식집에 가보자고 해야지.'

print(SolveTheResult(test_text))