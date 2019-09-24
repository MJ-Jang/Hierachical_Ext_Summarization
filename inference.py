#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import torch
import json
import numpy as np
import sentencepiece as sp
import torch.nn.functional as F
import torch.nn as nn

from modules import HierSumTransformer
from torch.utils.data import DataLoader, Dataset
from config import ModelConfig as config


class ExtSumDataset_infer(Dataset):
    def __init__(self, data, tok, type):
        self.tok = tok
        self.PAD = self.tok.piece_to_id('[PAD]')
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        document = self.data[idx] #document: list of sentences
        document_len = np.array([len(document)])        
        document_encode = [self.tok.EncodeAsIds(sent) for sent in document]
        sent_len = [len(d) for d in document_encode]
        
        # generage mask
        if len(document_encode) < config.max_doc_len:
            doc_mask = np.array([1]*len(document_encode) + [0]*(config.max_doc_len - len(document_encode)))
        else:
            doc_mask = np.array([1]*config.max_doc_len)

        # process document
        for sentences in document_encode:
            if len(sentences) < config.max_sent_len:
                extended_words = [self.PAD for _ in range(config.max_sent_len - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < config.max_doc_len:
            extended_sentences = [[self.PAD for _ in range(config.max_sent_len)] for _ in
                                  range(config.max_doc_len - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:config.max_sent_len] for sentences in document_encode][:config.max_doc_len]
        document_encode = np.stack(arrays=document_encode, axis=0)
        
        # process sentence len
        if len(sent_len) < config.max_doc_len:
            sent_len += [0]*(config.max_doc_len - len(sent_len))
        else:
            sent_len = sent_len[:config.max_doc_len]
        sent_len = np.array(sent_len)
        
        return [document_encode, document_len, sent_len, doc_mask]
    
    
class Inferencer:

    def __init__(self):
        pass

    def load_model(self):
        """
        Load pre-trained models from binary
        """
        self.tok = sp.SentencePieceProcessor()
        self.tok.Load(config.tok_path)
        self.vocab = self.tok.GetPieceSize()
        self.pad = self.tok.piece_to_id('[PAD]')

        self.model = HierSumTransformer(self.vocab, config.emb_dim, config.d_model, config.N, config.heads, config.max_sent_len, config.max_doc_len)

        self.model.load_state_dict(torch.load(config.model_path, map_location=lambda storage, location: storage))
        self.model.eval()

    def infer(self, documents, batch_size=64):

        loader = self._prepare_batch(documents, batch_size)

        for batch_id, batch in enumerate(loader):
            doc_id, doc_len, sent_len, doc_mask = batch
            doc_mask = doc_mask.unsqueeze(1)
            sent_mask = torch.stack([self._create_mask(sent) for sent in doc_id])                         
            
            preds = self.model(doc_id, sent_mask, doc_mask, sent_len)
            probs = F.softmax(preds, dim=2)
            pred_label = [torch.argmax(p, 1).tolist() for p in preds]
        
        for i in range(len(pred_label)):
            pred_label[i] = pred_label[i][:doc_len[i].tolist()[0]]
        
        result = []
        for i in range(len(documents)):
            tmp_result = [documents[i][j] for j in range(len(documents[i])) if pred_label[i][j] == 1]
            result.append(tmp_result)
        
        return result

    def _prepare_batch(self, documents, batch_size=64):
        dataset = ExtSumDataset_infer(documents, self.tok, type)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, )
        return loader
    
    def _create_mask(self, tok_ids):
        mask = (tok_ids != self.pad).unsqueeze(1)
        return mask



if __name__ == '__main__':
    inferencer = Inferencer()
    inferencer.load_model()

    src =  [["정부의 강력한 부동산 규제의 후폭풍이 고위공직자의 잇따른 낙마로 이어져 눈길을 끈다.", "정부의 강력한 부동산 규제가 부메랑으로 돌아왔다는 의견이 나온다.", "최근 장관 후보자가 다주택으로 인한 구설수로 인해 낙마하는 초유의 사태가 발생했다.", "31일 오전 최정호 국토교통부 장관 후보자는 ‘다주택’에 대한 논란으로 스스로 후보에서 물러났다.", "최 후보자는 이날 국토부 출입기자들에게 이메일을 보내 “국토부 장관 후보자에서 사퇴한다.", "성원해주신 모든 분들께 깊이 감사드린다”라고 밝혀 사진사퇴한다는 사실을 알렸다.", "국교부 장관 후보자에 지명된 후 최 후보자는 경기도 분당과 서울 잠실에 아파트를 보유한 데다가 세종시에 아파트 분양권을 가지고 있는 3주택자인 것이 알려져 대중들의 입길에 올랐다.", "2003년 전세를 안고 3억원에 구입한 잠실 엘스59㎡ 아파트는 현재 13억원을 호가하고 세종시 반곡동에 공무원 특별공급으로 6억원에 분양받은 ‘캐슬파밀리에 디아트’ 팬트하우스155㎡는 인근 시세와 연동하면 약 13억원 정도로 시세차익이 7억원 정도가 발생했다.", "20여년 거주 중인 분당 정자동 상록마을라이프2단지84㎡ 아파트는 장관 후보자 지명 직전 딸 부부에 증여하고 보증금 3000만원 월세 160만원에 월세로 거주하고 있다.", "네티즌들 사이에서는 현 김현미 국교부 장관이 문 정부 출범 이후 줄곧 “살 집 아니면 파시라”고 주장해왔는데 새 국토부 장관이 다주택자라는 것은 말이 되지 않는다는 지적이 쏟아져나왔다.", "특히 최 후보자가 국토부 장관 후보 지명을 앞둔 시점에서 분당 아파트를 딸 부부에게 증여해 한 채를 줄였지만 편법 증여가 아니냐는 논란까지 일었다.", "부동산으로 인한 사퇴는 또 있다.", "앞서 29일 김의겸 청와대 대변인이 흑석동 재개발 상가 건물 매입과 관련해 논란이 불거지자 사직서를 제출했다.", "김 전 대변인은 대변인이 되고 난 직후인 지난해 흑석뉴타운 9구역 내 25억원 짜리 상가건물을 11억원 대출을 받아 구입한 사실이 알려지며 논란을 빚었다.", "이 건물은 재개발이 되면 아파트 2채와 상가 1채를 받을 수 있는 물건이었다.", "김 전 대변인이 30년 동안 전세를 살다가 노모를 모시기 위해 해당 건물을 샀다고 해명했지만 국민들의 반응은 냉담했다.", "노모를 모시기 위해서라면 이미 완공돼있는 아파트를 사야지 언제 입주할지 모르는 재개발 건물을 산 것은 말이 되지 않는다는 의견이 많았다.", "게다가 정부가 “빚 내서 집 사지 말라”고 경고했는데 김 전 대변인이 과도한 빚을 냈다는 사실도 아이러니로 꼽혔다.", "“문 대통령의 입口인 대변인이 입으로는 부동산 투기 금지를 말하면서 정작 본인은 대출을 최대한 당겨 투기를 했다”는 질타에 부담을 느낀 김 전 대변인은 사직서를 제출했다.", "문재인 정부는 출범 이후 꾸준히 ‘부동산 투기와 전쟁’을 선포하고 다주택자들에게 집을 팔 것을 강권했다.", "그러나 정작 고위공직자들은 다주택은 물론 보유한 주택으로 인해 큰 시세차익 등을 봤다는 사실이 드러나면서 정부의 강력 부동산 대책에 대한 반발이 커지고 있다."], ["애플이 국내 시장 진출 20여년만에 처음으로 고용 인원을 밝혔다.", "애플의 한국지사에 직접 고용 인원은 500명이며, 국내에서 창출한 일자리는 32만5000개라고 밝혔다.", "애플은 19일 애플코리아 공식 홈페이지에 고용 창출 페이지를 개설하고, 6월 30일 현재 한국 지사에 직접 고용된 직원 수가 500명이라고 밝혔다.", "애플이 한국 지사의 고용 인원을 직접 밝힌 건 처음이다.", "애플은 “20여년 전 단 2명의 직원으로 시작했지만, 현재 디자이너, 제작 전문가, 리테일 직원, 고객 서비스 담당자, 하드웨어 및 소프트웨어 엔지니어 등 500여명이 근무하고 있다”며 “이 수치는 빠른 속도로 증가하고 있다”고 밝혔다.", "2010년 이후 직원 증가율은 1500%다. 지난해 국내에 처음으로 문을 연 애플스토어의 개장 준비를 위해 2017년 직원을 급격히 늘렸기 때문이다.", "애플은 직접 고용이 아니더라도 국내에서 창출한 일자리 수가 32만5000개라고 밝혔다.", "국내 부품사 등 협력업체를 통해 12만5000개, 앱 스토어 생태계를 통해 20만개라는 것이다.", "그러면서 이는 “시작에 불과하다”고 말했다.", "32만5000개라는 숫자는 컨설팅 업체인 애널리시스 그룹이 2018년 애플이 한국에서 상품 및 서비스에 지출한 투자총액 정보를 토대로, 직간접적으로 창출된 일자리 수를 산출한 것다.", "구체적인 국내 협력업체 수와 사례도 공개했다.", "애플은 국내 협력업체 200여개사와 일하고 있으며 제조 6만명, 도매 및 소매·차량 수리 2만명, 전문·과학 및 기술 활동 1만명, 행정 및 지원 서비스 활동 8000명 등의 일자리가 창출됐다고 밝혔다.", "예를 들어 포스코와 2016년부터 초청정 비자성 스테인리스를 만들기 위해 의기투합했고, 새로운 소재를 개발해 아이폰X부터 제품에 도입했다고 설명했다.", " 아이폰에 들어가는 스테인리스를 얇게 펴고 표면을 정밀하게 가공하는 풍산, 경연성인쇄회로기판 제조하는 영풍전자, 애플 카메라의 성능 및 안전성을 테스트하는 하이비젼시스템 등도 소개됐다.", "애플코리아는 또 2008년 이후 앱 스토어를 통해 한국 개발자들이 전 세계적으로 번 수익은 4조7000억원이고 작년 기준 관련 일자리 수도 20만개에 달한다고 말했다."]]
    res = inferencer.infer(src)
    print(res)
