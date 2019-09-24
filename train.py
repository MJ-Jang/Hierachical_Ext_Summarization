import torch
import json
import numpy as np
import sentencepiece as sp
import torch.nn.functional as F
import torch.nn as nn

from modules import HierSumTransformer
from torch.utils.data import DataLoader, Dataset
from config import ModelConfig as config
import multiprocessing

from utils import RAdam


class ExtSumDataset(Dataset):
    def __init__(self, data_path, tok, type):
        self.tok = tok
        self.PAD = self.tok.piece_to_id('[PAD]')

        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if type == 'train':
            self.data = data['train']
        elif type == 'dev':
            self.data = data['dev']
        else:
            raise ValueError("type should be one of 'train' or 'dev'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc_sample = self.data[idx] #document: list of sentences
        document = doc_sample['document']
        label = doc_sample['ext_target']
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
        
        # process label
        if len(label) < config.max_doc_len:
            label += [config.ignore_index_ext]*(config.max_doc_len - len(label))
        else:
            label = label[:config.max_doc_len]
        label = np.array(label)

        # process sentence len
        if len(sent_len) < config.max_doc_len:
            sent_len += [0]*(config.max_doc_len - len(sent_len))
        else:
            sent_len = sent_len[:config.max_doc_len]
        sent_len = np.array(sent_len)
        
        return [document_encode, document_len, sent_len, label, doc_mask]


class ExtSumDataset_old(Dataset):
    def __init__(self, data_path, tok, type):
        self.tok = tok
        self.PAD = self.tok.piece_to_id('[PAD]')

        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if type == 'train':
            self.data = data['train']
        elif type == 'dev':
            self.data = data['dev']
        else:
            raise ValueError("type should be one of 'train' or 'dev'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc_sample = self.data[idx] #document: list of sentences
        document = doc_sample['document']
        label = doc_sample['ext_target']
        document_len = np.array([len(document)])
        document_encode = [self.tok.EncodeAsIds(sent) for sent in document]

        # generage mask
        if len(document_encode) < config.max_sent_len:
            mask = np.array([1]*len(document_encode) + [0]*(config.max_sent_len - len(document_encode)))
        else:
            mask = np.array([1]*config.max_sent_len)

        # process document
        for sentences in document_encode:
            if len(sentences) < config.max_word_len:
                extended_words = [self.PAD for _ in range(config.max_word_len - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < config.max_sent_len:
            extended_sentences = [[self.PAD for _ in range(config.max_word_len)] for _ in
                                  range(config.max_sent_len - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:config.max_word_len] for sentences in document_encode][:config.max_sent_len]
        document_encode = np.stack(arrays=document_encode, axis=0)

        # process label
        if len(label) < config.max_sent_len:
            label += [config.ignore_index_ext]*(config.max_sent_len - len(label))
        else:
            label = label[:config.max_sent_len]
        label = np.array(label)

        return [document_encode, document_len, label, mask]

    
class TrainOperator:
    def __init__(self):
        # source
        self.tok = sp.SentencePieceProcessor()
        self.tok.Load(config.tok_path)
        self.vocab = self.tok.GetPieceSize()
        self.pad = self.tok.piece_to_id('[PAD]')
        self.num_workers = multiprocessing.cpu_count()

        self.cuda = config.cuda and torch.cuda.is_available()

        # for data parallel
        if self.cuda:
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0

        # load loader
        self.train_loader = self._construct_loader('train')
        self.dev_loader = self._construct_loader('dev')
        print('* Train Operator is loaded')

    def setup_train(self, model_path=None):
        self.loss_weight = torch.FloatTensor([1-config.alpha, config.alpha])
        if self.cuda:
            self.loss_weight = self.loss_weight.cuda()
        
        self.model = HierSumTransformer(self.vocab, config.emb_dim, config.d_model, config.N, config.heads, config.max_sent_len, config.max_doc_len)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, location: storage))
        else:
            for p in self.model.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
        # Data Parallel
        if self.cuda:
            if self.n_gpu == 1:
                pass
            elif self.n_gpu > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

        #self.optim = torch.optim.Adam(self.model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
        self.optim = RAdam(self.model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)

        print('* Training model is prepared')

    def train(self):
        printInterval = 20
        init_loss = 1e5
        init_f1 = 0
        
        
        for n in range(config.n_epoch):
            loss_tr_total = 0
            for batch_id, batch in enumerate(self.train_loader):
                loss_tr = self._train_one_batch(batch)
                loss_tr_total += loss_tr
                if (batch_id + 1) % printInterval == 0 or batch_id == 0:
                    loss_tr = round(loss_tr, 4)
                    loss_eval, recall, pre, f1 = [round(l, 4) for l in self._evaluate()]
                    print("| epoch: {} | batch: {}/{}| tr_loss: {} | val_loss: {} |".format(n + 1,
                                                                                            batch_id + 1,
                                                                                            len(self.train_loader),
                                                                                            round(loss_tr_total / (batch_id+1), 4),
                                                                                            loss_eval))
                    print("| epoch: {} |Recall: {} | Precision: {} | F1: {} |".format(n + 1, recall, pre, f1))
                    print("-" * 100)

                    if loss_eval < init_loss:
                        init_loss = loss_eval

                        if self.n_gpu <= 1:
                            torch.save(self.model.state_dict(), './resource/RNN_TR_HiSum_v2.0.pkl')
                        elif self.n_gpu > 1:
                            torch.save(self.model.module.state_dict(), './resource/RNN_TR_HiSum_v2.0.pkl')

                    # change model state to train
                    self.model.train()

    def _construct_loader(self, type):
        dataset = ExtSumDataset(config.data_path, self.tok, type)
        loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0, )
        return loader

    def _train_one_batch(self, batch):
        doc_id, doc_len, sent_len, label, doc_mask = batch
        doc_mask = doc_mask.unsqueeze(1)
        sent_mask = torch.stack([self._create_mask(sent) for sent in doc_id])
       
        if self.cuda:
            doc_id = doc_id.cuda()
            doc_len = doc_len.cuda()
            doc_mask = doc_mask.cuda()
            sent_mask = sent_mask.cuda()
            sent_len = sent_len.cuda()
            label = label.cuda()
        
        preds = self.model(doc_id, sent_mask, doc_mask, sent_len)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), label.reshape(-1), ignore_index=config.ignore_index_ext, weight=self.loss_weight)
        #loss = focal_loss(preds.view(-1, preds.size(-1)), label.reshape(-1), ignore_index=config.ignore_index_ext,  alpha = config.alpha, gamma = config.gamma)

        loss.backward()
        self.optim.step()
        return loss.tolist()

    def _evaluate(self):
        right = 0
        origin = 0
        found = 0
        total_loss = 0
        self.model.eval()

        for i, data in enumerate(self.dev_loader):
            doc_id, doc_len, sent_len, label, doc_mask = data
            doc_mask = doc_mask.unsqueeze(1)
            sent_mask = torch.stack([self._create_mask(sent) for sent in doc_id])                         
        
            if self.cuda:
                doc_id = doc_id.cuda()
                doc_len = doc_len.cuda()
                doc_mask = doc_mask.cuda()
                sent_mask = sent_mask.cuda()
                sent_len = sent_len.cuda()
                label = label.cuda()
        
            preds = self.model(doc_id, sent_mask, doc_mask, sent_len)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), label.reshape(-1), ignore_index=config.ignore_index_ext, weight=self.loss_weight)
            #loss = focal_loss(preds.view(-1, preds.size(-1)), label.reshape(-1), ignore_index=config.ignore_index_ext, alpha = config.alpha, gamma = config.gamma)

            total_loss += loss.tolist()

            pred_label = [torch.argmax(p, 1).tolist() for p in preds]
            labels = label.tolist()

            for p_tag, label in zip(pred_label, labels):
                for p, l in zip(p_tag, label):
                    if l == config.ignore_index_ext:
                        break
                    elif p == 1 and l == 1:
                        right += 1
                        origin += 1
                        found += 1
                    elif p == 0 and l == 1:
                        origin += 1
                    elif p == 1 and l == 0:
                        found += 1
                    else:
                        pass

        recall = (right / (origin + 1e-5))
        precision = (right / (found + 1e-5))
        f1 = (2 * precision * recall) / (precision + recall + 1e-5)
        return round(total_loss/(i+1), 4), round(recall,4), round(precision,4), round(f1,4)
                                 
    def _create_mask(self, tok_ids):
        mask = (tok_ids != self.pad).unsqueeze(1)
        return mask


    
def focal_loss(input, targets, ignore_index=None, alpha=0.1, gamma=2, reduce=True):
    BCE_loss = F.cross_entropy(input, targets, ignore_index=ignore_index, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    if reduce:
        return torch.sum(F_loss)
    else:
        return F_loss


if __name__ == '__main__':
    trainer = TrainOperator()
    trainer.setup_train(model_path='./resource/RNN_TR_HiSum_v2.0.pkl')
    trainer.train()
