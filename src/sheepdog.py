import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import argparse
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from utils.load_data import *
import warnings
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='politifact', type=str)
parser.add_argument('--model_name', default='Pretrained-LM', type=str)
parser.add_argument('--iters', default=2, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--n_epochs', default=5, type=int)
args = parser.parse_args()
device = torch.device("cuda")

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(0)
        

class NewsDatasetAug(Dataset):
    def __init__(self, texts, aug_texts1, aug_texts2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len):
        self.texts = texts
        self.aug_texts1 = aug_texts1
        self.aug_texts2 = aug_texts2
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        self.fg_label = fg_label
        self.aug_fg1 = aug_fg1
        self.aug_fg2 = aug_fg2

    def __getitem__(self, item):
        text = self.texts[item]
        aug_text1 = self.aug_texts1[item]
        aug_text2 = self.aug_texts2[item]
        label = self.labels[item]
        fg_label = self.fg_label[item]
        aug_fg1 = self.aug_fg1[item]
        aug_fg2 = self.aug_fg2[item]
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                pad_to_max_length=True, truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')

        aug1_encoding = self.tokenizer.encode_plus(aug_text1, add_special_tokens=True, max_length=self.max_len,
                pad_to_max_length=True, truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')

        aug2_encoding = self.tokenizer.encode_plus(aug_text2, add_special_tokens=True, max_length=self.max_len,
                pad_to_max_length=True, truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'input_ids_aug1': aug1_encoding['input_ids'].flatten(),
            'input_ids_aug2': aug2_encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'attention_mask_aug1': aug1_encoding['attention_mask'].flatten(),
            'attention_mask_aug2': aug1_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'fg_label': torch.FloatTensor(fg_label),
            'fg_label_aug1': torch.FloatTensor(aug_fg1),
            'fg_label_aug2': torch.FloatTensor(aug_fg2),
        }

    def __len__(self):
        return len(self.texts)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                pad_to_max_length=True, truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')

        return {
            'news_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)


class RobertaClassifier(nn.Module):
    def __init__(self, n_classes):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(p = 0.5)
        self.fc_out = nn.Linear(self.roberta.config.hidden_size, n_classes)
        self.binary_transform = nn.Linear(self.roberta.config.hidden_size, 2)


    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_outputs = outputs[1]
        pooled_outputs = self.dropout(pooled_outputs)
        output = self.fc_out(pooled_outputs)
        binary_output = self.binary_transform(pooled_outputs)

        return output, binary_output




def create_train_loader(contents, contents_aug1, contents_aug2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len, batch_size):
    ds = NewsDatasetAug(texts = contents, aug_texts1 = contents_aug1, aug_texts2 = contents_aug2, labels = np.array(labels), \
                        fg_label = fg_label, aug_fg1 = aug_fg1, aug_fg2 = aug_fg2, tokenizer=tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)

def create_eval_loader(contents, labels, tokenizer, max_len, batch_size):
    ds = NewsDataset(texts = contents, labels = np.array(labels), tokenizer=tokenizer, max_len=max_len)
    
    return DataLoader(ds, batch_size=batch_size, num_workers=0)



def set_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def train_model(tokenizer, max_len, n_epochs, batch_size, datasetname, iter):

    x_train, x_test, x_test_res, y_train, y_test = load_articles(datasetname)
    test_loader = create_eval_loader(x_test, y_test, tokenizer, max_len, batch_size)
    test_loader_res = create_eval_loader(x_test_res, y_test, tokenizer, max_len, batch_size)

    model = RobertaClassifier(n_classes = 4).to(device)
    train_losses = []
    train_accs = []
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = 10000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(n_epochs):
        model.train()
        x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t = load_reframing(args.dataset_name)
        train_loader = create_train_loader(x_train, x_train_res1, x_train_res2, y_train, y_train_fg, y_train_fg_m, y_train_fg_t, tokenizer, max_len, batch_size)

        avg_loss = []
        avg_acc = []
        batch_idx = 0

        for Batch_data in tqdm(train_loader):

            input_ids = Batch_data["input_ids"].to(device)
            attention_mask = Batch_data["attention_mask"].to(device)
            input_ids_aug1 = Batch_data["input_ids_aug1"].to(device)
            attention_mask_aug1 = Batch_data["attention_mask_aug1"].to(device)
            input_ids_aug2 = Batch_data["input_ids_aug2"].to(device)
            attention_mask_aug2 = Batch_data["attention_mask_aug2"].to(device)
            targets = Batch_data["labels"].to(device)
            fg_labels = Batch_data["fg_label"].to(device)
            fg_labels_aug1 = Batch_data["fg_label_aug1"].to(device)
            fg_labels_aug2 = Batch_data["fg_label_aug2"].to(device)
            
            out_labels, out_labels_bi = model(input_ids=input_ids, attention_mask=attention_mask)
            out_labels_aug1, out_labels_bi_aug1  = model(input_ids=input_ids_aug1, attention_mask=attention_mask_aug1)
            out_labels_aug2, out_labels_bi_aug2 = model(input_ids=input_ids_aug2, attention_mask=attention_mask_aug2)
            fg_criterion = nn.BCELoss()
            finegrain_loss = (fg_criterion(F.sigmoid(out_labels),fg_labels) + fg_criterion(F.sigmoid(out_labels_aug1),fg_labels_aug1) + \
                               fg_criterion(F.sigmoid(out_labels_aug2),fg_labels_aug2)) / 3

            out_probs = F.softmax(out_labels_bi, dim = -1)
            aug_log_prob1 = F.log_softmax(out_labels_bi_aug1, dim = -1)
            aug_log_prob2 = F.log_softmax(out_labels_bi_aug2, dim = -1)

            sup_criterion = nn.CrossEntropyLoss()
            sup_loss = sup_criterion(out_labels_bi, targets)

            cons_criterion = nn.KLDivLoss(reduction = 'batchmean')
            cons_loss = 0.5 * cons_criterion(aug_log_prob1, out_probs) + 0.5 * cons_criterion(aug_log_prob2, out_probs)
       
            loss = sup_loss + cons_loss + finegrain_loss

            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            scheduler.step()
            _, pred = out_labels_bi.max(dim=-1)
            correct = pred.eq(targets).sum().item()
            train_acc = correct / len(targets)
            avg_acc.append(train_acc)
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))


        print("Iter {:03d} | Epoch {:05d} | Train Acc. {:.4f}".format(iter, epoch, train_acc))

        if epoch == n_epochs - 1:
            model.eval()
            y_pred = []
            y_pred_res = []
            y_test = []

            for Batch_data in tqdm(test_loader):
                with torch.no_grad():
                    input_ids = Batch_data["input_ids"].to(device)
                    attention_mask = Batch_data["attention_mask"].to(device)
                    targets = Batch_data["labels"].to(device)
                    _, val_out = model(input_ids=input_ids, attention_mask=attention_mask)
                    _, val_pred = val_out.max(dim=1)

                    y_pred.append(val_pred)
                    y_test.append(targets)

            for Batch_data in tqdm(test_loader_res):
                with torch.no_grad():
                    input_ids_aug = Batch_data["input_ids"].to(device)
                    attention_mask_aug = Batch_data["attention_mask"].to(device)
                    _, val_out_aug = model(input_ids=input_ids_aug, attention_mask=attention_mask_aug)
                    _, val_pred_aug = val_out_aug.max(dim=1)
                    y_pred_res.append(val_pred_aug)


            y_pred = torch.cat(y_pred, dim=0)
            y_test = torch.cat(y_test, dim=0)
            y_pred_res = torch.cat(y_pred_res, dim=0)

            acc = accuracy_score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            precision, recall, fscore, _ = score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), average='macro')


            acc_res = accuracy_score(y_test.detach().cpu().numpy(), y_pred_res.detach().cpu().numpy())
            precision_res, recall_res, fscore_res, _ = score(y_test.detach().cpu().numpy(), y_pred_res.detach().cpu().numpy(), average='macro')


    torch.save(model.state_dict(), 'checkpoints/' + datasetname + '_iter' + str(iter) + '.m')

    print("-----------------End of Iter {:03d}-----------------".format(iter))
    print(['Global Test Accuracy:{:.4f}'.format(acc),
        'Precision:{:.4f}'.format(precision),
        'Recall:{:.4f}'.format(recall),
        'F1:{:.4f}'.format(fscore)])

    print("-----------------Restyle-----------------")
    print(['Global Test Accuracy:{:.4f}'.format(acc_res),
        'Precision:{:.4f}'.format(precision_res),
        'Recall:{:.4f}'.format(recall_res),
        'F1:{:.4f}'.format(fscore_res)])
    
    return acc, precision, recall, fscore, acc_res, precision_res, recall_res, fscore_res


datasetname=args.dataset_name
batch_size = args.batch_size
max_len = 512
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
n_epochs = args.n_epochs
iterations=args.iters

test_accs = []
prec_all, rec_all, f1_all = [], [], []
test_accs_res = []
prec_all_res, rec_all_res, f1_all_res = [], [], []
f1_r_all, f1_f_all, f1_r_all_res, f1_f_all_res = [], [], [], []


for iter in range(iterations):
    set_seed(iter)
    acc, prec, recall, f1, \
    acc_res, prec_res, recall_res, f1_res = train_model(tokenizer,
                                                max_len,
                                                n_epochs,
                                                batch_size,
                                                datasetname,
                                                iter)

    test_accs.append(acc)
    prec_all.append(prec)
    rec_all.append(recall)
    f1_all.append(f1)
    test_accs_res.append(acc_res)
    prec_all_res.append(prec_res)
    rec_all_res.append(recall_res)
    f1_all_res.append(f1_res)

print("Total_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
    sum(test_accs) / iterations, sum(prec_all) /iterations, sum(rec_all) /iterations, sum(f1_all) / iterations))

print("Restyle_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
    sum(test_accs_res) / iterations, sum(prec_all_res) /iterations, sum(rec_all_res) /iterations, sum(f1_all_res) / iterations))


with open('logs/log_' +  datasetname + '_' + args.model_name + '.' + 'iter' + str(iterations), 'a+') as f:
    f.write('-------------Original-------------\n')
    f.write('All Acc.s:{}\n'.format(test_accs))
    f.write('All Prec.s:{}\n'.format(prec_all))
    f.write('All Rec.s:{}\n'.format(rec_all))
    f.write('All F1.s:{}\n'.format(f1_all))
    f.write('Average acc.: {} \n'.format(sum(test_accs) / iterations))
    f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all) /iterations, sum(rec_all) /iterations, sum(f1_all) / iterations))
    f.write('Average F1 by class (Real / Fake): {}, {} \n'.format(sum(f1_r_all) / iterations, sum(f1_f_all) / iterations))

    f.write('\n-------------Adversarial------------\n')
    f.write('All Acc.s:{}\n'.format(test_accs_res))
    f.write('All Prec.s:{}\n'.format(prec_all_res))
    f.write('All Rec.s:{}\n'.format(rec_all_res))
    f.write('All F1.s:{}\n'.format(f1_all_res))    
    f.write('Average acc.: {} \n'.format(sum(test_accs_res) / iterations))
    f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all_res) /iterations, sum(rec_all_res) /iterations, sum(f1_all_res) / iterations))
    f.write('Average F1 by class (Real / Fake): {}, {} \n'.format(sum(f1_r_all_res) / iterations, sum(f1_f_all_res) / iterations))
