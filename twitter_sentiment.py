from transformers import DistilBertForSequenceClassification, get_linear_schedule_with_warmup, DistilBertTokenizer
import pandas as pd
# from torch import DataLoader, RandomSampler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
import torch
import random
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score


class twitter_sentiment():
    def __init__(self, model_path=None) -> None:
        #load pre-trained BERT
        if model_path:
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path,
                                                                num_labels = 3,
                                                                output_attentions = False,
                                                                output_hidden_states = False)
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                num_labels = 3,
                                                                output_attentions = False,
                                                                output_hidden_states = False)
        #load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',
                                                do_lower_case = True)
        
        seed_val = 17
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
            
    
    def train(self, train_data):
        dataset_train, dataset_val, y_train, y_val = self.preprocess(train_data)
        
        epochs = 5
        batch_size = 32

        #load optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(),
                        lr = 1e-5,
                        eps = 1e-8) #2e-5 > 5e-5
    
        #load train set
        dataloader_train = DataLoader(dataset_train,
                                    sampler = RandomSampler(dataset_train),
                                    batch_size = batch_size)

        #load val set
        dataloader_val = DataLoader(dataset_val,
                                    sampler = RandomSampler(dataset_val),
                                    batch_size = 32) #since we don't have to do backpropagation for this step
        
        #load scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = len(dataloader_train)*epochs)
        
        for epoch in tqdm(range(1, epochs+1)):

            #set model in train mode
            self.model.train()

            #tracking variable
            loss_train_total = 0
            
            #set up progress bar
            progress_bar = tqdm(dataloader_train, 
                                desc='Epoch {:1d}'.format(epoch), 
                                leave=False, 
                                disable=False)
            
            for batch in progress_bar:
                #set gradient to 0
                self.model.zero_grad()

                #load into GPU
                batch = tuple(b.to(self.device) for b in batch)

                #define inputs
                inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'labels': batch[2]}
                
                outputs = self.model(**inputs)
                loss = outputs[0] #output.loss
                loss_train_total +=loss.item()

                #backward pass to get gradients
                loss.backward()
                
                #clip the norm of the gradients to 1.0 to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                #update optimizer
                optimizer.step()

                #update scheduler
                scheduler.step()
                
                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})     
            
            tqdm.write('\nEpoch {epoch}')
            
            #print training result
            loss_train_avg = loss_train_total/len(dataloader_train)
            tqdm.write(f'Training loss: {loss_train_avg}')
            
            #evaluate
            val_loss, predictions, true_vals = self.evaluate(dataloader_val)
            #f1 score
            val_f1 = self.f1_score_func(predictions, true_vals)
            tqdm.write(f'Validation loss: {val_loss}')
            tqdm.write(f'F1 Score (weighted): {val_f1}')
            self.accuracy_per_class(predictions, true_vals)

        self.model.save_pretrained("drive/MyDrive/Colab Notebooks/model/bert_v2")


    def evaluate(self, dataloader_eval):
        #evaluation mode disables the dropout layer 
        self.model.eval()
        
        #tracking variables
        loss_val_total = 0
        predictions, true_vals = [], []
        
        for batch in tqdm(dataloader_eval):
            
            #load into GPU
            batch = tuple(b.to(self.device) for b in batch)
            
            #define inputs
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2]}

            #compute logits
            with torch.no_grad():        
                outputs = self.model(**inputs)
            
            #compute loss
            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            #compute accuracy
            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)
        
        #compute average loss
        loss_val_avg = loss_val_total/len(dataloader_eval) 
        
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
                
        return loss_val_avg, predictions, true_vals
      
    def predict(self, predict_data):
        encoded_data_predict = self.tokenizer.batch_encode_plus(predict_data.text.values,
                                                        add_special_tokens = True,
                                                        return_attention_mask = True,
                                                        padding = True,
                                                        return_tensors = 'pt')
        #encode train set
        input_ids_predict = encoded_data_predict['input_ids']
        attention_masks_predict = encoded_data_predict['attention_mask']

        dataset_predict = TensorDataset(input_ids_predict, 
                                    attention_masks_predict,)
        
        self.model.eval()

        dataloader_pred = DataLoader(dataset_predict,
                                    sampler = RandomSampler(dataset_predict),
                                    batch_size = 32) #since we don't have to do backpropagation for this step

        #tracking variables
        loss_val_total = 0
        predictions, true_vals = [], []
        
        for batch in dataloader_pred:
            
            #load into GPU
            batch = tuple(b.to(self.device) for b in batch)
            
            #define inputs
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1]}

            #compute logits
            with torch.no_grad():        
                outputs = self.model(**inputs)

            #compute loss
            logits = outputs["logits"]

            #compute accuracy
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)

        predictions = np.concatenate(predictions, axis=0)

        return np.argmax(predictions, axis=1).flatten()

    def f1_score_func(self, preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average = 'weighted')
    
    def accuracy_per_class(self, preds, labels):
        # negative(-1), neutral(0), and positive(+1)
        label_dict_inverse = {0:"negative", 1:"neutral", 2:"positive"}
        
        #make prediction
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
    
        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat==label]
            y_true = labels_flat[labels_flat==label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')
            print(f'Accuracy_value:{len(y_preds[y_preds==label]) / len(y_true)}\n')
    


    def preprocess(self, train_data):
        #train test split
        X_train, X_val, y_train, y_val = train_test_split(train_data.index.values, 
                                                        train_data.label.values,
                                                        test_size = 0.15,
                                                        random_state = 17,
                                                        stratify = train_data.label.values)
        #create new column
        train_data['data_type'] = ['not_set'] * train_data.shape[0]
        #fill in data type
        train_data.loc[X_train, 'data_type'] = 'train'
        train_data.loc[X_val, 'data_type'] = 'val'
        
        #tokenize train set
        encoded_data_train = self.tokenizer.batch_encode_plus(train_data[train_data.data_type == 'train'].text.values,
                                                        add_special_tokens = True,
                                                        return_attention_mask = True,
                                                        padding = True,
                                                        return_tensors = 'pt')
        
        #tokenizer val set
        encoded_data_val = self.tokenizer.batch_encode_plus(train_data[train_data.data_type == 'val'].text.values,
                                                        #add_special_tokens = True,
                                                        return_attention_mask = True,
                                                        padding = True,
                                                        return_tensors = 'pt')
        
        #encode train set
        input_ids_train = encoded_data_train['input_ids']
        attention_masks_train = encoded_data_train['attention_mask']
        labels_train = torch.tensor(train_data[train_data.data_type == 'train'].label.values)

        #encode val set
        input_ids_val = encoded_data_val['input_ids']
        attention_masks_val = encoded_data_val['attention_mask']

        #convert data type to torch.tensor
        labels_val = torch.tensor(train_data[train_data.data_type == 'val'].label.values)

        #create dataloader
        dataset_train = TensorDataset(input_ids_train, 
                                    attention_masks_train,
                                    labels_train)

        dataset_val = TensorDataset(input_ids_val, 
                                    attention_masks_val, 
                                    labels_val)
        
        return dataset_train, dataset_val, y_train, y_val

    def load_data(self, data_location, names):
        return pd.read_csv(data_location, names=names).dropna()
    

if __name__ == "__main__":
    trainer = twitter_sentiment()
    data = trainer.load_data("Twitter_Data.csv", ["text", "category"])
    data = data.astype({'category':int})
    label_dict = {0: 1, 1: 2, -1: 0}

    #convert labels into numeric values
    data['label'] = data.category.replace(label_dict)
    trainer.train(data)