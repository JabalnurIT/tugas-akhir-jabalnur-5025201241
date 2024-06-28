import json

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

import os
import time
import datetime
import math
import random
import numpy as np
import random
import shutil
import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import DefaultFlowCallback, AdamW, get_linear_schedule_with_warmup

from .gpt2_dataset import GPT2Dataset

from tqdm import tqdm

import locale
locale.getpreferredencoding = lambda: "UTF-8"



with open("config.json") as json_file:
    config = json.load(json_file)


def check_dir(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return False
    else:
        return True

def get_dir_list(directory):
    # Check if the directory exists
    if not check_dir(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    # Get list of files and directories in the specified directory
    contents = os.listdir(directory)
    
    return contents

def delete_dir(directory):
    # Check if the directory exists
    if not check_dir(directory):
        print(f"Directory '{directory}' does not exist.")
    
    # Remove the directory with all its contents
    shutil.rmtree(directory)


def is_dir_empty(directory):
    # Check if the directory exists
    if not check_dir(directory):
        print(f"Directory '{directory}' does not exist.")
        return False
    
    # Get list of files and directories in the specified directory
    contents = os.listdir(directory)
    
    # Check if the directory is empty
    if len(contents) == 0:
        return True
    else:
        return False

def make_dir_empty(directory):
    # Check if the directory exists
    if not check_dir(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    # Get list of files and directories in the specified directory
    contents = os.listdir(directory)
    
    # Remove each file or subdirectory within the directory
    for item in contents:
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            os.rmdir(item_path)
        else:
            print(f"Unknown item type: {item_path}")

def copy_files(src, dest):
    # Check if the source directory exists
    if not check_dir(src):
        print(f"Source directory '{src}' does not exist.")
        return
    
    # Check if the destination directory exists
    if not check_dir(dest):
        # make the destination directory
        os.makedirs(dest)
    
    if not os.path.isdir(dest):
        print(f"Destination '{dest}' is not a directory.")
        return

    # Get list of files and directories in the source directory
    contents = os.listdir(src)
    
    # Copy each file or subdirectory from the source directory to the destination directory
    for item in contents:
        src_item_path = os.path.join(src, item)
        dest_item_path = os.path.join(dest, item)
        if os.path.isfile(src_item_path):
            shutil.copy(src_item_path, dest_item_path)
        elif os.path.isdir(src_item_path):
            shutil.copytree(src_item_path, dest_item_path)
        else:
            print(f"Unknown item type: {src_item_path}")

class Model:

    learning_rate = 5e-4
    warmup_steps = 1e2
    epsilon = 1e-8
    batch_size = 2
    epochs = 10
    sample_every = 100

    def __init__(self):

        self.current_model = "none"

        self.model_list = get_dir_list(config["USED_DIR"])
        print(self.model_list)
        
        
    def format_time(self, elapsed):
        return str(datetime.timedelta(seconds=int(round((elapsed)))))

    def get_model_list(self):
        self.model_list = get_dir_list(config["USED_DIR"])

        return self.current_model, self.model_list

    def use_model(self, model):
        try:
            self.model_list = get_dir_list(config["USED_DIR"])    
            self.current_model = model
            
            if model not in self.model_list:
                print(f"Model '{model}' does not exist.")
                copy_files(config["BASE_DIR"], f'{config["USED_DIR"]}{model}')
                
            
            self.tokenizer = GPT2Tokenizer.from_pretrained(config["DISTIL_GPT2_MODEL"],bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

            self.config = GPT2Config.from_pretrained(f'{config["USED_DIR"]}{self.current_model}/config.json', output_hidden_states=False)

            self.model = GPT2LMHeadModel.from_pretrained(f'{config["USED_DIR"]}{self.current_model}/pytorch_model.bin',config=self.config)

            self.model.resize_token_embeddings(len(self.tokenizer))

            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                lr = self.learning_rate,
                eps = self.epsilon
            )

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.model.cuda()
            else:
                self.device = torch.device("cpu")
                self.model.to(self.device)

            seed_val = 42

            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            torch.cuda.manual_seed_all(seed_val)
            status = "Success"
        except Exception as e:
            status = "Error retraining model: " + str(e)
        return status

    def save_finetuned_model(self, output_dir):
        self.model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        self.model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def retrain(self, texts):
        if self.current_model == "none":
            return "Error: No model selected. Please select a model first from this list: " + str(self.model_list)
        texts = texts.split("###")
        data = pd.Series(texts)
        print(data)

        dataset = GPT2Dataset(data, self.tokenizer, max_length=40)

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        
        train_dataloader = DataLoader(
            train_dataset, 
            sampler = RandomSampler(train_dataset),
            batch_size = self.batch_size 
        )

        
        validation_dataloader = DataLoader(
                    val_dataset,
                    sampler = SequentialSampler(val_dataset),
                    batch_size = self.batch_size 
                )

        total_steps = len(train_dataloader) * self.epochs

        scheduler = get_linear_schedule_with_warmup(self.optimizer,
            num_warmup_steps = self.warmup_steps,
            num_training_steps = total_steps
            )
        

        total_t0 = time.time()

        self.model = self.model.to(self.device)


        for epoch_i in range(0, self.epochs):

            # ========================================
            #               Training
            # ========================================

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            t0 = time.time()

            total_train_loss = 0

            self.model.train()

            for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader), desc="Processing")):

                b_input_ids = batch[0].to(self.device)
                b_labels = batch[0].to(self.device)
                b_masks = batch[1].to(self.device)

                self.model.zero_grad()

                outputs = self.model(  
                    b_input_ids,
                    labels=b_labels,
                    attention_mask = b_masks,
                    token_type_ids=None
                )

                loss = outputs[0]

                batch_loss = loss.item()
                total_train_loss += batch_loss

                loss.backward()

                self.optimizer.step()

                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = model.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            t0 = time.time()
            self.model.eval()

            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:

                b_input_ids = batch[0].to(self.device)
                b_labels = batch[0].to(self.device)
                b_masks = batch[1].to(self.device)

                with torch.no_grad():

                    outputs  = self.model(b_input_ids,
                                    attention_mask = b_masks,
                                    labels=b_labels)

                    loss = outputs[0]

                batch_loss = loss.item()
                total_eval_loss += batch_loss

            avg_val_loss = total_eval_loss / len(validation_dataloader)

            validation_time = model.format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(model.format_time(time.time()-total_t0)))

        prompt = "<|startoftext|>"

        generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        generated = generated.to(self.device)

        outputs = self.model.generate(
                        generated,
                        do_sample=True,
                        top_k=50,
                        max_length = 50,
                        top_p=0.95,
                        num_return_sequences=5,
                    )
        text_outputs = []
        for n in outputs:
            text_outputs.append(self.tokenizer.decode(n, skip_special_tokens=True))

        data_text = "###".join(text_outputs)
        print(data_text)
        try:
            model.save_finetuned_model(output_dir=f'{config["USED_DIR"]}{self.current_model}')
            status = "Success"
        except Exception as e:
            status = "Error retraining model: " + str(e)
        return status

    def generate(self, num_text):
        if self.current_model == "none":
            return "Error: No model selected. Please select a model first from this list: " + str(self.model_list)
        self.model.eval()

        prompt = "<|startoftext|>"
        # prompt = "90024"

        generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        generated = generated.to(self.device)

        outputs = self.model.generate(
                        generated,
                        do_sample=True,
                        top_k=50,
                        max_length = 50,
                        top_p=0.95,
                        num_return_sequences=num_text,
                    )
        text_outputs = []
        for n in outputs:
            text_outputs.append(self.tokenizer.decode(n, skip_special_tokens=True))

        data_text = "###".join(text_outputs)
        return data_text
        
    def reset_model(self):
        try:
            if self.current_model == "none":
                return "Error: No model selected. Please select a model first from this list: " + str(self.model_list)
            make_dir_empty(f'{config["USED_DIR"]}{self.current_model}')
            copy_files(config["BASE_DIR"], f'{config["USED_DIR"]}{self.current_model}')
            self.tokenizer = GPT2Tokenizer.from_pretrained(config["DISTIL_GPT2_MODEL"],bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
            self.config = GPT2Config.from_pretrained(f'{config["USED_DIR"]}{self.current_model}/config.json', output_hidden_states=False)
            self.model = GPT2LMHeadModel.from_pretrained(f'{config["USED_DIR"]}{self.current_model}/pytorch_model.bin',config=self.config)
            
            self.model.resize_token_embeddings(len(self.tokenizer))

            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                lr = self.learning_rate,
                eps = self.epsilon
            )

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.model.cuda()
            else:
                self.device = torch.device("cpu")
                self.model.to(self.device)
            status = "Success"
        except Exception as e:
            status = "Error when reseting the model: " + str(e)
        return status

    def delete_model(self):
        try:
            if self.current_model == "none":
                return "Error: No model selected. Please select a model first from this list: " + str(self.model_list)
            delete_dir(f'{config["USED_DIR"]}{self.current_model}')
            self.current_model = "none"
            self.model_list = get_dir_list(config["USED_DIR"])
            status = "Success"
        except Exception as e:
            status = "Error when reseting the model: " + str(e)
        return status


model = Model()


def get_model():
    return model
