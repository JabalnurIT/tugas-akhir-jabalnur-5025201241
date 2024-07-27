texts = dataset.split("###")
data = pd.Series(texts)

self.tokenizer = tokenizer
self.input_ids = []
self.attn_masks = []

for txt in txt_list:

    encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

    self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
    self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    

dataset = self.input_ids, self.attn_masks

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