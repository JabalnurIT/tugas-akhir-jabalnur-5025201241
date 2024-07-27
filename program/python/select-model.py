learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8
batch_size = 2
epochs = 10
sample_every = 100
seed_val = 42

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

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)