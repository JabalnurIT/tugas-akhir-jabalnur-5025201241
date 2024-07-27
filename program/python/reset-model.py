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