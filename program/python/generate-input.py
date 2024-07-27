if self.current_model == "none":
    return "Error: No model selected. Please select a model first from this list: " + str(self.model_list)
self.model.eval()

prompt = "<|startoftext|>"

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