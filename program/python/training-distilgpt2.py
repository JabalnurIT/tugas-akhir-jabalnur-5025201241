for epoch_i in range(0, self.epochs):
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

    avg_train_loss = total_train_loss / len(train_dataloader)