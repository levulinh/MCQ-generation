import datasets
import torch
from qa_dataset import SEP_TOKEN, QA_Dataset_squad
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (  # T5ForQuestionAnswering,; T5Tokenizer,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)


def _train_epoch(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    train_batch_count = 0
    for batch in tqdm(train_loader, desc="Training batches"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        optimizer.zero_grad()
        outputs.loss.backward()
        optimizer.step()
        train_loss += outputs.loss.item()
        train_batch_count += 1
    return train_loss / train_batch_count


def _eval_epoch(model, val_loader, device):
    model.eval()
    val_loss = 0
    val_batch_count = 0
    for batch in tqdm(val_loader, desc="Validation batches"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        val_loss += outputs.loss.item()
        val_batch_count += 1
    return outputs, val_loss / val_batch_count


def train(model, train_loader, val_loader, optimizer, device, tokenizer, writer, n_epochs=30):
    train_loss = 0
    val_loss = 0

    for epoch in range(n_epochs):
        # Training
        train_loss = _train_epoch(model, train_loader, optimizer, device)

        # Evaluation
        outputs, val_loss = _eval_epoch(model, val_loader, device)

        # Optimize
        optimizer.zero_grad()
        outputs.loss.backward()
        optimizer.step()

        # Logging the losses
        writer.add_scalar("Train/Train Loss", train_loss, epoch)
        writer.add_scalar("Eval/Eval Loss", val_loss, epoch)
        print(f"{epoch+1}/{n_epochs} -> Train loss: {train_loss}" f"\tValidation loss: {val_loss}")
        if (epoch + 1) % 2 == 0:
            print("Saving the models...")
            model.save_pretrained(f"model/ckpt_qg_squad/{epoch}")
            tokenizer.save_pretrained(f"model/tokenizer_qg_squad/{epoch}")


if __name__ == "__main__":

    writer = SummaryWriter("runs/qg_squad")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    tokenizer.add_tokens(SEP_TOKEN)

    model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True).to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)
    Q_LEN = 256  # Question Length
    T_LEN = 48  # Target Length
    BATCH_SIZE = 64
    N_EPOCHS = 30

    dataset = datasets.load_dataset("squad_v2")
    train_set = dataset["train"]
    val_set = dataset["validation"]

    qa_train_dataset = QA_Dataset_squad(tokenizer, train_set, Q_LEN, T_LEN)
    qa_val_dataset = QA_Dataset_squad(tokenizer, val_set, Q_LEN, T_LEN)
    train_loader = DataLoader(qa_train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(qa_val_dataset, batch_size=BATCH_SIZE)

    train(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        tokenizer=tokenizer,
        writer=writer,
        n_epochs=N_EPOCHS,
    )
