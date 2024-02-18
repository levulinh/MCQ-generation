import torch
from torch.utils.data import Dataset
from tqdm import tqdm

SEP_TOKEN = "<sep>"


class QA_Dataset_squad(Dataset):
    def __init__(self, tokenizer, raw_data, q_len, t_len, answer_masked_chance=0.3):
        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len
        self.data = []
        # Remove non-answers questions
        print("Removing non-answers questions")
        for d in tqdm(raw_data, desc="Filtering items"):
            if d["answers"]["text"] != []:
                self.data.append(d)
        print(f"Removed {len(raw_data) - len(self.data)} non-answers questions")
        self.answer_masked_chance = answer_masked_chance

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        question = data_dict["question"]
        context = data_dict["context"]
        correct_answer = data_dict["answers"]["text"][0]

        if torch.rand(1) < self.answer_masked_chance:
            answer = "[MASK]"
        else:
            answer = correct_answer

        source_tokens = self.tokenizer(
            f"{answer} {SEP_TOKEN} {context}",
            max_length=self.q_len,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )
        target_tokens = self.tokenizer(
            f"{correct_answer} {SEP_TOKEN} {question}",
            max_length=self.t_len,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )

        labels = torch.tensor(target_tokens["input_ids"], dtype=torch.long)
        labels[labels == 0] = -100

        return {
            "input_ids": torch.tensor(source_tokens["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(source_tokens["attention_mask"], dtype=torch.long),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(target_tokens["attention_mask"], dtype=torch.long),
        }


class QA_Dataset_race(Dataset):
    def __init__(self, tokenizer, raw_data, q_len, t_len, answer_masked_chance=0.3):
        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len
        self.data = raw_data
        self.answer_masked_chance = answer_masked_chance

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        question = data_dict["question"]
        context = data_dict["article"]
        options = data_dict["options"]
        answer_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        correct_answer = options[answer_mapping[data_dict["answer"]]]
        # distraction = [option for option in options if option != answer]

        if torch.rand(1) < self.answer_masked_chance:
            answer = "[MASK]"
        else:
            answer = correct_answer

        source_tokens = self.tokenizer(
            f"{answer} {SEP_TOKEN} {context}",
            max_length=self.q_len,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )
        target_tokens = self.tokenizer(
            f"{correct_answer} {SEP_TOKEN} {question}",
            max_length=self.t_len,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )

        labels = torch.tensor(target_tokens["input_ids"], dtype=torch.long)
        labels[labels == 0] = -100

        return {
            "input_ids": torch.tensor(source_tokens["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(source_tokens["attention_mask"], dtype=torch.long),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(target_tokens["attention_mask"], dtype=torch.long),
        }


class QA_Dataset_mctest(Dataset):
    def __init__(self, raw_data):
        self.data = raw_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        print(data_dict)
        question = data_dict["question"]
        context = data_dict["story"]
        options = data_dict["answer_options"]
        correct_answer = options[data_dict["answer"]]
        distractions = [option for option in options.values() if option != correct_answer]

        return {
            "question": question,
            "context": context,
            "correct_answer": correct_answer,
            "distractions": distractions,
        }
