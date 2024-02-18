from typing import Tuple

import datasets
import evaluate
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast

# Load the dataset
mctest_data = datasets.load_dataset("sagnikrayc/mctest", trust_remote_code=True)["validation"]

model = T5ForConditionalGeneration.from_pretrained("model/ckpt_qg_squad/17", return_dict=True)
model.eval()
# model.push_to_hub("t5_question_generation_squad")
tokenizer = T5TokenizerFast.from_pretrained("model/tokenizer_qg_squad/17")
# tokenizer.push_to_hub("t5_question_generation_squad")

tokenizer.add_tokens("<sep>")
Q_LEN = 256  # Question Length
T_LEN = 64  # Target Length
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


def predict_answer(context, answer):
    inputs = tokenizer(
        f"{answer} <sep> {context}", max_length=Q_LEN, padding="max_length", truncation=True, add_special_tokens=True
    )

    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=T_LEN)

    predicted_answer = tokenizer.decode(
        outputs.flatten(),
        skip_special_tokens=True,
    )
    return predicted_answer


def parse_qa(pred_string: str) -> Tuple[str, str]:
    if len(pred_split := pred_string.split("<sep>")) != 2:
        return None, None
    ans, ques = pred_split
    ans = ans.strip()
    ques = ques.strip()
    if ques[-1] in "!@#$%^&*()_+{}[]|\\:;\"'<>,./":
        # Replace the special character with a question mark
        ques = ques[:-1] + "?"
    elif ques[-1] not in "?":
        # Add a question mark to the end
        ques += "?"
    return ans, ques


def calculate_bleu(bleu, source_str, target_str):
    score = bleu.compute(predictions=[source_str], references=[target_str])
    return score["google_bleu"]


bleu = evaluate.load("google_bleu")

answer_bleu_scores = []
question_bleu_scores = []
for data in tqdm(mctest_data):
    ref_question = data["question"]
    context = data["story"]
    options = data["answer_options"]
    ref_answer = options[data["answer"]]
    prediction = predict_answer(context, ref_answer)
    pred_answer, pred_question = parse_qa(prediction)
    if pred_question is not None:
        print(f"Predicted Question: {pred_question}")
        print(f"Predicted Answer: {pred_answer}")
        print(f"Reference Question: {ref_question}")
        print(f"Reference Answer: {ref_answer}")
        print(f"Question BLEU: {calculate_bleu(bleu, pred_question, ref_question)}")
        question_bleu_scores.append(calculate_bleu(bleu, pred_question, ref_question))
        answer_bleu_scores.append(calculate_bleu(bleu, pred_answer, ref_answer))

print("Average BLEU score for the answers:", sum(answer_bleu_scores) / len(answer_bleu_scores))
print("Average BLEU score for the questions:", sum(question_bleu_scores) / len(question_bleu_scores))
