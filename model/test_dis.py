import datasets
import evaluate
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast

# Load the dataset
mctest_data = datasets.load_dataset("sagnikrayc/mctest", trust_remote_code=True)["test"]

model = T5ForConditionalGeneration.from_pretrained("model/ckpt_dis_mctest/49", return_dict=True)
model.eval()
# model.push_to_hub("t5_distraction_mctest")
tokenizer = T5TokenizerFast.from_pretrained("model/tokenizer_dis_mctest/49")
# tokenizer.push_to_hub("t5_distraction_mctest")

tokenizer.add_tokens("<sep>")
Q_LEN = 256  # Question Length
T_LEN = 80  # Target Length
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


def get_prediction_dis(context, answer, question):
    inputs = tokenizer(
        f"{answer} <sep> {question} <sep> {context}",
        max_length=256,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )

    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=80)

    prediction = tokenizer.decode(
        outputs.flatten(),
        skip_special_tokens=True,
    )
    return prediction


def calculate_bleu(bleu, source_str, target_str):
    score = bleu.compute(predictions=[source_str], references=[target_str])
    return score["google_bleu"]


bleu = evaluate.load("google_bleu")

dis_bleu_scores = []
duplicated = 0
for data in tqdm(mctest_data):
    question = data["question"]
    context = data["story"]
    options = data["answer_options"].values()
    correct_answer_id = data["answer"]
    correct_answer = data["answer_options"][correct_answer_id]
    distraction = [option for option in options if option != correct_answer]
    pred_distraction = get_prediction_dis(context, correct_answer, question)
    breakpoint()
    if len(set([p.strip() for p in pred_distraction.split("<sep>")])) < 3:
        duplicated += 1
    ref_distraction = " ".join(distraction)
    pred_distraction = " ".join([p.strip() for p in pred_distraction.split("<sep>")])
    dis_bleu_scores.append(calculate_bleu(bleu, pred_distraction, ref_distraction))


print("Average BLEU score for the answers:", sum(dis_bleu_scores) / len(dis_bleu_scores))
print("Number of duplicated distractions:", duplicated)
