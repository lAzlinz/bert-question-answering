from datasets import load_dataset
from transformers import DefaultDataCollator
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer, pipeline

datasets_dir: str = './datasets/'
output_dir: str = './models/'

model_name: str = "deepset/roberta-base-squad2"
output_name: str = "my_model"
train_set_name: str = 'train_dataset.json' # 'new_training_set_v2.json'
# val_set_name: str = 'new_testing_set_v2.json'

train_path: str = datasets_dir + train_set_name
# val_path: str = datasets_dir + val_set_name
output_path: str = output_dir + output_name

dataset = load_dataset("json", data_files={"train": train_path}) # , "test": val_path})
# dataset = load_dataset('squad', split="train[:5000]")
# dataset = dataset.train_test_split(test_size=0.2)
dataset

tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    questions = [q.strip() for q in examples['question']]
    inputs = tokenizer(
        questions,
        examples['context'],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length"
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples['answers']
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1

        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # if the answer is not fully inside the context, label it (0,0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

data_collator = DefaultDataCollator()

model = AutoModelForQuestionAnswering.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir=output_path,
    evaluation_strategy="no", # epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    # push_to_hub=True,
    # for exporting
    save_strategy="epoch",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    # eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if __name__ == '__main__':
    trainer.train()
    model.config.save_pretrained(output_path)