from datasets import Dataset, DatasetDict
import torch
from random import randrange, sample
from transformers import DataCollatorForSeq2Seq
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import argparse
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
from collections import Counter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path to save fine-tuned t5 model')
parser.add_argument('--model', type=str, help='pretrained t5', default='google/flan-t5-xl')
parser.add_argument('--synthetic_data', type=str, help='path to synthetic data file')
parser.add_argument('--adverse', action='store_true', help='only add adverse labels')
parser.add_argument('--prompt', type=str, help='prepend string to prompt T5 model', default='summarize: ')
parser.add_argument('--undersample', type=float, help='amount to keep', default=0.0)
parser.add_argument('--gold', type=float, help='amount fo REAL data to keep', default=0.0)
args = parser.parse_args()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if args.adverse:
    LABELS = {'TRANSPORTATION_distance', 'TRANSPORTATION_resource',
        'TRANSPORTATION_other', 'HOUSING_poor', 'HOUSING_undomiciled','HOUSING_other',
        'RELATIONSHIP_divorced', 'RELATIONSHIP_widowed', 'RELATIONSHIP_single',
        'PARENT','EMPLOYMENT_underemployed','EMPLOYMENT_unemployed', 'EMPLOYMENT_disability','SUPPORT_minus'}
else:
    LABELS = {'TRANSPORTATION_distance', 'TRANSPORTATION_resource',
        'TRANSPORTATION_other', 'HOUSING_poor', 'HOUSING_undomiciled',
        'HOUSING_other', 'RELATIONSHIP_married', 'RELATIONSHIP_partnered',
        'RELATIONSHIP_divorced', 'RELATIONSHIP_widowed', 'RELATIONSHIP_single',
        'PARENT','EMPLOYMENT_employed', 'EMPLOYMENT_underemployed',
        'EMPLOYMENT_unemployed', 'EMPLOYMENT_disability', 'EMPLOYMENT_retired',
        'EMPLOYMENT_student', 'SUPPORT_plus', 'SUPPORT_minus'}

BROAD_LABELS = {lab.split('_')[0] for lab in LABELS}
BROAD_LABELS.add('<NO_SDOH>')

LABEL_BROAD_NARROW = LABELS.union(BROAD_LABELS)

MODEL_ID= args.model
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)

MAX_S_LEN = 100
MAX_T_LEN = 40


def undersample(df, label, keep_percent):
    """
    Undersamples the majority class in a Pandas dataframe to balance the classes.

    Parameters:
    df (pandas.DataFrame): The dataframe to undersample.
    keep_percent (float): The percentage of the majority class to keep.

    Returns:
    pandas.DataFrame: The undersampled dataframe.
    """
    # Find the majority class based on the labels column
    counts = df[label].value_counts()
    majority_class = counts.idxmax()

    # Get the indices of rows in the majority class
    majority_indices = df[df[label] == majority_class].index

    # Calculate the number of majority class rows to keep
    num_majority_keep = int(keep_percent * counts[majority_class])

    # Get a random subset of the majority class rows to keep
    majority_keep_indices = np.random.choice(majority_indices, num_majority_keep, replace=False)

    # Get the indices of rows in the minority class
    minority_indices = df[df[label] != majority_class].index

    # Combine the majority class subset and the minority class rows
    undersampled_indices = np.concatenate([majority_keep_indices, minority_indices])

    # Return the undersampled dataframe
    return df.loc[undersampled_indices]


def filter_rows_by_label_percentage(df, percentage):
    # Calculate the number of rows to keep for '<NO_SDOH>' label
    no_sdoh_rows = int(len(df[df['LABEL'] == '<NO_SDOH>']) * percentage)

    # Calculate the number of rows to keep for other label values
    other_rows = int(len(df[df['LABEL'] != '<NO_SDOH>']) * percentage)

    # Filter rows with '<NO_SDOH>' label and sample
    no_sdoh_data = df[df['LABEL'] == '<NO_SDOH>'].sample(n=no_sdoh_rows)

    # Filter rows with other label values and sample
    other_data = df[df['LABEL'] != '<NO_SDOH>'].sample(n=other_rows)

    # Concatenate the two filtered DataFrames
    filtered_df = pd.concat([no_sdoh_data, other_data])

    filtered_df.reset_index(inplace=True, drop=True)

    return filtered_df


def generate_label_list(row: pd.DataFrame) -> str:
    """
    Generate a label list based on the given row from a Pandas DataFrame.

    Args:
        row (pd.DataFrame): A row from a Pandas DataFrame.

    Returns:
        str: A comma-separated string of labels extracted from the row.

    Examples:
        >>> df = pd.DataFrame({'label1_1': [1], 'label2_0': [0], 'label3_1': [1]})
        >>> generate_label_list(df.iloc[0])
        'label1,label3'

        >>> df = pd.DataFrame({'label2_0': [0], 'label3_0': [0]})
        >>> generate_label_list(df.iloc[0])
        '<NO_SDOH>'
    """
    labels = set()
    for col_name, value in row.items():
        if col_name in LABELS and value == 1:
            labels.add(col_name.split('_')[0])
    if len(labels) == 0:
        labels.add('<NO_SDOH>')
    return ','.join(list(labels))


def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = [args.prompt + item for item in sample["text"]]
    # tokenize inputs
    model_inputs = TOKENIZER(inputs, max_length=MAX_S_LEN, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = TOKENIZER(text_target=sample["SDOHlabels"], max_length=MAX_T_LEN, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != TOKENIZER.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == '__main__':
    train_data = pd.read_csv('../data/train_sents.csv')
    train_data.fillna(value={'text':''}, inplace=True)
    train_data['LABEL'] = train_data.apply(generate_label_list, axis=1).tolist()

    if args.undersample:
        train_data = undersample(train_data, label='LABEL', keep_percent=args.undersample)
    if args.gold:
        train_data = filter_rows_by_label_percentage(train_data, args.gold)

    train_text = train_data['text'].tolist()
    train_labels = train_data['LABEL'].tolist()

    if args.synthetic_data:
        synthetic_data = pd.read_csv(args.synthetic_data)
        synthetic_data = synthetic_data[synthetic_data['label'].isin(BROAD_LABELS)]
        if args.adverse:
            synthetic_data = synthetic_data[synthetic_data['adverse']=='adverse']
        synthetic_data.reset_index(inplace=True, drop=True)
        binary_synthetic = pd.get_dummies(synthetic_data['label'])
        binary_synthetic['text'] = synthetic_data['text']
        synth_labels = binary_synthetic.apply(generate_label_list, axis=1).tolist()
        synth_text = synthetic_data['text'].tolist()

        train_text.extend(synth_text)
        train_labels.extend(synth_labels)

    train_t5 = pd.DataFrame({'text':train_text, 'SDOHlabels':train_labels})
    train_dataset = Dataset.from_pandas(train_t5)


    dataset = DatasetDict()
    dataset['train'] = train_dataset
    print(f"Train dataset size: {len(dataset['train'])}")    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text", "SDOHlabels"])

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, load_in_8bit=True, device_map={"":0}) #{"":1}

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        TOKENIZER,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    peft_model_id = args.model_path
    output_dir = peft_model_id
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir, 
        per_device_train_batch_size=32,
        learning_rate=1e-3, # higher learning rate
        num_train_epochs=3,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",
        # report_to="tensorboard",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # train model
    trainer.train()
    # Save our LoRA model & TOKENIZER results
    trainer.model.save_pretrained(output_dir)
    TOKENIZER.save_pretrained(output_dir)