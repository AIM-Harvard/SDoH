from datasets import Dataset, DatasetDict
import torch
from random import randrange, sample
from transformers import DataCollatorForSeq2Seq, T5ForConditionalGeneration
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, PeftModel, PeftConfig
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import json
import argparse
import tqdm
import numpy as np
import random
import os

SEED_VAL = 42
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path to trained t5 model. FORMAT: model_task_augmentationBool_undersampleValue_syntheticDataPath')
parser.add_argument('--output_path', type=str, help='path to json store metrics')
parser.add_argument('--error_file', type=str, help='path to synthetic data file')
parser.add_argument('--batch_size', type=int, help='prediction batches')
parser.add_argument('--adverse', action='store_true', help='only add adverse labels')
parser.add_argument('--test', action='store_true', help='eval on test set')
args = parser.parse_args()

if args.adverse:
    LABELS = {'TRANSPORTATION_distance', 'TRANSPORTATION_resource',
        'TRANSPORTATION_other', 'HOUSING_poor', 'HOUSING_undomiciled','HOUSING_other',
        'RELATIONSHIP_divorced', 'RELATIONSHIP_widowed', 'RELATIONSHIP_single',
        'PARENT','EMPLOYMENT_underemployed','EMPLOYMENT_unemployed', 'EMPLOYMENT_disability', 'EMPLOYMENT_retired',
        'EMPLOYMENT_student','SUPPORT_minus'}
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
TOKENIZER = AutoTokenizer.from_pretrained(args.model_path)
MAX_S_LEN = 100
MAX_T_LEN = 40


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


def postprocess_function(preds):
    """
    Perform post-processing on the predictions.

    Args:
        preds (list): A list of predictions.

    Returns:
        list: Processed predictions with fixed labels.

    Examples:
        >>> preds = ['REL', 'EMPLO', 'HOUS', 'UNKNOWN']
        >>> postprocess_function(preds)
        ['RELATIONSHIP', 'EMPLOYMENT', 'HOUSING', 'UNKNOWN']

        >>> preds = ['NO_SD', np.nan, 'SUPP']
        >>> postprocess_function(preds)
        ['<NO_SDOH>', '<NO_SDOH>', 'SUPPORT']
    """
    lab_fixed_dict = {
        'REL': 'RELATIONSHIP',
        'RELAT': 'RELATIONSHIP',
        'EMP': 'EMPLOYMENT',
        'EMPLO': 'EMPLOYMENT',
        'SUPP': 'SUPPORT',
        'HOUS': 'HOUSING',
        'PAREN': 'PARENT',
        'TRANSPORT': 'TRANSPORTATION',
        'NO_SD': '<NO_SDOH>',
        np.nan: '<NO_SDOH>',
        'NO_SDOH>': '<NO_SDOH>',
        '<NO_SDOH': '<NO_SDOH>',
    }

    new_preds = []
    for pred in preds:
        pred_ls = []
        pred = str(pred)
        for pp in pred.split(','):
            if pp in lab_fixed_dict.keys():
                pred_ls.append(lab_fixed_dict[pp])
            else:
                pred_ls.append(pp)
        new_preds.append(','.join(pred_ls))

    return new_preds


def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["text"]]
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


def normal_eval(preds, gold):
    """
    Evaluate the model predictions against the gold labels.

    Args:
        preds (list): A list of prediction strings.
        gold (list): A list of gold label strings.

    Returns:
        dict: Metrics computed for the evaluation.

    """
    pred_temp = [p.split(",") for p in preds]
    gold_list = [g.split(',') for g in gold]

    pred_list = []
    for labs in pred_temp:
        point_pred = [p for p in labs if p in BROAD_LABELS]
        pred_list.append(point_pred)
    mlb = MultiLabelBinarizer()
    oh_gold = mlb.fit_transform(gold_list)
    oh_pred = mlb.transform(pred_list)

    prec, rec, f1, _ = precision_recall_fscore_support(oh_gold, oh_pred)
    micro_f1  = precision_recall_fscore_support(oh_gold, oh_pred, average='micro')[2]
    weight_f1 = precision_recall_fscore_support(oh_gold, oh_pred, average='weighted')[2]
    macro_f1 = precision_recall_fscore_support(oh_gold, oh_pred, average='macro')[2]

    metrics_out = {'macro_f1':macro_f1, 'micro_f1': micro_f1, 'weighted_f1': weight_f1}
    for i, lab in enumerate(list(mlb.classes_)):
        metrics_out['precision_'+str(lab)] = prec[i]
        metrics_out['recall_'+str(lab)] = rec[i]
        metrics_out['f1_'+str(lab)] = f1[i]
    print(classification_report(oh_gold, oh_pred, target_names=mlb.classes_))
    return metrics_out


def predict(dataset, model, batch_size):
    # Initialize empty lists to store predictions and references
    predictions, references = [], []

    # Iterate over the dataset in batches
    for i in tqdm.tqdm(range(0, len(dataset["dev"]), batch_size)):
        # Get the texts for the current batch
        texts = dataset['dev'][i:i+batch_size]

        # Tokenize the texts and convert them to input tensors
        input_ids = TOKENIZER(texts["text"], return_tensors="pt", truncation=True, padding="max_length").input_ids.cuda()

        # Generate predictions using the model
        outputs = model.generate(input_ids=input_ids, do_sample=False, top_p=0.9, max_new_tokens=5, num_beams=4)

        # Decode the generated outputs into text
        outputs = TOKENIZER.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

        # Get the reference labels for the current batch
        labels = dataset['dev'][i:i+batch_size]["SDOHlabels"]

        # Extend the predictions and references lists
        predictions.extend(outputs)
        references.extend(labels)

    # Return the final predictions and references
    return predictions, references

if __name__ == '__main__':
    if args.test:
        dev_data = pd.read_csv('../data/test_sents.csv')
    else:
        dev_data = pd.read_csv('../data/dev_sents.csv')

    dev_data.fillna(value={'text':''}, inplace=True)

    dev_text = dev_data['text'].tolist()
    dev_labels = dev_data.apply(generate_label_list, axis=1).tolist()
    dev_t5 = pd.DataFrame({'text':dev_text, 'SDOHlabels':dev_labels})
    dev_dataset = Dataset.from_pandas(dev_t5)
    dataset = DatasetDict()
    dataset['dev'] = dev_dataset

    config = PeftConfig.from_pretrained(args.model_path)
    # load base LLM model and tokenizer
    reloaded_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,  load_in_8bit=True,  device_map={"":0})
    # Load the Lora model
    reloaded_model = PeftModel.from_pretrained(reloaded_model, args.model_path, device_map={"":0})
    reloaded_model.eval()

    predictions, references = predict(dataset, reloaded_model, 6)

    df = pd.DataFrame({'gold':references, 'pred':predictions})
    df.to_csv(args.error_file, index=False)

    params = args.model_path.split('_')
    param_dict = {'model':params[0], 'task':params[1], 'train_data':params[2], 'undersample':params[3], 'synthetic_data':params[4]}

    metrics = normal_eval(predictions, references)
    print('='*30+'POST PROCESSED'+'='*30)
    processed_predictions = postprocess_function(predictions)
    processed_metrics = normal_eval(processed_predictions, references)
    output_dict = {**param_dict, **processed_metrics}
    if os.path.isfile('./processed_results_dev.csv'):
        indf = pd.read_csv('./processed_results_dev.csv')
        outdf = pd.concat([indf, pd.DataFrame([output_dict])], ignore_index=True)
    else:
        outdf = pd.DataFrame([output_dict])
    outdf.to_csv('./processed_results_dev.csv', index=False)

    with open(args.output_path, 'w') as j:
        json.dump(metrics, j, indent=4)