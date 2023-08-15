import transformers
import torch
import pandas as pd
import argparse
import random
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, BertForSequenceClassification
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler
import ray
from ray import tune
from ray.tune import CLIReporter
from datasets import Dataset, load_dataset, DatasetDict, concatenate_datasets
from functools import partial
from utils import grade_preproc, group_labels, undersample_dataset, data_split
import os
from collections import Counter
import pathlib
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from torch import nn
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.utils import class_weight

# Disable logging for raytune, but it will still make folders and jsons for experiment states
# They're not big files, but should be deleted PATH: ./to_be_deleted_rayArtifact
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='The path to the directory to temporarily store checkpoints')
parser.add_argument('--evaldir', type=str, help='The path to the directory to store model evaluation results')
parser.add_argument('--num_trials', type=int, help='Number hyperparameter trials', default=5)
parser.add_argument('--seqlens', type=str, help='list of sequence lengths to search for ray', default='20,35,50')
parser.add_argument('--batches', type=str, help='list of batch sizes to search for ray', default='32,64,128')
parser.add_argument('--model', type=str, help='select model to run classification: (BERT, ROBERTA, BIOBERT)', default='bert-base-uncased')
parser.add_argument('--synth_data', type=str, help='path to synthetic data file', default='')
parser.add_argument('--undersample', type=float, default=0.0, help='undersample majority class in train set by proportion. E.g. 0.2 will keep 20 percent of majority class data')
parser.add_argument('--ray', action='store_true', help='tune hyperparameters')
parser.add_argument('--adverse', action='store_true', help='for non adverse synthetic data')
parser.add_argument('--epochs', type=int, default=5)

args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED_VAL = 42
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

MLB = MultiLabelBinarizer()
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
if args.ray:
    ray.init(log_to_driver=False)
    

class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to(DEVICE) # batch[0, 1, 0, 1, 0, 0]
        # forward pass
        outputs = model(inputs['input_ids'])
        logits = outputs.get("logits").to(DEVICE)
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.BCEWithLogitsLoss().to(DEVICE)
        loss = loss_fct(logits.to(DEVICE), labels.float().to(DEVICE))
        return (loss, outputs) if return_outputs else loss
    

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


def compute_metrics(pred):
    """
    Calculate Evaluation metrics
    """
    labels = pred.label_ids
    logits = torch.tensor(pred.predictions)
    act = nn.Sigmoid()
    probs = act(logits)
    preds = (probs>= 0.5).int()
    
    # labels = mlb.fit_transform(labels)
    # preds = MLB.transform(preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds)
    micro_f1  = precision_recall_fscore_support(labels, preds, average='micro')[2]
    weight_f1 = precision_recall_fscore_support(labels, preds, average='weighted')[2]
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro')[2]

    metrics_out = {'macro_f1':macro_f1, 'micro_f1': micro_f1, 'weighted_f1': weight_f1}
    for i, lab in enumerate(list(MLB.classes_)):
        metrics_out['precision_'+str(lab)] = prec[i]
        metrics_out['recall_'+str(lab)] = rec[i]
        metrics_out['f1_'+str(lab)] = f1[i]
    print(classification_report(labels, preds, target_names=MLB.classes_))
    return metrics_out


def train_hf(config, dataset):
    # Define the Trainer and TrainingArguments objects
    # Initialize the tokenizer with the sequence_length parameter
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt", max_length=config["sequence_length"])
    
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    training_args = TrainingArguments(
        output_dir=args.logdir,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["epochs"],
        disable_tqdm=False,
        bf16=True, # bfloat16 training
        optim='adamw_hf',
        logging_dir=f"{args.logdir}/logs",
        overwrite_output_dir = True,
        evaluation_strategy = 'epoch',
        weight_decay= config["weight_decay"],      
        save_strategy='epoch',
        save_total_limit = 1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed = SEED_VAL,
        gradient_accumulation_steps = config["gradient_accumulation_steps"]
        )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.model,
        num_labels=len(dataset['train']['labels'][0]),
        attention_probs_dropout_prob=config["hidden_dropout_prob"],
        hidden_dropout_prob=config["hidden_dropout_prob"]
        )

    # clws = torch.tensor([config["class_weight0"], config["class_weight1"]], dtype=torch.float).to(DEVICE)
    trainer = BCETrainer(
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['dev'],
        model=model,
        compute_metrics=compute_metrics,
        )

    # Train the model and return the evaluation
    trainer.train()
    eval_result = trainer.evaluate()
    if args.ray:
        tune.report(eval_result)
    else:
        return eval_result


def main(args):
    train_data = pd.read_csv('./data/train_sents.csv')
    dev_data = pd.read_csv('./data/dev_sents.csv')

    train_data.fillna(value={'text':''}, inplace=True)
    dev_data.fillna(value={'text':''}, inplace=True)

    dev_text = dev_data['text'].tolist()
    dev_labels = dev_data.apply(generate_label_list, axis=1).tolist()

    train_data['LABEL'] = train_data.apply(generate_label_list, axis=1).tolist()
    
    if args.undersample:
        train_data = undersample(train_data, label='LABEL', keep_percent=args.undersample)
    train_text = train_data['text'].tolist()
    train_labels = train_data['LABEL'].tolist()

    if args.synth_data:
        synthetic_data = pd.read_csv(args.synth_data)
        if args.adverse:
            synthetic_data = synthetic_data[synthetic_data['adverse']=='adverse']
        synthetic_data.reset_index(inplace=True, drop=True)

        binary_synthetic = pd.get_dummies(synthetic_data['label'])
        binary_synthetic['text'] = synthetic_data['text']
        synth_labels = binary_synthetic.apply(generate_label_list, axis=1).tolist()
        synth_text = synthetic_data['text'].tolist()

        train_text.extend(synth_text)
        train_labels.extend(synth_labels)

    train_labels = [labs.split(',') for labs in train_labels]
    train_labs_mlb = MLB.fit_transform(train_labels)
    train_labs_mlb = [ar.tolist() for ar in train_labs_mlb]

    dev_labels = [labs.split(',') for labs in dev_labels]
    dev_labs_mlb = MLB.transform(dev_labels)
    dev_labs_mlb = [ar.tolist() for ar in dev_labs_mlb]

    train_t5 = pd.DataFrame({'text':train_text, 'labels':train_labs_mlb})
    dev_t5 = pd.DataFrame({'text':dev_text, 'labels':dev_labs_mlb})

    train_dataset = Dataset.from_pandas(train_t5)
    dev_dataset = Dataset.from_pandas(dev_t5)

    dataset = DatasetDict()
    dataset['train'] = train_dataset
    dataset['dev'] = dev_dataset

    seq_length_search = [int(x) for x in args.seqlens.split(',')]
    batch_size_search = [int(x) for x in args.batches.split(',')]

    params_dict ={
            'model':args.model,
            'undersample_bool':args.undersample
            }
    
    if args.ray:
        if args.undersample:
            usample = args.undersample
        else:
            usample = 1
        config_space = {
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice(batch_size_search),
            "hidden_dropout_prob": tune.uniform(0.1, 0.5),
            "undersample": usample,
            "weight_decay": tune.loguniform(1e-8, 1e-5),
            "sequence_length": tune.choice(seq_length_search),
            "gradient_accumulation_steps": 3,
            "epochs": args.epochs  
            }

        scheduler = ASHAScheduler(
            metric="_metric/eval_macro_f1",
            mode="max",
            grace_period=1,
            reduction_factor=2
            )
        
        met_cols = ["training_iteration","macro_f1", "micro_f1", "precision", "recall"]
        for i in range(len(train_labs_mlb[0])):
            met_cols.append('precision_'+str(i))
            met_cols.append('recall_'+str(i))
            met_cols.append('f1_'+str(i))

        reporter = CLIReporter(
            parameter_columns=list(config_space.keys()),
            metric_columns=met_cols,
        )
        result = tune.run(
            partial(train_hf,dataset=dataset),
            config=config_space,
            num_samples=args.num_trials,
            resources_per_trial={"gpu": 1},
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir="./to_be_deleted_rayArtifact",
            name='empty_folders',
            log_to_file=False,
            )

        best_trial = result.get_best_trial(metric='_metric/eval_macro_f1', mode='max', scope="all")
        config_dict = best_trial.config
        dev_eval_dict = best_trial.last_result['_metric']
        output_dict = {**params_dict, **config_dict, **dev_eval_dict}

        outpath = pathlib.Path().joinpath(args.evaldir, 'multi_BERT_ray.csv')
        print(output_dict)
        if os.path.isfile(outpath):
            indf = pd.read_csv(outpath)
            outdf = pd.concat([indf, pd.DataFrame([output_dict])], ignore_index=True)
        else:
            outdf = pd.DataFrame([output_dict])
        outdf.to_csv(outpath, index=False)
    else:
        config_space = {
            "learning_rate": 5e-5,
            "batch_size":32, #32
            "hidden_dropout_prob": 0.1,
            "undersample": 1.0,
            "weight_decay": 2e-8,
            "sequence_length": 100,
            "gradient_accumulation_steps": 3,
            "epochs": 10 
        }

        dev_eval_dict = train_hf(config_space, dataset)
        output_dict = {**params_dict, **config_space, **dev_eval_dict}
        outpath = pathlib.Path().joinpath(args.evaldir, 'multi_BERT_noray.csv')
        print(output_dict)
        if os.path.isfile(outpath):
            indf = pd.read_csv(outpath)
            outdf = pd.concat([indf, pd.DataFrame([output_dict])], ignore_index=True)
        else:
            outdf = pd.DataFrame([output_dict])
        outdf.to_csv(outpath, index=False)
        

if __name__ =='__main__':
    main(args)