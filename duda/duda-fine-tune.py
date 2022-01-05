print('has changed 24.')

import nlp
import re
import string

import spacy
spacy_nlp = spacy.load("en_core_web_sm")

import datasets
import transformers
import pandas as pd
from datasets import Dataset

#Tokenizer
from transformers import RobertaTokenizerFast

#Encoder-Decoder Model
from transformers import EncoderDecoderModel

# from transformers import Seq2SeqTrainingArguments
from seq2seq_training_args import Seq2SeqTrainingArguments

#Training
from seq2seq_trainer import Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from datasets import Dataset

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
#parameter setting
batch_size=8  #
encoder_max_length=500
decoder_max_length=200

def mark_verb_target(text):
    doc = spacy_nlp(text)

    src_text = ''
    target_text = ''
    for token in doc:
        if token.pos_ == 'VERB':
            tp_text = token.text

            # target
            target_text += ' {} '.format(tp_text)
        else:
            tp_text = token.text

            # target
            if tp_text == '\n':
                target_text += tp_text
            elif token.pos_ == 'SPACE':
                target_text += ''
            elif tp_text in string.punctuation:
                target_text += ' {} '.format(tp_text)
            else:
                if token.pos_ == 'NOUN' or token.pos_ == 'PRON' or token.pos_ == 'PROPN':
                    target_text += ' + '
                else:
                    target_text += ' = '

    target_text = re.sub(' +', ' ', target_text)
    target_text = target_text.replace(' ,', ',')
    target_text = target_text.replace(' .', '.')
    target_text = target_text.replace('\n ', '\n')
    target_text = target_text.replace(' \n', '\n')
    target_text = target_text.strip()

    return target_text

def mark_noun_target(text):
    doc = spacy_nlp(text)

    src_text = ''
    target_text = ''
    for token in doc:
        if token.pos_ == 'NOUN' or token.pos_ == 'PRON' or token.pos_ == 'PROPN':
            tp_text = token.text

            # target
            target_text += ' {} '.format(tp_text)
        else:
            tp_text = token.text

            # target
            if tp_text == '\n':
                target_text += tp_text
            elif token.pos_ == 'SPACE':
                target_text += ''
            elif tp_text in string.punctuation:
                target_text += ' {} '.format(tp_text)
            else:
                if token.pos_ == 'VERB':
                    target_text += ' - '
                else:
                    target_text += ' = '

    target_text = re.sub(' +', ' ', target_text)
    target_text = target_text.replace(' ,', ',')
    target_text = target_text.replace(' .', '.')
    target_text = target_text.replace('\n ', '\n')
    target_text = target_text.replace(' \n', '\n')
    target_text = target_text.strip()

    return target_text

def mark_rest_target(text):

    doc = spacy_nlp(text)

    src_text = ''
    target_text = ''
    for token in doc:
        if token.pos_ != 'NOUN' and token.pos_ != 'PRON' and token.pos_ != 'PROPN' and token.pos_ != 'VERB':
            tp_text = token.text

            # target
            target_text += ' {} '.format(tp_text)
        else:
            tp_text = token.text

            # target
            if tp_text == '\n':
                target_text += tp_text
            elif token.pos_ == 'SPACE':
                target_text += ''
            elif tp_text in string.punctuation:
                target_text += ' {} '.format(tp_text)
            else:
                if token.pos_ == 'VERB':
                    target_text += ' - '
                else:
                    target_text += ' + '

    target_text = re.sub(' +', ' ', target_text)
    target_text = target_text.replace(' ,', ',')
    target_text = target_text.replace(' .', '.')
    target_text = target_text.replace('\n ', '\n')
    target_text = target_text.replace(' \n', '\n')
    target_text = target_text.strip()


    return target_text

def mark_rest(text):

    doc = spacy_nlp(text)

    src_text = ''
    target_text = ''
    for token in doc:
        if token.pos_ != 'NOUN' and token.pos_ != 'PRON' and token.pos_ != 'PROPN' and token.pos_ != 'VERB':
            tp_text = token.text

            # target
            target_text += ' {} '.format(tp_text)
        else:
            tp_text = token.text

            # target
            if tp_text == '\n':
                target_text += tp_text
            elif token.pos_ == 'SPACE':
                target_text += ''
            elif tp_text in string.punctuation:
                target_text += ' {} '.format(tp_text)
            else:
                target_text += ' + '

    target_text = re.sub(' +', ' ', target_text)
    target_text = target_text.replace(' ,', ',')
    target_text = target_text.replace(' .', '.')
    target_text = target_text.replace('\n ', '\n')
    target_text = target_text.replace(' \n', '\n')
    target_text = target_text.strip()


    return target_text

def mark_verb(text):

    doc = spacy_nlp(text)

    src_text = ''
    target_text = ''
    for token in doc:
        if token.pos_ == 'VERB':
            tp_text = token.text

            # target
            target_text += ' {} '.format(tp_text)
        else:
            tp_text = token.text

            # target
            if tp_text == '\n':
                target_text += tp_text
            elif token.pos_ == 'SPACE':
                target_text += ''
            elif tp_text in string.punctuation:
                target_text += ' {} '.format(tp_text)
            else:
                target_text += ' + '

    target_text = re.sub(' +', ' ', target_text)
    target_text = target_text.replace(' ,', ',')
    target_text = target_text.replace(' .', '.')
    target_text = target_text.replace('\n ', '\n')
    target_text = target_text.replace(' \n', '\n')
    target_text = target_text.strip()


    return target_text
    
def mark_noun(text):

    doc = spacy_nlp(text)

    src_text = ''
    target_text = ''
    for token in doc:
        if token.pos_ == 'NOUN' or token.pos_ == 'PRON' or token.pos_ == 'PROPN':
            tp_text = token.text

            # target
            target_text += ' {} '.format(tp_text)
        else:
            tp_text = token.text

            # target
            if tp_text == '\n':
                target_text += tp_text
            elif token.pos_ == 'SPACE':
                target_text += ''
            elif tp_text in string.punctuation:
                target_text += ' {} '.format(tp_text)
            else:
                target_text += ' + '

    target_text = re.sub(' +', ' ', target_text)
    target_text = target_text.replace(' ,', ',')
    target_text = target_text.replace(' .', '.')
    target_text = target_text.replace('\n ', '\n')
    target_text = target_text.replace(' \n', '\n')
    target_text = target_text.strip()


    return target_text


def mark(text):

    doc = spacy_nlp(text)

    noun_chunks = []
    for chunk in doc.noun_chunks:
        noun_chunks.append(chunk)

    len_doc = len(doc)
    len_noun_chunks = len(noun_chunks)

    target_text = ''
    src_idx = 0

    for i in range(len_noun_chunks):

        chunk = noun_chunks[i]

        if chunk.start - 1 >= 0:
            for j in list(range(src_idx, chunk.start)):

                tp_text = doc[j].text

                # target
                if tp_text == '\n':
                    target_text += tp_text
                elif doc[i].pos_ == 'SPACE':
                    target_text += ''
                elif tp_text in string.punctuation:
                    target_text += ' {} '.format(tp_text)
                else:
                    target_text += ' + '

        for j in list(range(chunk.start, chunk.end)):

            tp_text = doc[j].text

            # target
            if tp_text == '\n':
                target_text += tp_text
            elif doc[i].pos_ == 'SPACE':
                target_text += ''
            else:
                target_text += ' {} '.format(tp_text)

        src_idx = chunk.end

    if src_idx < len_doc:
        for i in list(range(src_idx, len_doc)):

            tp_text = doc[i].text

            # target
            if tp_text == '\n':
                target_text += tp_text
            elif doc[i].pos_ == 'SPACE':
                target_text += ''
            elif tp_text in string.punctuation:
                target_text += ' {} '.format(tp_text)
            else:
                target_text += ' + '


    target_text = re.sub(' +', ' ', target_text)
    target_text = target_text.replace(' ,', ',')
    target_text = target_text.replace(' .', '.')
    target_text = target_text.replace('\n ', '\n')
    target_text = target_text.replace(' \n', '\n')
    target_text = target_text.strip()


    return target_text

def punctuation_replace(match):
    match = match.group()
    return ' ' + match + ' '

def create_target(summary_text):

    summary_text = summary_text.replace('@highlight', '')

    punctuation = r'[\\"`{|}~@^_\[\]:;<=>\'()*+\-/#$%&.,!?]+'

    summary_text = re.sub(punctuation, punctuation_replace, summary_text)
    summary_text = re.sub(' +', ' ', summary_text)
    summary_text = re.sub('\.+', '.', summary_text)
    summary_text = re.sub('\n+', '\n', summary_text)


    pre_sentences = summary_text.split('\n')
    summary_text = ''
    for sen in pre_sentences:
        if re.match(r'\s', sen):
            continue
        if len(sen) > 0:
            sen = sen.strip()
            if sen[-1] != '.' and sen[-1] != ',' and sen[-1] != '!' and sen[-1] != '?':
                sen += '.\n'
            else:
                sen += '\n'
            summary_text += sen

    # mark_target_text = mark(summary_text)
    # mark_target_text = mark_rest(summary_text)
    # mark_target_text = mark_noun(summary_text)
    # mark_target_text = mark_verb(summary_text)
    
    # mark_target_text = mark_noun_target(summary_text)
    mark_target_text = mark_verb_target(summary_text)
    # mark_target_text = mark_rest_target(summary_text)

    return  mark_target_text

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length)

  for i in range(len(batch["highlights"])):
      batch["highlights"][i] = create_target(batch["highlights"][i])

  outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because RoBERTa automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

# load train and validation data
train_data = nlp.load_dataset("cnn_dailymail", "3.0.0", split="train")
train_dataset = train_data.filter(lambda example: len(example['article'].split(' ')) < 500)

val_data = nlp.load_dataset("cnn_dailymail", "3.0.0", split="validation[:1%]")
val_dataset = val_data.filter(lambda example: len(example['article'].split(' ')) < 500)

print('-----------------{}'.format(train_data.num_rows))
print('-----------------{}'.format(train_dataset.num_rows))
print('Will save ever {} steps.'.format(train_dataset.num_rows / 8))

train_data = ''
val_data = ''

# make train dataset ready
train_dataset = train_dataset.map(
    process_data_to_model_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
)
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# same for validation dataset
val_dataset = val_dataset.map(
    process_data_to_model_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
)
val_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

from transformers import EncoderDecoderModel

roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base", tie_encoder_decoder=True)
# roberta_shared = EncoderDecoderModel.from_pretrained("./checkpoint-23346/", tie_encoder_decoder=True)
# roberta_shared = EncoderDecoderModel.from_pretrained("../../Store/Merge/Verb/checkpoint-70038/", tie_encoder_decoder=True)

# roberta_shared = EncoderDecoderModel.from_pretrained("../../Store/Noun/checkpoint-23344/", tie_encoder_decoder=True)

# set special tokens
roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id
roberta_shared.config.eos_token_id = tokenizer.eos_token_id
roberta_shared.config.pad_token_id = tokenizer.pad_token_id

# sensible parameters for beam search
# set decoding params
roberta_shared.config.max_length = 200
roberta_shared.config.early_stopping = True
roberta_shared.config.no_repeat_ngram_size = 3
roberta_shared.config.length_penalty = 2.0
roberta_shared.config.num_beams = 5
roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

# load rouge for validation
rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


training_args = Seq2SeqTrainingArguments(
    output_dir="./",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    do_train=True,
    do_eval=True,
    num_train_epochs=10,
    logging_steps=10,
    save_steps=11673,
    eval_steps=11673,
    warmup_steps=1024,
    overwrite_output_dir=True,
    save_total_limit=1,
    fp16=True,
)
#
# instantiate trainer
trainer = Seq2SeqTrainer(
    model=roberta_shared,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

transformers.logging.set_verbosity_info()

trainer.train()
# trainer.train("./checkpoint-23346/")
# trainer.train("../../Store/Noun/checkpoint-23344/")
# trainer.train("../../Store/Merge/Verb/checkpoint-70038/")