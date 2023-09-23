from collections import defaultdict
import torch
import tqdm
import math
import tree
import functools
import ray
import copy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from torch.utils.data import TensorDataset
MODEL_PATH = '/media/ertank/Data/ml/training/finetune/full_4/checkpoint_000000'


def verita_collate_fn(batch, tokenizer, block_size, device):
    IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

    question_column_name = 'question'
    answer_column_name = 'answer'
    final_input_ids = []
    final_labels = []
    final_attention_mask = []
    for i in range(len(batch[question_column_name])):
        question = batch[question_column_name][i]
        example_row = question + batch[answer_column_name][i]
        just_question_encoded = tokenizer.encode(question)

        encoded_example = torch.tensor(
            tokenizer.encode(example_row), dtype=torch.int64)

        labels = copy.deepcopy(encoded_example)
        labels[: len(just_question_encoded)] = -1
        padding = block_size - encoded_example.shape[0]
        if padding > 0:
            # Padding with -1 to be able to mask later
            encoded_example = torch.cat(
                (encoded_example, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat(
                (labels, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            if len(just_question_encoded) > block_size:
                print ('Question is longer than block size, ignoring this example')
                continue
            encoded_example = encoded_example[: block_size]

        example_mask = encoded_example.ge(0)
        label_mask = labels.ge(0)
        encoded_example[~example_mask] = tokenizer.pad_token_id
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        if encoded_example.shape[0] != block_size:
            print ('Example shape is not equal to block size, ignoring this example')
            continue
        if labels.shape[0] != block_size:
            print ('Labels shape is not equal to block size, ignoring this example')
            continue
        if example_mask.shape[0] != block_size:
            print ('Example mask shape is not equal to block size, ignoring this example')
            continue
        print('Encoded example: ' + str(encoded_example))
        print('Labels: ' + str(labels))
        print('Example mask: ' + str(example_mask))

        final_input_ids.append(torch.tensor(
            [encoded_example.tolist(),], dtype=torch.int64))
        final_labels.append(torch.tensor(
            [labels.tolist(), ], dtype=torch.int64))
        final_attention_mask.append(torch.tensor(
            [example_mask.tolist(), ], dtype=torch.int64))

    final_input_ids = torch.cat(final_input_ids, dim=0).to(device)
    final_labels = torch.cat(final_labels, dim=0).to(device)
    final_attention_mask = torch.cat(final_attention_mask, dim=0).to(device)

    return {'input_ids': final_input_ids,
            'labels': final_labels,
            'attention_mask': final_attention_mask
            }

def question_answer_collate_fn(batch, tokenizer, block_size, device):
    IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
    question_column_name = 'question'
    answer_column_name = 'answer'
    final_input_ids = []
    final_labels = []
    final_attention_mask = []
    for i in range(len(batch[question_column_name])):
        example = batch[question_column_name][i] + batch[answer_column_name][i]
        question = batch[question_column_name][i]
        prompt = torch.tensor(
            tokenizer.encode(question), dtype=torch.int64
        )
        example = tokenizer.encode(example)
        example.append(tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        # To have at least some answering
        if prompt.shape[0] > block_size - 3:
            print('Question is very long, ignoring this example')
            continue
        padding = block_size - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: block_size]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        print('Encoded example: ' + str(example))
        print('Labels: ' + str(labels))
        print('Example mask: ' + str(example_mask))

        final_input_ids.append(torch.tensor(
            [example.tolist(),], dtype=torch.int64))
        final_labels.append(torch.tensor(
            [labels.tolist(), ], dtype=torch.int64))
        final_attention_mask.append(torch.tensor(
            [example_mask.tolist(), ], dtype=torch.int64))

    final_input_ids = torch.cat(final_input_ids, dim=0).to(device)
    final_labels = torch.cat(final_labels, dim=0).to(device)
    final_attention_mask = torch.cat(final_attention_mask, dim=0).to(device)

    return {'input_ids': final_input_ids,
            'labels': final_labels,
            'attention_mask': final_attention_mask
            }

def collate_fn(batch, tokenizer, block_size, device):
    out_batch = tokenizer(
        list(batch["input"]),
        padding="max_length",
        max_length=block_size,
        truncation=True,
        return_tensors="pt",
    )
    out_batch["labels"] = out_batch["input_ids"].clone()

    out_batch = tree.map_structure(lambda x: x.to(device), out_batch)
    print('Out batch: ' + str(out_batch))
    print('Out batch type: ' + str(type(out_batch)))
    return out_batch


def get_tokenizer(special_tokens):

    # Context for legacy=True: https://github.com/huggingface/transformers/issues/25176
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(special_tokens, special_tokens=True)

    return tokenizer


def evaluate(
    *, model, eval_ds, bsize, ds_kwargs, tokenizer
):
    model.eval()
    losses = []

    eval_dataloader = eval_ds.iter_torch_batches(batch_size=bsize, **ds_kwargs)
    print('Eval dataloader: ' + str(eval_dataloader))
    eval_ds_len = len(list(eval_ds.iter_batches(batch_size=bsize)))
    for batch in eval_dataloader:
        print('Batch: ' + str(batch))
        with torch.no_grad():
            outputs = model(**batch)
        for logit in outputs.logits:
            print(tokenizer.decode(torch.argmax(
                logit, dim=1).tolist()))
        loss = outputs.loss
        # The tensors are gathered by concatenating them on the first dimension, so we
        # add a new dimension to the scalar loss to get a tensor of shape (K,) for K
        # workers.
        losses.append(loss)

    # We stack losses so that we have a tensor of shape (T, K) where T is the number of
    # steps and K is the number of workers.
    losses = torch.stack(losses)
    try:
        eval_loss = torch.mean(losses).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss


def start_eval():
    print("Running evaluation ...")
    valid_ds = ray.data.read_json(
        '/media/ertank/Data/ml/training/ray_verita/data/test.jsonl')
    tokenizer = get_tokenizer(
        special_tokens=["[INST]", "<<SYS>>", "<</SYS>>", "[/INST]", "<PAD>"])
    tokenizer.pad_token = '<PAD>'
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # `use_cache=True` is incompatible with gradient checkpointing.
        use_cache=False,
    )
    print(f"Done loading model")
    model.resize_token_embeddings(len(tokenizer))
    collate_partial = functools.partial(
        question_answer_collate_fn,
        tokenizer=tokenizer,
        block_size=512,
        device='cpu',
    )
    perplex, eloss = evaluate(
        model=model,
        eval_ds=valid_ds,
        bsize=512,
        ds_kwargs={"collate_fn": collate_partial},
        tokenizer=tokenizer
    )
    print('Perplexity: ' + str(perplex) + ' Loss: ' + str(eloss))


if __name__ == "__main__":
    start_eval()
