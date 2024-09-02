from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import torch
import typer
from diversity import homogenization_score, compression_ratio, ngram_diversity_score, get_pos, self_repetition
from lexical_diversity import lex_div as ld

app = typer.Typer()

@app.command()
def summarization(
        dataset: str = 'cnn_dailymail',
        dataset_config: str = '3.0.0',
        column: str = 'article',
        split: str = 'test',
        model: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        tokenizer: Optional[str] = None,
        ngram: Optional[int] = 5
):
    tokenizer = tokenizer or model

    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer,
                                              padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    # load dataset (either custom CSV or dataset from HF)
    if dataset.endswith('.csv'): 
        data = load_dataset("csv", data_files=dataset)[split][:10][column]
    else: 
        data = load_dataset(dataset, dataset_config)
        data = data[split][:20][column]
    data = [d + ' [INST] Summarize the above text. [/INST] ' for d in data]
    outputs = []

    for d in data:
        input = tokenizer(d, padding=True, return_tensors="pt").to('cuda')
        input_length = input.input_ids.shape[1]
        outputs.append(tokenizer.batch_decode(model.generate(**input, max_new_tokens=input_length/2, do_sample=False, pad_token_id=tokenizer.eos_token_id)[:, input_length:], skip_special_tokens=True)[0])
    
    joint_outputs = ' '.join(outputs)
    tokenized_outputs = sent_tokenize(joint_outputs)
    flt = ld.flemmatize(joint_outputs)
    joined_pos, _ = get_pos(tokenized_outputs)
    
    bleu = homogenization_score(outputs, 'bleu')
    rougel = homogenization_score(outputs, 'rougel')
    bertscore = homogenization_score(outputs, 'bertscore')
    self_repetition_score = self_repetition(outputs)
    mattr = ld.mattr(flt)
    diversity_score = ngram_diversity_score(outputs, ngram-1)
    hdd = ld.hdd(flt)
    compression = compression_ratio(outputs, 'gzip')
    pos_compression = compression_ratio(joined_pos, 'gzip')
    out_lens = [len(output) for output in outputs]
    avg_len = sum(out_lens)/len(outputs)
    
    print(f"avg_len: {avg_len}")
    print(f"bleu: {round(bleu, 4)}")
    print(f"rougel: {round(rougel, 4)}")
    print(f"bertscore: {round(bertscore, 4)}")
    print(f"self_repetition_score: {round(self_repetition_score, 4)}")
    print(f"mattr: {round(mattr, 4)}")
    print(f"diversity_score: {round(diversity_score, 4)}")
    print(f"hdd: {round(hdd, 4)}")
    print(f"compression: {round(compression, 4)}")
    print(f"pos_compression: {round(pos_compression, 4)}")
    return [avg_len, compression, pos_compression, diversity_score, self_repetition_score, rougel, bertscore, bleu, mattr, hdd]


def display_results():
    pass


if __name__ == "__main__":
    app()
