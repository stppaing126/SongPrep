from transformers import AutoTokenizer
from omegaconf import OmegaConf

from megatron.megatron_tokenizer import MegatronTokenizer


class _Qwen2Tokenizer(MegatronTokenizer):
    def __init__(self, tokenizer_path, extra_vocab_size, vocab_file):
        super().__init__(tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
        self.vocal_list = list(OmegaConf.load(vocab_file))
        self.extra_vocab_size = extra_vocab_size
        self.tokenizer.add_tokens(self.vocal_list)
        self.tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|extra_0|>"))
        self.tokenizer.add_special_tokens(special_tokens_dict=dict(sep_token="<|extra_1|>"))
        self._n_words_size = len(self.tokenizer.get_vocab()) + self.extra_vocab_size

    def __call__(self, text, return_tensors=None,
                    padding=None, max_length=None, truncation=None, add_special_tokens=None):

        return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

    @property
    def vocab_size(self):
        # https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/discussions/7
        # return len(self.tokenizer.encoder) + self.extra_vocab_size
        return self._n_words_size

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def pad(self):
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py#L107
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def sep_token_id(self):
        return self.tokenizer.sep_token_id


def build_tokenizer(args):
    tokenizer = _Qwen2Tokenizer(args.load, args.extra_vocab_size, args.vocab_file)
    # args.padded_vocab_size = _vocab_size_with_padding(
    #     tokenizer.vocab_size, args)
    args.padded_vocab_size  = tokenizer.vocab_size
    # print("args.tensor_model_parallel_size:",args.tensor_model_parallel_size)
    print(f"padded_vocab_size: {args.padded_vocab_size}")
    # args.padded_vocab_size = tokenizer.vocab_size
    return tokenizer

