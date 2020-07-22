from dataclasses import dataclass
from typing import List
import functools

from transformers import PreTrainedTokenizer
import torch
import torch.nn as nn

from .tokenize import BatchTokenizer
from pygaggle.rerank.base import TextType


__all__ = ['LongBatchEncoder', 'EncoderOutputBatch', 'SingleEncoderOutput',
           'SpecialTokensCleaner']


@dataclass
class SingleEncoderOutput:
    encoder_output: torch.Tensor
    token_ids: torch.Tensor
    text: TextType


@dataclass
class EncoderOutputBatch:
    batch: List[torch.Tensor]
    token_ids: List[torch.Tensor]
    texts: List[TextType]

    def as_single(self) -> 'SingleEncoderOutput':
        return SingleEncoderOutput(self.batch[0],
                                   self.token_ids[0], self.texts[0])

    def __iter__(self):
        return iter(SingleEncoderOutput(enc_out, token_ids, text) for
                    (enc_out, token_ids, text)
                    in zip(self.batch, self.token_ids, self.texts))


class SpecialTokensCleaner:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.special_ids = tokenizer.all_special_ids

    def clean(self, output: SingleEncoderOutput) -> SingleEncoderOutput:
        indices = [idx for idx, tok in enumerate(output.token_ids.tolist())
                   if tok not in self.special_ids]
        return SingleEncoderOutput(output.encoder_output[indices],
                                   output.token_ids[indices], output.text)


class LongBatchEncoder(object):
    """
    Encodes batches of documents that are longer than the maximum sequence
    length by striding a window across the sequence dimension.

    Parameters
    ----------
    encoder : nn.Module
        The encoder module, such as `BertModel`.
    tokenizer : BatchTokenizer
        The batch tokenizer to use.
    max_seq_length : int
        The maximum sequence length, typically 512.
    """
    def __init__(self,
                 model: nn.Module,
                 tokenizer: BatchTokenizer,
                 max_seq_length: int = 512):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def encode_single(self, input: TextType) -> SingleEncoderOutput:
        return self.encode([input]).as_single()

    def encode(self, batch_input: List[TextType]) -> EncoderOutputBatch:
        batch_output = []
        batch_ids = []
        for ret in self.tokenizer.traverse(batch_input):
            # this code aligns/pads the whole matrix
            input_ids = ret.features['input_ids']
            lengths = [len(ids) for ids in input_ids]
            batch_ids.extend(map(torch.tensor, input_ids))
            input_ids = [(idx, x) for idx, x in enumerate(input_ids)]
            max_len = min(max(lengths), self.max_seq_length)
            encode_lst = [[] * len(input_ids) ]
            new_input_ids = [(idx, x[:max_len]) for idx, x in input_ids]
            while new_input_ids:
                attn_mask = [[1] * len(x[1]) +
                             [0] * (max_len - len(x[1]))
                             for x in new_input_ids]
                attn_mask = torch.tensor(attn_mask) #.to(self.device)
                nonpadded_input_ids = new_input_ids
                new_input_ids = [x + [0] * (max_len - len(x[:max_len]))
                                 for _, x in new_input_ids]
                new_input_ids = torch.tensor(new_input_ids) #.to(self.device)
                outputs  = self.model(input_ids=new_input_ids.to(self.model.device),
                                        attention_mask=attn_mask.to(self.model.device))
                outputs = outputs[0]
                for (idx, _), output in zip(nonpadded_input_ids, outputs):
                    encode_lst[idx].append(output)

                new_input_ids = [(idx, x[max_len:])
                                 for idx, x in nonpadded_input_ids
                                 if len(x) > max_len]
                max_len = min(max((len(x[1]) for x in new_input_ids),
                                  default=0), self.max_seq_length)
            encode_lst = [torch.cat(l) for l in encode_lst]
            batch_output.extend(encode_lst)
        return EncoderOutputBatch(batch=batch_output, token_ids=batch_ids, texts=batch_input)

    def align(self, input_ids:torch.TensorType):
        #input_ids = features['input_ids']
        #lengths = [len(ids) for ids in input_ids]
        # batch_ids.extend(map(torch.tensor, input_ids))
        # input_ids = [(idx, x) for idx, x in enumerate(input_ids)]
        # figure out the max length of all the encoded elements in the batch
        max_len = min(functools.reduce(max, (len(ids) for ids in input_ids)), self.max_seq_length)
        #max_len = min(max(lengths), self.msl)
        # encode_lst = [[] * len(input_ids) ]
        # new_input_ids = [(idx, x[:max_len]) for idx, x in enumerate(input_ids)]
        def pad_tensor(x):
            length = x.shape[-1]
            return torch.nn.functional.pad(x, [0, max_len-length])

        new_input_ids = torch.stack( [ pad_tensor(torch.tensor(x)) for x in input_ids ], axis=0)

        attn_mask = torch.stack( [pad_tensor(torch.ones(len(x))) for x in input_ids] , axis=0)

        # print(attn_mask.shape)
        # print(new_input_ids.shape)
        #while new_input_ids:
            # mask 
            # attn_mask = np.zeros((batch_size, max_len))
            # 1 on valid tokens, 0 on padding
            # attn_mask = [[1] * len(x[1]) +
            #                 [0] * (max_len - len(x[1]))
            #                 for x in new_input_ids]
            # attn_mask = torch.tensor(attn_mask) #.to(self.device)

            # nonpadded_input_ids = new_input_ids
            # new_input_ids = [x + [0] * (max_len - len(x[:max_len]))
            #                     for _, x in new_input_ids]
            #new_input_ids = torch.tensor(new_input_ids) #.to(self.device)

        outputs  = self.model(input_ids=new_input_ids.to(self.model.device),
                              attention_mask=attn_mask.to(self.model.device))
        outputs = outputs[0]
        # for (idx, _), output in zip(nonpadded_input_ids, outputs):
        #     encode_lst[idx].append(output)

        # new_input_ids = [(idx, x[max_len:])
        #                     for idx, x in nonpadded_input_ids
        #                     if len(x) > max_len]
        # max_len = min(max((len(x[1]) for x in new_input_ids),
        #                     default=0), self.max_seq_length)
        #encode_lst = [torch.cat(l) for l in encode_lst]
        #batch_output.extend(encode_lst)
        return outputs