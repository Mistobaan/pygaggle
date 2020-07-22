from copy import deepcopy
from typing import List
from dataclasses import dataclass

from transformers import (PreTrainedModel,
                          PreTrainedTokenizer,
                          T5ForConditionalGeneration)
import torch

from .base import Reranker, Query, Text
from .similarity import SimilarityMatrixProvider
from pygaggle.model import (BatchTokenizer,
                            LongBatchEncoder,
                            QueryDocumentBatch,
                            QueryDocumentBatchTokenizer,
                            SpecialTokensCleaner,
                            SpacySenticizer,
                            greedy_decode)

from pygaggle.rerank.base import TextType

__all__ = ['T5Reranker',
           'UnsupervisedTransformerReranker',
           'SequenceClassificationTransformerReranker',
           'QuestionAnsweringTransformerReranker']


@dataclass
class SingleEncoderOutput:
    encoder_output: torch.Tensor
    token_ids: torch.Tensor
    text: TextType


class SpecialTokensCleaner:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.special_ids = tokenizer.all_special_ids

    def clean(self, output: SingleEncoderOutput) -> SingleEncoderOutput:
        indices = [idx for idx, tok in enumerate(output.token_ids.tolist())
                   if tok not in self.special_ids]
        
        if not indices:
            # the whole input is a special id...
            return None
        #if len(output.token_ids) != len(output.encoder_output):
        #    pass
        #assert max(indices) < len(output.encoder_output)
        #assert max(indices) < len(output.token_ids)
        # assert len(output.token_ids) == len(output.encoder_output)
        # assert 0 <= min(indices)
        return SingleEncoderOutput(encoder_output=output.encoder_output[indices],
                                   token_ids=output.token_ids[indices], 
                                   text=output.text)


class T5Reranker(Reranker):
    def __init__(self,
                 model: T5ForConditionalGeneration,
                 tokenizer: QueryDocumentBatchTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(self.model.parameters(), None).device

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        batch_input = QueryDocumentBatch(query=query, documents=texts)
        for batch in self.tokenizer.traverse_query_document(batch_input):
            input_ids = batch.output['input_ids'].to(self.device)
            attn_mask = batch.output['attention_mask'].to(self.device)
            _, batch_scores = greedy_decode(self.model,
                                            input_ids,
                                            length=1,
                                            attention_mask=attn_mask,
                                            return_last_logits=True)

            # 6136 and 1176 are the indexes of the tokens false and true in T5.
            batch_scores = batch_scores[:, [6136, 1176]]
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(batch.documents, batch_log_probs):
                doc.score = score
        return texts

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class UnsupervisedTransformerReranker(Reranker):
    methods = dict(max=lambda x: x.max().item(),
                   mean=lambda x: x.mean().item(),
                   absmean=lambda x: x.abs().mean().item(),
                   absmax=lambda x: x.abs().max().item())

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 sim_matrix_provider: SimilarityMatrixProvider,
                 method: str = 'max',
                 clean_special: bool = True,
                 argmax_only: bool = False):
        assert method in self.methods, 'inappropriate scoring method'
        self.model = model
        max_seq_length = self.model.config.max_position_embeddings
        self.tokenizer = tokenizer
        self.batch_encoder = LongBatchEncoder(model, tokenizer, max_seq_length=max_seq_length)
        self.sim_matrix_provider = sim_matrix_provider
        self.method = method
        self.clean_special = clean_special
        self.cleaner = SpecialTokensCleaner(tokenizer.tokenizer)
        self.device = next(self.model.parameters(), None).device
        self.argmax_only = argmax_only

        print(self.tokenizer.__class__)

    def split(self, documents: List[Text]) -> List[Text]:
        batch_size = 16
        senticizer = SpacySenticizer()
        for idx, doc in enumerate(documents):
            document_features = []
            # split document into sentences
            sentences = senticizer(doc.text)
            # encode sentences
            sentences_to_features = self.tokenizer.tokenizer.batch_encode_plus(sentences, max_len=128)
            # max num sentences per doc, max length sentence
            # for b in batch(sentences_to_features, batch_size):
            #     document_features.append((idx, b))
            yield sentences_to_features

    @torch.no_grad()
    def rerank(self, query: Query, documents: List[Text]) -> List[Text]:
        MIN_SCORE = -10_000

        query_features = list(self.split([query]))[0]
        encoded_query = self.batch_encoder.align(query_features['input_ids'])
        encoded_query = encoded_query[:, 0, :]

        result = []
        document_features = []
        for features in self.split(documents):
            output = self.batch_encoder.align(features['input_ids'])
            # output = output.unsqueeze(0) # add batch
            # print(output.shape)
            document_features.append(torch.squeeze(output[:, 0, :]))

        for b in batch(document_features, 128):
            batch_torch = torch.stack(b)
            #print(encoded_query.shape, batch_torch.shape)
            matrix = self.sim_matrix_provider.compute_matrix_v2(encoded_query, batch_torch)
            # batch_scores = self.methods[self.method](matrix)
            if matrix.size(1) == 1:
                result.append(matrix.squeeze().tolist())
            else:
                print(matrix.shape)
                result.extend(matrix.squeeze().tolist())
        # print(result)
        return result

        # import sys;sys.exit()
        encoded_documents = self.batch_encoder.encode(documents)
        documents = deepcopy(documents)
        max_score = None
        for enc_doc, text in zip(encoded_documents, documents):
            if self.clean_special:
                enc_doc = self.cleaner.clean(enc_doc)
                if enc_doc is None:
                    print('invalid enc_doc')
                    continue
                print('after:', enc_doc.shape, text.shape)

            matrix = self.sim_matrix_provider.compute_matrix(encoded_query, enc_doc)
            if matrix.size(1) > 0:
                score = self.methods[self.method](matrix)
            else:
                score = MIN_SCORE
            text.score = score
            max_score = score if max_score is None else max(max_score, score)
        if self.argmax_only:
            for text in documents:
                if text.score != max_score:
                    text.score = max_score - 10_000
        return documents


class SequenceClassificationTransformerReranker(Reranker):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        for text in texts:
            ret = self.tokenizer.encode_plus(query.text,
                                             text.text,
                                             max_length=512,
                                             return_token_type_ids=True,
                                             return_tensors='pt')
            input_ids = ret['input_ids'].to(self.device)
            tt_ids = ret['token_type_ids'].to(self.device)
            output, = self.model(input_ids, token_type_ids=tt_ids)
            if output.size(1) > 1:
                text.score = torch.nn.functional.log_softmax(
                    output, 1)[0, -1].item()
            else:
                text.score = output.item()
        return texts


class QuestionAnsweringTransformerReranker(Reranker):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def rerank(self, query: Query, hits: List[Text]) -> List[Text]:
        hits = deepcopy(hits)
        for text in hits:
            ret = self.tokenizer.encode_plus(query.text,
                                             text.text,
                                             max_length=512,
                                             truncation=True,
                                             return_tensors='pt',
                                             return_token_type_ids=True)
            input_ids = ret['input_ids'].to(self.device)
            tt_ids = ret['token_type_ids'].to(self.device)
            start_scores, end_scores = self.model(input_ids,
                                                  token_type_ids=tt_ids)
            start_scores = start_scores[0]
            end_scores = end_scores[0]
            start_scores[(1 - tt_ids[0]).bool()] = -5000
            end_scores[(1 - tt_ids[0]).bool()] = -5000
            smax_val, smax_idx = start_scores.max(0)
            emax_val, emax_idx = end_scores.max(0)
            text.score = max(smax_val.item(), emax_val.item())
        return hits
