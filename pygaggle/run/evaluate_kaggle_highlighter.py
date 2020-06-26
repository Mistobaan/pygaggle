from transformers import BertPreTrainedModel, BertModel
from torch import nn
from typing import Optional, List
from pathlib import Path
import logging

from pydantic import BaseModel, validator
from transformers import (AutoModel,
                          AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          # BertForQuestionAnswering,
                          PretrainedConfig,
                          BertConfig,
                          BertForSequenceClassification)
import torch

from .args import ArgumentParserBuilder, opt
from pygaggle.rerank.base import Reranker
from pygaggle.rerank.bm25 import Bm25Reranker
from pygaggle.rerank.transformer import (
    QuestionAnsweringTransformerReranker,
    SequenceClassificationTransformerReranker,
    T5Reranker,
    UnsupervisedTransformerReranker
)
from pygaggle.rerank.random import RandomReranker
from pygaggle.rerank.similarity import CosineSimilarityMatrixProvider
from pygaggle.model import (CachedT5ModelLoader,
                            RerankerEvaluator,
                            SimpleBatchTokenizer,
                            T5BatchTokenizer,
                            metric_names)
from pygaggle.data import LitReviewDataset
from pygaggle.settings import Cord19Settings

from torch.nn import CrossEntropyLoss, MSELoss

SETTINGS = Cord19Settings()
METHOD_CHOICES = ('transformer', 'bm25', 't5', 'seq_class_transformer',
                  'qa_transformer', 'random')
# Patching
# from: https://github.com/huggingface/transformers/issues/1619


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return (start_logits,
                end_logits)  # (hidden_states), (attentions)


class KaggleEvaluationOptions(BaseModel):
    dataset: Path
    index_dir: Path
    method: str
    batch_size: int
    device: str
    split: str
    do_lower_case: bool
    metrics: List[str]
    model_name: Optional[str]
    tokenizer_name: Optional[str]

    @validator('dataset')
    def dataset_exists(cls, v: Path):
        assert v.exists(), 'dataset must exist'
        return v

    @validator('model_name')
    def model_name_sane(cls, v: Optional[str], values, **kwargs):
        method = values['method']
        if method == 'transformer' and v is None:
            raise ValueError('transformer name must be specified')
        elif method == 't5':
            return SETTINGS.t5_model_type
        if v == 'biobert':
            return 'monologg/biobert_v1.1_pubmed'
        return v

    @validator('tokenizer_name')
    def tokenizer_sane(cls, v: str, values, **kwargs):
        if v is None:
            return values['model_name']
        return v


def construct_t5(options: KaggleEvaluationOptions) -> Reranker:
    loader = CachedT5ModelLoader(SETTINGS.t5_model_dir,
                                 SETTINGS.cache_dir,
                                 'ranker',
                                 SETTINGS.t5_model_type,
                                 SETTINGS.flush_cache)
    device = torch.device(options.device)
    model = loader.load().to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        options.model_name, do_lower_case=options.do_lower_case)
    tokenizer = T5BatchTokenizer(tokenizer, options.batch_size)
    return T5Reranker(model, tokenizer)


def construct_transformer(options: KaggleEvaluationOptions) -> Reranker:
    device = torch.device(options.device)
    try:
        model = AutoModel.from_pretrained(options.model_name).to(device).eval()
    except OSError:
        model = AutoModel.from_pretrained(options.model_name,
                                          from_tf=True).to(device).eval()
    tokenizer = SimpleBatchTokenizer(
        AutoTokenizer.from_pretrained(
            options.tokenizer_name, do_lower_case=options.do_lower_case),
        options.batch_size)
    provider = CosineSimilarityMatrixProvider()
    return UnsupervisedTransformerReranker(model, tokenizer, provider)


def construct_seq_class_transformer(options:
                                    KaggleEvaluationOptions) -> Reranker:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            options.model_name)
    except OSError:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                options.model_name,
                from_tf=True)
        except AttributeError:
            # Hotfix for BioBERT MS MARCO. Refactor.
            BertForSequenceClassification.bias = torch.nn.Parameter(
                torch.zeros(2))
            BertForSequenceClassification.weight = torch.nn.Parameter(
                torch.zeros(2, 768))
            model = BertForSequenceClassification.from_pretrained(
                options.model_name, from_tf=True)
            model.classifier.weight = BertForSequenceClassification.weight
            model.classifier.bias = BertForSequenceClassification.bias
    device = torch.device(options.device)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        options.tokenizer_name, do_lower_case=options.do_lower_case)
    return SequenceClassificationTransformerReranker(model, tokenizer)


def construct_qa_transformer(options: KaggleEvaluationOptions) -> Reranker:
    # We load a sequence classification model first -- again, as a workaround.
    # Refactor
    try:
        logging.info('Loading: %s', options.model_name)
        if 't5' in options.model_name:
            model = AutoModelForQuestionAnswering.from_pretrained(
                options.model_name)
        else:
            print(options.model_name)
            config_dict, _ = PretrainedConfig.get_config_dict(options.model_name)
            config = BertConfig.from_dict(config_dict)
            model = BertForQuestionAnswering(config)
    except OSError:
        model = AutoModelForQuestionAnswering.from_pretrained(
            options.model_name, from_tf=True)
    device = torch.device(options.device)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        options.tokenizer_name, do_lower_case=options.do_lower_case)
    return QuestionAnsweringTransformerReranker(model, tokenizer)


def construct_bm25(options: KaggleEvaluationOptions) -> Reranker:
    return Bm25Reranker(index_path=str(options.index_dir))


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(opt('--dataset', type=Path, required=True),
                 opt('--index-dir', type=Path, required=True),
                 opt('--method',
                     required=True,
                     type=str,
                     choices=METHOD_CHOICES),
                 opt('--model-name', type=str),
                 opt('--split', type=str, default='nq', choices=('nq', 'kq')),
                 opt('--batch-size', '-bsz', type=int, default=96),
                 opt('--device', type=str, default='cuda:0'),
                 opt('--tokenizer-name', type=str),
                 opt('--do-lower-case', action='store_true'),
                 opt('--metrics',
                     type=str,
                     nargs='+',
                     default=metric_names(),
                     choices=metric_names()))
    args = apb.parser.parse_args()
    options = KaggleEvaluationOptions(**vars(args))
    ds = LitReviewDataset.from_file(str(options.dataset))
    examples = ds.to_senticized_dataset(str(options.index_dir),
                                        split=options.split)
    construct_map = dict(transformer=construct_transformer,
                         bm25=construct_bm25,
                         t5=construct_t5,
                         seq_class_transformer=construct_seq_class_transformer,
                         qa_transformer=construct_qa_transformer,
                         random=lambda _: RandomReranker())
    reranker = construct_map[options.method](options)
    evaluator = RerankerEvaluator(reranker, options.metrics)
    width = max(map(len, args.metrics)) + 1
    stdout = []
    import time
    start = time.time()
    with open(f'{options.model_name}.csv', 'w') as fd:
        logging.info('writing %s.csv', options.model_name)
        for metric in evaluator.evaluate(examples):
            logging.info(f'{metric.name:<{width}}{metric.value:.5}')
            stdout.append(f'{metric.name}\t{metric.value:.3}')
            fd.write(f"{metric.name}\t{metric.value:.3}\n")
        end = time.time()
        fd.write(f"time\t{end-start:.3}\n")

    print('\n'.join(stdout))


if __name__ == '__main__':
    main()
