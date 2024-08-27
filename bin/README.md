---
base_model: Alibaba-NLP/gte-multilingual-base
datasets: []
language: []
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
- pearson_manhattan
- spearman_manhattan
- pearson_euclidean
- spearman_euclidean
- pearson_dot
- spearman_dot
- pearson_max
- spearman_max
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:574389
- loss:MultipleNegativesRankingLoss
- loss:CosineSimilarityLoss
widget:
- source_sentence: 에르도안과 푸틴이 목소리를 높여 말했다
  sentences:
  - 소말리아에서 자동차 파업으로 사망한 외국 무장세력
  - 더블 데커 버스가 시내 거리에서 운전한다.
  - 그러나 인종과 같은 다양한 하위 그룹에서 학생들의 부적절한 성과는 연방 정부의 어떤 아동도 행동하지 않고 요구 개선 목록에 주정부를 밀어 넣었습니다.
- source_sentence: 거의 10년 동안, 텔레마케팅 동료들은 7백만 달러 이상을 모금했다.
  sentences:
  - 베트남이 고용한 텔레마케팅 동료들은 1987년부터 1995년까지 710만 달러를 모금했다.
  - 남자가 하프를 연주하고 있다.
  - 그 관리는 익명을 전제로 말했다.
- source_sentence: 그들의 자연 음식은 물고기가 아닙니다.
  sentences:
  - 그들은 어떤 이유에서인지 BBQ 소스 냄새를 좋아한다.
  - 사실, 여러분은 물고기가 그들의 자연 음식이라는 것을 알고 있습니다.
  - 그래, 사실 넌 물고기가 그들의 자연 음식이 아니라는 걸 알잖아. 그게 바로
- source_sentence: 여자들은 아시아인이다.
  sentences:
  - 여자들은 부엌 안에 있다.
  - 짧은 개입은 알코올 섭취의 약간의 변화로 이어질 뿐이기 때문에, 아마도 연구는 주요 결과로서 재상해나 건강 관리 사용에 초점을 맞춰야 할 것이다.
  - 길모퉁이에 앉아 있는 두 명의 아시아 여성, 하얀 차가 지나가는 동안.
- source_sentence: 내가 가서 질문할게.
  sentences:
  - 우리는 아침 동안 그것이 일어날 마을의 홀을 준비하고 꾸미느라 바빴다.
  - 조용히 여기 앉아 있을게.
  - '"내가 여기저기 물어볼게.'
model-index:
- name: SentenceTransformer based on Alibaba-NLP/gte-multilingual-base
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: sts dev
      type: sts-dev
    metrics:
    - type: pearson_cosine
      value: 0.8610279793025273
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.8621343498654654
      name: Spearman Cosine
    - type: pearson_manhattan
      value: 0.805701531243174
      name: Pearson Manhattan
    - type: spearman_manhattan
      value: 0.8088204675990345
      name: Spearman Manhattan
    - type: pearson_euclidean
      value: 0.8076656960386089
      name: Pearson Euclidean
    - type: spearman_euclidean
      value: 0.8109381416564464
      name: Spearman Euclidean
    - type: pearson_dot
      value: 0.7750752150785605
      name: Pearson Dot
    - type: spearman_dot
      value: 0.7936631482492172
      name: Spearman Dot
    - type: pearson_max
      value: 0.8610279793025273
      name: Pearson Max
    - type: spearman_max
      value: 0.8621343498654654
      name: Spearman Max
---

# SentenceTransformer based on Alibaba-NLP/gte-multilingual-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) <!-- at revision f7d567e1f2493bb0df9413965d144de9f15e7bab -->
- **Maximum Sequence Length:** 8192 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 8192, 'do_lower_case': False}) with Transformer model: NewModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '내가 가서 질문할게.',
    '"내가 여기저기 물어볼게.',
    '조용히 여기 앉아 있을게.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity
* Dataset: `sts-dev`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric             | Value      |
|:-------------------|:-----------|
| pearson_cosine     | 0.861      |
| spearman_cosine    | 0.8621     |
| pearson_manhattan  | 0.8057     |
| spearman_manhattan | 0.8088     |
| pearson_euclidean  | 0.8077     |
| spearman_euclidean | 0.8109     |
| pearson_dot        | 0.7751     |
| spearman_dot       | 0.7937     |
| pearson_max        | 0.861      |
| **spearman_max**   | **0.8621** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Datasets

#### Unnamed Dataset


* Size: 568,640 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         | sentence_2                                                                        |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                            |
  | details | <ul><li>min: 4 tokens</li><li>mean: 19.7 tokens</li><li>max: 220 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 18.95 tokens</li><li>max: 190 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 14.54 tokens</li><li>max: 42 tokens</li></ul> |
* Samples:
  | sentence_0                                   | sentence_1                                                                                   | sentence_2                                       |
  |:---------------------------------------------|:---------------------------------------------------------------------------------------------|:-------------------------------------------------|
  | <code>관리 관리가 미국의 건강관리를 악화시켰다는 증거는 없다.</code> | <code>미국의 건강 관리가 실제로 관리되는 관리 하에서 악화되었다는 확실한 증거가 존재하지 않는다는 것은 신경 쓰지 마라.</code>                | <code>미국의 건강관리가 관리된 관리로 고통받았다는 증거가 있다.</code>    |
  | <code>우리는 여기에 많은 물고기를 가지고 있다.</code>         | <code>네가 자연스럽게 가질 수 있는 것처럼, 우리는 여기 많은 물고기들과 조개류들을 가지고 있고, 왜냐하면 우리는 바로 해안에 살고 있기 때문이다.</code> | <code>여기서는 물고기를 잡을 수 없다.</code>                  |
  | <code>파란 셔츠를 입은 작은 소녀가 혼자 걷고 있다.</code>      | <code>파란색 셔츠를 입은 작은 소녀가 노란 주차장에서 혼자 걷고 있다.</code>                                            | <code>빨간 셔츠를 입은 작은 소녀가 노란 주차장에서 혼자 걷고 있다.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

#### Unnamed Dataset


* Size: 5,749 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                          |
  | details | <ul><li>min: 6 tokens</li><li>mean: 19.31 tokens</li><li>max: 67 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 19.32 tokens</li><li>max: 57 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.54</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                             | sentence_1                                            | label                           |
  |:---------------------------------------|:------------------------------------------------------|:--------------------------------|
  | <code>남자가 토르티야를 튀기고 있다.</code>         | <code>사람이 카드 트릭을 하고 있다.</code>                        | <code>0.0</code>                |
  | <code>개가 물을 떨쳐내고 있다.</code>            | <code>개가 물을 떨쳐내고 있다.</code>                           | <code>1.0</code>                |
  | <code>작은 흰 개와 큰 갈색 개는 풀밭에서 놀아요.</code> | <code>작은 햇볕에 그을린 개가 풀밭에 서 있는 커다란 갈색 개를 지나고 있다.</code> | <code>0.6799999999999999</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `num_train_epochs`: 10
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | sts-dev_spearman_max |
|:------:|:----:|:-------------:|:--------------------:|
| 0.3477 | 500  | 0.3862        | -                    |
| 0.6954 | 1000 | 0.229         | 0.8588               |
| 1.0007 | 1439 | -             | 0.8621               |


### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.0.1
- Transformers: 4.42.4
- PyTorch: 2.4.0+cu121
- Accelerate: 0.32.1
- Datasets: 2.21.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply}, 
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->