# SRL-ICL: 인컨텍스트 학습에서의 최적화된 예제 선택 및 순서 재배치를 통한 의미역 결정 성능 향상

이 저장소는 논문에서 설명한 실험을 재현하기 위한 소스코드를 제공합니다. 대규모 언어 모델(LLM)의 인컨텍스트 학습(ICL)을 활용하여 의미역 결정(SRL)을 수행하며, 최적화된 예제 선택 및 순서 배치 전략을 적용합니다.

## 파이프라인 개요

```
[학습 데이터] ──► [BERT-CRF 학습] ──► [인코더 체크포인트]
                                              │
[학습 데이터] ──► [인코딩 & DB 구축] ◄─────────┘
                        │
[테스트 데이터] ──► [예제 선택 (Top-K / MMR)] ◄──┘
                        │
                 [ConE 순서 최적화] (선택적)
                        │
                 [LLM 추론 & 평가]
```

**Step 1.** BERT-CRF 모델을 학습하여 검색용 문장 인코더로 사용합니다.
**Step 2.** 모든 학습 예제를 인코딩하여 벡터 DB를 구축합니다.
**Step 3.** 각 테스트 인스턴스에 대해 Top-K 또는 MMR로 유사 예제를 검색합니다.
**Step 4.** (선택적) ConE (조건부 엔트로피)로 최적 예제 순서를 탐색합니다.
**Step 5.** 최적화된 프롬프트로 LLM 추론을 실행하고 F1을 측정합니다.


## 요구사항

- Python 3.10+
- CUDA 지원 NVIDIA GPU
- 27B 모델: ~48GB VRAM (9B 모델 4-bit 양자화: ~16GB)

```bash
pip install -r requirements.txt
```

## 데이터 준비

라이센스 제한으로 분할별 10개 샘플 문장만 포함되어 있습니다. 전체 데이터셋은 별도로 취득해야 합니다:

- **영어**: CoNLL 2009 Shared Task 데이터 ([LDC](https://www.ldc.upenn.edu/))
- **한국어**: Korean PropBank (데이터셋 저자에게 문의)

데이터 파일을 `configs/en_config.yaml` 또는 `configs/ko_config.yaml`에 명시된 경로에 배치하세요.

형식에 대한 자세한 내용은 `data/README.md`를 참고하세요.

## 빠른 시작

```bash
# ConE 순서 최적화 포함 전체 파이프라인 실행 (영어)
bash scripts/run_all.sh --config configs/en_config.yaml

# ConE 없이 고정 순서 [0,1,2,3,4]로 실행
bash scripts/run_without_cone.sh --config configs/en_config.yaml

# 사용자 정의 순서로 실행
bash scripts/run_without_cone.sh --config configs/en_config.yaml --order 2,0,4,1,3

# GPU 디바이스 지정 (예: GPU 1번 사용)
bash scripts/run_all.sh --config configs/en_config.yaml --gpu 1
```

## 단계별 재현

```bash
# Step 1: BERT-CRF 인코더 학습 (개발셋으로 early stopping)
python scripts/01_train_crf.py --config configs/en_config.yaml

# Step 2: 검색 DB 구축
python scripts/02_build_retrieval_db.py --config configs/en_config.yaml

# Step 3: 예제 선택 (Top-K 또는 MMR)
python scripts/03_select_examples.py --config configs/en_config.yaml --strategy topk
python scripts/03_select_examples.py --config configs/en_config.yaml --strategy mmr --lambda_param 0.9

# Step 4: ConE로 예제 순서 최적화 (cone_llm 사용)
python scripts/04_optimize_order.py --config configs/en_config.yaml

# Step 5: 평가 (eval_llm 사용, Step 4의 순서 자동 로드)
python scripts/05_evaluate.py --config configs/en_config.yaml
# 또는 순서를 직접 지정:
python scripts/05_evaluate.py --config configs/en_config.yaml --order 2,1,3,4,0

# Step 6: 레이턴시 벤치마크
python scripts/06_latency_benchmark.py --config configs/en_config.yaml --mode breakdown --sample_size 100
```

## 추론 (단일 문장)

```python
from inference.pipeline import SRLPipeline

pipe = SRLPipeline.from_config("configs/ko_config.yaml")

result = pipe.predict(
    sentence="한국탁구가 2000년 시드니올림픽 본선에 남녀복식 2개조씩을 파견할 수 있게 됐다.",
    predicate="파견.01",          # 술어 의미번호 (framefile 검색용)
    predicate_index=7,            # 0-based 어절 위치
    output_format="conll",        # "conll" 또는 "dict"
    verbose=True,                 # 단계별 소요 시간 출력
)

print(result["prediction"])
```

**참고**: 술어 인식(predicate identification)은 이미 완료되었다고 가정합니다. 입력에는 동사 의미번호(예: `파견.01`)와 문장 내 위치가 필요합니다.

## 프로젝트 구조

```
SRL-ICL/
├── models/          # BERT-CRF 아키텍처 (인코더, CRF 레이어, BiLSTM)
├── retrieval/       # 예제 검색 (Euclidean/Mahalanobis, CRF/pretrained 인코더)
├── ordering/        # ConE 순서 최적화
├── prompts/         # 프롬프트 템플릿 (EN/KO × CoNLL/Dict) 및 빌더
├── evaluation/      # 평가 메트릭 (micro-F1) 및 포맷 변환
├── inference/       # End-to-End 추론 파이프라인
├── utils/           # GPU 설정 및 공유 유틸리티
├── scripts/         # 단계별 재현 스크립트 + 셸 러너
├── data/            # 샘플 데이터 및 형식 문서
├── configs/         # YAML 설정 파일
├── docs/            # 의사코드 및 파이프라인 다이어그램
├── requirements.txt
└── LICENSE
```

## 설정

모든 하이퍼파라미터와 파일 경로는 `configs/`의 YAML 파일로 관리합니다. 주요 설정:

| 파라미터 | 설명 | 기본값 |
|---|---|---|
| `language` | `"en"` 또는 `"ko"` | — |
| `output_format` | `"conll"` 또는 `"dict"` | — |
| `gpu_id` | CUDA 디바이스 번호 | `0` |
| `cone_llm.model_id` | ConE 순서 최적화용 LLM (소형) | `"google/gemma-2-9b-it"` |
| `eval_llm.model_id` | 최종 평가용 LLM (대형) | `"google/gemma-2-27b-it"` |
| `retrieval.encoder_type` | `"crf"` 또는 `"pretrained"` | `"crf"` |
| `retrieval.metric` | `"euclidean"` 또는 `"mahalanobis"` | `"euclidean"` |
| `retrieval.strategy` | `"topk"` 또는 `"mmr"` | `"topk"` |
| `cone.num_examples` | 순서를 최적화할 예제 수 (k) | `5` |
| `crf_training.dev_file` | Early stopping용 개발셋 | — |

## 실험 환경

- **OS**: Ubuntu 22.04
- **GPU**: NVIDIA RTX A6000 (48GB)
- **프레임워크**: PyTorch 2.7.1, Transformers 4.40+

