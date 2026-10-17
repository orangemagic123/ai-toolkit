# NoRA 구현 설계안 (init + continuous)

작성일: 2026-10-17. 상태: 백엔드(7절의 1–3)와 UI 구현, CPU·CUDA 단위 검증과 UI 동작 확인 완료. A/B 실험은 미착수. 이전의 init 단독 초안을 대체한다.

LoRA의 down projection A(`lora_down.weight`, shape `[rank, input_dim]`)를 rank 차원으로 정규화하는 NoRA(arXiv 2608.31036)를 두 모드로 추가한다. 기본값은 꺼짐이며, 꺼짐 상태의 동작·난수 소비·저장 형식은 바뀌지 않는다.

| 모드 | 초기화 | forward | 저장 |
|---|---|---|---|
| `init` (NoRA-init) | 모듈 기본 초기화 후 A의 각 열을 한 번 정규화 | 일반 LoRA (`B·A`) | 일반 LoRA |
| `continuous` (NoRA) | 원본 A를 **kaiming uniform** 그대로 둔다 (LoRA·DoRA 공통) | `B·Norm(A)`, gradient가 정규화를 통과 | Norm(A)를 `lora_down`에 굳혀 일반 LoRA 형식으로 기록 |

```text
Norm(A)[:, j] = A[:, j] / max(||A[:, j]||_2, 1e-6)      # j = 입력 좌표, 노름은 rank 축
Conv: A를 [rank, c_in*kh*kw]로 펼쳐 같은 연산 후 원래 shape으로 되돌린다
```

## 1. 저자 구현과의 대응, 그리고 continuous의 초기화 원칙

- 저자 PEFT fork(`Joluck/NoRA`)의 `use_nora="init"`은 초기화 직후 원본 A를 정규화하고, `use_nora=True`는 원본 A를 kaiming 그대로 둔 채 forward에서만 정규화한다. 이 설계의 두 모드는 각각에 1:1로 대응한다.
- **continuous에서는 원본 A를 미리 정규화하지 않는다.** forward 출력은 원본 A의 열 크기와 무관하지만, AdamW의 원소 단위 step이 원본 크기에 대해 상대적으로 작용하므로 원본 A가 작을수록 A의 방향이 빠르게 학습된다. 원본 크기가 학습 동역학을 결정하므로 논문 조건(kaiming, 열 길이 ≈ √(r/3k))을 그대로 따른다.
- **DoRA도 continuous에서는 원본 A를 kaiming으로 초기화한다.** 이 저장소의 `DoRAModule` 기본 초기화는 `randn/√r`(열 길이 ≈ 1)이라 그대로 두면 A의 방향 학습 속도가 논문 NoRA-DoRA 대비 약 √(3k/r)배 느려진다. continuous는 새 opt-in 모드이므로 기존 DoRA 기본값에는 영향이 없다.
- `init` 모드는 각 모듈의 기존 초기화(LoRA: kaiming, DoRA: `randn/√r`) 뒤에 정규화만 한다. 정규화 후 방향 분포는 초기 분포와 거의 무관하므로(논문 표 3) 기본 난수 경로를 유지한다. 같은 seed에서 `init`의 A는 기본 A를 정규화한 값과 정확히 같아 짝 비교에 쓸 수 있다.
- 이 저장소 DoRA의 기본 A는 이미 NoRA 크기(열 길이 ≈ 1)이므로 **DoRA + `init`의 효과는 작을 것으로 예상**한다. 지원은 하되 문서·로그에서 효과를 약속하지 않는다.

## 2. 설정과 지원 범위

```yaml
network:
  type: lora
  nora: continuous      # 생략 | false | "off" | init | continuous
  linear: 32
  linear_alpha: 32
  use_dora: false
```

- `NetworkConfig.nora`로 파싱한다. 생략, `false`, `"off"`는 꺼짐이다. config 로더가 `yaml.SafeLoader`라서 따옴표 없는 `off`는 `False`로 읽히기 때문에 둘 다 꺼짐으로 받는다.
- `true`는 모드가 모호하므로 오류로 처리한다(`init` 또는 `continuous`를 쓰라고 안내). 그 밖의 값과 오타도 오류다.
- `network_kwargs.nora`는 `LoRASpecialNetwork(**kwargs)`로 흘러가 조용히 무시되므로, 발견하면 올바른 위치를 안내하는 오류를 낸다.
- 지원: `type: lora`(레거시 `type: dora` 별칭 포함), `use_dora` true/false, Linear와 Conv 어댑터. DoRA Conv는 기존 네트워크가 이미 건너뛴다.
- 오류: `lokr`, `locon`/`lycoris`(LycorisSpecialNetwork), `lorm`, `fullrank`. LoKr full-rank 기본 구성에는 별도의 down projection이 없다.
- FullModule(`full_if_contains`, `all_layers`)은 LoRA 분해가 아니므로 적용 대상이 아니며, 로그에 미적용 개수만 표시한다.
- 지원 대상 LoRA 모듈이 0개면 학습 전에 오류를 낸다.
- ARA 네트워크는 자체 `NetworkConfig`를 만들기 때문에 영향을 받지 않는다. 보조 어댑터(control LoRA, i2v, mean_flow, subpixel)는 같은 `get_state_dict`/`load_weights` 경로를 써서 구조상 동작하지만 v1의 실기 검증 범위에는 넣지 않는다.
- 양자화 base도 지원한다. 정규화 대상은 base weight가 아니라 부동소수점 LoRA A다.

## 3. 구현 구조

### 3.1 공통 함수: `toolkit/nora.py` (신규)

- `parse_nora_mode(value) -> "off" | "init" | "continuous"`: 2절의 검증 규칙.
- `normalize_down_weight(weight, eps=1e-6) -> Tensor`: 2D와 4D를 처리한다. 노름은 FP32로 계산하고 원래 dtype으로 반환한다. autograd를 끊지 않는다(continuous의 forward에서 사용).
- `apply_nora_init_(weight)`: `torch.no_grad()`에서 위 함수 결과를 `copy_`한다. Parameter 객체, device, dtype을 보존한다.

### 3.2 모듈

- `LoRAModule`, `DoRAModule`이 `nora_mode="off"` 인자를 받는다. 네트워크는 `module_kwargs`로 명시 전달한다. 기본값이 `off`이므로 테스트, ARA 등 직접 생성 경로는 영향을 받지 않는다.
- `LoRAModule.__init__`: 기존 kaiming 초기화 직후 `init`이면 `apply_nora_init_`를 적용한다. `continuous`는 초기화를 바꾸지 않는다(이미 kaiming).
- `DoRAModule.__init__`: `continuous`면 A를 `kaiming_uniform_(a=√5)`로 초기화하고, 아니면 기존 `randn/√r`를 쓴다. `init`이면 그 뒤 정규화한다. magnitude는 B=0이라 ‖W‖이므로 순서와 무관하다.
- `ToolkitModuleMixin`에 두 helper를 둔다.
  - `_get_lora_down_weight()`: continuous면 `normalize_down_weight(self.lora_down.weight)`, 아니면 원본 weight.
  - `_lora_down_forward(x)`: continuous면 `F.linear(x, Wn)` 또는 `F.conv2d(x, Wn, None, stride, padding, dilation, groups)`, 아니면 `self.lora_down(x)`.
  - 다른 모듈 클래스(LoKr 등)는 `getattr(self, "nora_mode", "off")`로 기존 경로를 탄다.

### 3.3 Norm(A)를 써야 하는 지점 (continuous)

| 위치 | 변경 |
|---|---|
| `ToolkitModuleMixin._call_forward` | `self.lora_down(x)` → `self._lora_down_forward(x)`. LoRA와 DoRA가 모두 이 경로를 지난다. dropout·rank dropout 위치는 기존대로 down 출력 뒤에 둔다. |
| `DoRAModule._get_weight_norm_factorized` | A를 직접 읽는 두 곳(`F.linear(weight, A)`, `gram`)에 정규화 A를 한 번만 계산해 쓴다. 기존처럼 `no_grad`(PEFT DoRA와 같은 detach). |
| `DoRAModule.apply_dora`, mixin forward의 DoRA 분기 | 현재 쓰이지 않는 경로지만 일관성을 위해 `_get_lora_down_weight()`로 바꾼다. |
| `ToolkitModuleMixin.merge_in` | `down_weight`를 `_get_lora_down_weight()`에서 가져온다. Linear, 1×1, 3×3 conv 분기는 그대로다. |
| `ToolkitNetworkMixin.get_state_dict` | `self.state_dict()` 직후 continuous 모듈의 `f"{lora_name}.lora_down.weight"`를 정규화 A로 교체한다(키 존재를 assert). 이후 PEFT 키 변환, dtype 변환, `convert_lora_weights_before_save`는 기존대로다. 보조 어댑터도 이 함수를 거친다. |

추가하지 않는 것: optimizer parameter group, 학습률 보정, 별도 scaler, 새 저장 텐서, 특수 추론 연산. `reset_weights`(merge_network_on_save)는 모듈 `state_dict()`로 `lora_up`만 0으로 만들므로 원본 A가 유지되며 변경이 필요 없다.

### 3.4 저장·로드·재개

| 상황 | 동작 |
|---|---|
| 저장된 어댑터 | 일반 LoRA(DoRA) 형식이다. continuous는 A가 정규화된 상태로 기록된다. NoRA를 모르는 loader(ComfyUI, `inference_lora.py`)도 같은 출력을 낸다. |
| 짝 학습 상태가 있는 재개 | `load_weights`가 정규화 A를 원본 자리에 넣은 뒤, `restore_training_parameters`가 원본 A(정규화 전 크기)와 optimizer 상태를 그대로 복원한다. 끊김 없이 이어진다. |
| 짝 학습 상태 없이 재개(레거시, 손상, rank 변경 등) | 저장된 정규화 A(열 길이 1)에서 이어진다. forward는 같지만 A의 방향 학습 속도가 원래보다 느려진다. continuous에서 `restore_paired_training_state`가 None을 반환하면 호출부에서 이 사실을 경고한다(이 메서드는 기존 테스트가 단독으로 떼어 호출하므로 본문은 바꾸지 않는다). |
| 모드를 바꿔 재개(continuous ↔ off·init) | 저장할 때 체크포인트 metadata의 `training_info.nora_mode`에 모드를 기록한다. `load_training_state_from_metadata`가 짝 학습 상태와 `optimizer.pt`를 복원하기 전에 비교해서 오류로 중단하고, 새 job 이름으로 시작하거나 `pretrained_lora_path`로 가중치만 이어받으라고 안내한다. continuous의 짝 상태에 든 원본 A는 일반 LoRA에서 다른 뜻이 되고, 반대로 일반 LoRA의 A는 continuous에서 출력이 바뀌기 때문이다. `init ↔ off`는 A를 같은 방식으로 해석하므로 허용한다. 기록이 없는 체크포인트는 `off`로 본다. `pretrained_lora_path` 로드와 `merge_network_on_save`는 재개가 아니므로 검사하지 않는다. |
| 일반 LoRA를 `pretrained_lora_path`로 불러와 continuous 학습 | 정규화 때문에 step 0 출력이 원래 LoRA와 달라진다. 로드 후 A의 열 길이가 1에서 벗어나 있으면(bf16 저장 오차를 넘는 편차) 한 번 경고한다. 출처 판별용 metadata는 추가하지 않는다. |
| `init`으로 학습한 어댑터 로드 | 일반 LoRA와 같다. 로드한 텐서를 다시 정규화하지 않는다. |
| rank 확장·축소 로드 | 기존 정책을 유지한다. continuous에서 축소는 열 길이가 바뀌므로 출력이 달라질 수 있으며, 이는 기존 LoRA 축소와 같은 성격이다. |
| EMA | 원본 파라미터의 EMA를 유지하고, 사용 시 forward가 정규화한다. 저장 시에는 EMA 원본을 정규화해 기록한다. |

`training_config` metadata에 raw config가 저장되므로 `network.nora`도 그 경로로 남는다. 별도 checkpoint 버전은 만들지 않는다.

## 4. alpha·학습률 정책과 로그

- 예제와 문서는 `linear_alpha = linear`(scale 1)를 권장한다. transformer 계열(PEFT 형식) 경로는 이미 alpha를 rank로 강제하므로 경고는 실제 생성된 모듈의 scale이 1이 아닐 때만 네트워크당 한 번 출력한다.
- 학습률은 자동으로 바꾸지 않는다. 대신 크기를 문서와 로그에 수치로 밝힌다.
  - (해석) AdamW는 gradient 크기를 정규화하므로, forward에 쓰이는 A가 기본 kaiming보다 c배 크면 B의 실질 학습률이 약 c배가 된다. 기본 LoRA 대비 c ≈ √(3k/r)이고, k는 입력 차원이다(conv는 c_in·kh·kw).
  - `init`은 추가로 A 자체의 상대 학습률이 약 1/c이 된다. `continuous`는 원본 A가 kaiming 크기라 A의 방향 학습이 기본 속도를 유지한다.
- 네트워크 생성 시 한 번 요약 로그를 낸다: 모드, 적용된 Linear/Conv 모듈 수, 미적용 FullModule 수, c의 min/median/max, DoRA continuous라면 "원본 A 초기화: kaiming(논문 기준, 기존 DoRA 기본값과 다름)".
- 문서 권장: 기존 최적 학습률을 그대로 최적값으로 간주하지 않는다. NoRA는 기존 lr의 1×, 1/3, 1/10을, 공정 비교를 위해 일반 LoRA는 1×, 3×, 10×를 탐색한다.

## 5. 수정 파일

| 파일 | 변경 |
|---|---|
| `toolkit/nora.py` (신규) | 파싱, 정규화, init 함수 |
| `toolkit/config_modules.py` | `NetworkConfig.nora` 파싱, 조합 검증, `network_kwargs.nora` 거부 |
| `toolkit/lora_special.py` | `nora_mode` 전달, LoRAModule init, 지원 개수 0 오류, 요약 로그 |
| `toolkit/models/DoRA.py` | continuous kaiming 초기화, init 정규화, weight norm과 `apply_dora`의 정규화 A |
| `toolkit/network_mixins.py` | helper 두 개, `_call_forward`, `merge_in`, DoRA 분기, `get_state_dict` 굳히기, 로드 후 열 길이 경고 |
| `jobs/process/BaseSDTrainProcess.py` | continuous에서 짝 학습 상태 없이 재개할 때의 경고(`nora_resume_warning`) |
| `testing/test_nora.py` (신규) | 6절 |
| `config/examples/train_lora_anima_29b_nora.yaml` | NoRA 예제. 기존 예제는 유지 |
| UI(`ui/src/types.ts`, `SimpleJob.tsx`, `docs.tsx`) | Target 카드에 `NoRA` 선택(Off / NoRA-init / NoRA (continuous))을 둔다. LoRA일 때만 보이고, LoKr로 바꾸면 `off`로 정리한다. 실험 기능 표시가 붙은 도움말을 단다. 값이 없으면 Off로 표시하므로 `jobConfig.ts`의 기본값과 migration은 바꾸지 않는다 |

## 6. 검증

CPU와 작은 모듈로 시작한다. 실행 환경은 sd-scripts venv + PYTHONPATH overlay를 쓴다. 기존 환경 이슈(LoCon compile 케이스 등)와 신규 실패를 구분한다.

1. **설정**: 생략, `false`, `"off"`, 따옴표 없는 YAML `off` → 꺼짐. `init`/`continuous` 통과. `true`, 오타, `network_kwargs.nora`, `lokr`/`locon`/`lorm`/`fullrank` 조합은 오류.
2. **꺼짐 호환**: 같은 seed에서 옵션 생략과 `off`의 A/B, 파라미터 수, 난수 상태가 기존과 동일하다.
3. **init**: Linear와 Conv에서 A의 각 열 노름이 1, B가 0, 초기 출력이 base와 같다. 같은 seed에서 A가 기본 A의 정규화 결과와 같다. 정규화 축은 비정방 행렬로 확인한다.
4. **continuous LoRA**: 원본 A가 `off`와 동일한 kaiming 값이다. 출력이 `W0x + s·B·Norm(A)x` 기준식과 일치한다. 원본 A 열을 상수배해도 출력이 불변이다. A의 gradient가 각 열과 직교한다(Σ_r a_rj·g_rj ≈ 0). Conv 3×3/1×1도 확인한다.
5. **continuous DoRA**: 원본 A가 kaiming 범위(|a| ≤ 1/√k)에 있다. `_get_weight_norm_factorized`가 명시적 ‖W + s·B·Norm(A)‖와 일치한다. forward가 정규화 A를 쓴 DoRA 기준식과 일치한다(fp32와 bf16 autocast 모두).
6. **저장·로드**: 원본 A 크기를 임의로 바꾼 뒤 저장하면 `lora_down` 열 노름이 1이다. 그 파일을 `off` 네트워크와 `continuous` 네트워크에 각각 로드해도 출력이 같다. PEFT 키 경로(transformer)와 kohya 키 경로를 모두 확인한다.
7. **merge_in**: 병합된 weight가 `W + s·B·Norm(A)`이고 병합 전 forward와 출력이 같다.
8. **재개**: 짝 학습 상태 복원 후 원본 A(노름 ≠ 1)가 정확히 복원되고, 다음 optimizer step이 끊지 않은 경로와 일치한다. 학습 상태가 없으면 경고가 출력된다. 일반 LoRA를 pretrained로 로드하면 열 길이 경고가 출력된다. continuous ↔ off·init으로 모드를 바꿔 재개하면 상태를 복원하기 전에 오류가 나고, `init ↔ off`와 같은 모드끼리의 재개는 출력이 보존된다. pretrained 로드와 merge 경로는 검사하지 않는다.
9. **compile·정밀도**: `test_lora_compile_scalars.py` 방식으로 continuous 모듈의 compile을 확인한다. bf16에서 NaN/Inf가 없고 epsilon 경계(영벡터 열)도 안전하다.
10. **회귀**: `test_lora_mixed_precision.py`, `test_lora_compile_scalars.py`, `test_dora_fast_forward.py`, `test_dora_serialization.py`, `test_training_correctness.py`, `test_training_metadata.py`.

코드 완료의 기준은 위 검사 통과다. 이미지 품질 향상은 별도의 실험 결과로 보고한다.

**결과(2026-10-17)**: `testing/test_nora.py` 25개 통과(CUDA compile 포함). 이 중 3개는 외부 검토가 찾은 모드 전환 재개 문제에 대한 것이다. 실제 트레이너의 `get_training_info`, `load_training_state_from_metadata`, `restore_paired_training_state`를 소스에서 실행하며, 저장 모드 3종 × 재개 모드 3종을 모두 확인한다. 핵심 경로(forward 정규화, export/merge/DoRA norm의 Norm(A), 로드 경고)를 일부러 끊으면 해당 테스트가 실패하는 것도 확인했다. 회귀 묶음(10번 항목 + `test_anima_loader.py`)의 실패 2건은 변경 전에도 같았다. 하나는 sd-scripts venv의 lycoris 3.4.0 때문에 실패하는 LoCon compile 케이스, 다른 하나는 `test_latent_cache_invalidates_same_name_size_and_mtime_content_replacement`다.

## 7. 구현 순서와 학습 평가

1. `toolkit/nora.py`, config 파싱, 모듈 초기화(init + continuous의 DoRA kaiming), 검증 1–3
2. continuous의 forward, DoRA norm, merge, 저장 굳히기, 로드 경고, 검증 4–7
3. 재개 경고와 검증 8–10, 예제와 로그
4. (사용자 요청 시) A/B 실험
5. (이점 확인 시) UI

A/B 원칙: 같은 base, 데이터, caption, rank/alpha, optimizer, seed, 고정 평가 prompt를 쓰고, 실험마다 새 job 이름을 쓴다. 우선순위는 LoRA `off`(lr 1×/3×/10×) 대 `continuous`(1×/1/3/1/10) 대 `init`(1×/1/3/1/10)이다. DoRA 쌍(`off` 대 `continuous`)은 LoRA 결과로 학습률 범위를 좁힌 뒤 진행한다. 평가는 같은 step 기준으로 대상 유사도, 의상·색상 세부, 새 pose/배경 prompt 대응, 트리거 없는 일반 출력 보존(망각)을 본다. PEFT 메인테이너 재현에서 continuous가 망각을 늘린 사례가 있으므로 마지막 항목을 반드시 포함한다. 작은 차이로 기본값을 바꾸지 않고, 효과가 보이면 여러 seed로 재검증한다.

## 근거

- 논문: [Normalized Low-Rank Adaptation](https://arxiv.org/abs/2608.31036). 식 (7)·(9)·(11), 표 3·5, 부록 B(α = r 권장)
- 저자 구현: [Joluck/NoRA](https://github.com/Joluck/NoRA). `use_nora="init"`은 초기화 직후 정규화, `use_nora=True`는 forward 정규화(원본 A는 kaiming)
- 독립 재현과 PEFT 논의: [huggingface/peft#3637](https://github.com/huggingface/peft/pull/3637), [huggingface/peft#3908](https://github.com/huggingface/peft/pull/3908)
