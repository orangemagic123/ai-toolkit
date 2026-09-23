# 학습 정확성 회귀 테스트

```bash
python -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
.venv/bin/pip install pytest accelerate optimum-quanto==0.2.7
.venv/bin/python -m pytest -q testing/test_training_correctness.py testing/test_training_metadata.py testing/test_lr_scheduler_warmup.py
```

실제 CPU 텐서, optimizer, EMA, Accelerate를 사용한다. 전체 diffusion 백엔드와 모델 가중치 다운로드 없이 실행하도록 큰 trainer의 메서드는 원본 AST에서 불러온다. GPU 전체 학습이나 attention 커널 성능을 검증하는 테스트는 아니다.

검증 범위:

- 새로운 `gradient_accumulation`과 기존 `gradient_accumulation_steps`의 평균 gradient, optimizer·scheduler·EMA 업데이트 횟수, epoch 경계와 마지막 부분 누적 처리
- 이미지·control 이미지 교체, 인코더 revision·로컬 가중치·최대 토큰 수 변경에 따른 캐시 무효화
- 추론용 EMA 가중치와 원래 학습 가중치의 분리, optimizer·scheduler·EMA·미완료 누적 gradient 복원, 저장 실패 시 원래 가중치 복구
- Anima의 두 내부 모듈에 attention backend 전달, 실제 블록 목록을 통한 컴파일, 준비된 모델의 forward 호출

호환성과 저장 형식:

- 기존 캐시는 새 키와 일치하지 않으므로 한 번 다시 생성된다. 미디어는 내용 해시, 로컬 모델은 파일 목록·크기·나노초 수정/변경 시각, Hub 모델은 ID와 로드된 컴포넌트 설정의 revision을 사용한다.
- 새 체크포인트마다 `.training_state/<체크포인트 이름>.pt`를 함께 저장한다. 재개할 때 이 파일도 보관해야 한다. 학습 파라미터와 optimizer 상태를 함께 저장하므로 저장 공간이 추가로 필요하다. 보존 개수를 넘은 체크포인트를 삭제하면 짝을 이루는 상태 파일도 삭제한다.
- 예전 EMA 체크포인트에는 원래 가중치와 EMA 이력이 없으므로 이를 복원할 수 없다는 경고 후 기존 방식으로 재개한다.
- 모델 가중치와 학습 상태를 복원한다. 데이터 순서, augmentation 및 난수 상태까지 동일하게 재생하는 기능은 포함하지 않는다.
- `merge_network_on_save`는 저장 시 학습 가중치를 영구적으로 병합·초기화한다. EMA와 함께 사용하면 가중치와 optimizer 상태의 대응이 깨지므로 이 조합은 명시적으로 거부한다.
