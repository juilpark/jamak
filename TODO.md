# jamak TODO

- 기준 문서: `/Users/parkjuil/Git/jamak/docs/PROJECT_PLAN.md` (v0.2)
- 현재 우선순위: 로컬 CLI MVP (Phase 0~1)

## 확정 정책

- [ ] Python 버전 정책 적용: `3.14 -> 3.13 -> 3.12` 단계 하향
- [ ] WebUI 스택은 확장 단계에서 `FastAPI` 사용
- [ ] 운영 우선순위는 `CUDA` 고정 (ROCm/Intel은 Phase 4 이후)
- [ ] MVP 출력 포맷은 `SRT` 단일
- [ ] 번역 인터페이스는 OpenAI-Compatible 표준 사용

## Phase 0 (기반 정리)

- [x] `src/jamak/` 패키지 구조 생성 (`core`, `infra`, `app`, `schemas`)
- [x] Typer CLI 엔트리포인트 생성 (`jamak transcribe`, `jamak batch`, `jamak doctor`)
- [x] 공통 설정 로더 추가 (`device`, `hf cache`, `output path`)
- [x] `jamak doctor` 1차 구현
- [x] `ffmpeg` 설치/접근성 체크
- [x] Python 런타임 정책 점검 로직 추가 (버전 미지원 시 하향 가이드 출력)
- [x] 최소 테스트 프레임 구성 (`tests/unit`, `tests/integration`)

## Phase 1 (로컬 CLI MVP)

- [x] 입력 파일에서 오디오 추출 (`ffmpeg`)
- [ ] FireRedVAD 연동
- [ ] Qwen3-ASR-1.7B 연동
- [ ] Qwen3-ForcedAligner-0.6B 연동
- [ ] 파이프라인 연결 (`VAD -> ASR -> Align -> Subtitle`)
- [ ] SRT writer 구현
- [ ] 메타데이터/로그 출력 (`segments.json`, `run.json`, `*.log`)
- [ ] `transcribe` 단일 파일 E2E 검증
- [ ] `batch` 다중 파일 E2E 검증
- [ ] 샘플 파일 2개(짧은/긴) 기준 품질-성능 측정(CER/WER/RTF)

## Phase 1 완료 기준

- [ ] 로컬에서 CLI 한 줄 명령으로 SRT 생성 가능
- [ ] 실패 시 원인 파악 가능한 로그 출력
- [ ] 동일 입력 재실행 시 결과 재현 가능

## 확장 백로그 (Phase 2+)

- [ ] Phase 2: FastAPI 기반 WebUI
- [ ] Phase 3: Docker + CUDA 기반 NAS 운영
- [ ] Phase 4: OpenAI-Compatible 번역 기능
- [ ] Phase 5: ROCm/Intel 런타임 검증 및 패키징

## 아키텍처 가드레일

- [ ] 백엔드 추상화 인터페이스 고정 (`cuda`, `rocm`, `intel` 어댑터 분리)
- [ ] CUDA 중심 구현에서도 타 백엔드 확장 시 core 파이프라인 변경 최소화
- [ ] 백엔드별 회귀 테스트 포인트 사전 정의
