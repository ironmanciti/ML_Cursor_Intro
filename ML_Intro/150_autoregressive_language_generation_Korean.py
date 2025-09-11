#!/usr/bin/env python
# coding: utf-8

# # Autoregressive (자동회귀) 문장 생성 - SKT GPT2 모델
# 
# SKT GPT2 모델을 사용한 자동 회귀적인 텍스트 생성 예제입니다.
# SKT GPT2는 SKT에서 개발한 한국어 특화 언어모델로,
# 단순한 텍스트 입력을 받아 다음 토큰을 예측하여 텍스트를 생성합니다.



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# SKT GPT2 모델 및 토크나이저 초기화
# SKT에서 개발한 한국어 특화 GPT2 모델
model_name = "skt/kogpt2-base-v2"
print(f"모델 로딩 중: {model_name}")
print("토크나이저 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("모델 로딩 중...")
model = AutoModelForCausalLM.from_pretrained(model_name)
print("모델 로딩 완료!")

# 문장 시작 부분 (한국어로 변경)
input_text = "옛날 옛적에"
print(f"원본 텍스트: {input_text}")

# 토크나이저 정보 확인
print(f"토크나이저 vocab 크기: {tokenizer.vocab_size}")
print(f"특수 토큰들: {tokenizer.special_tokens_map}")

# 간단한 토큰화 테스트
input_ids = tokenizer.encode(input_text, return_tensors="pt")
print(f"토큰화된 결과: {input_ids}")
print(f"토큰 개수: {input_ids.shape[1]}")




# SKT GPT2 모델을 사용한 문장 생성
# GPT2는 단순한 텍스트 입력을 받아 다음 토큰을 예측하는 모델

# 입력 텍스트 토큰화
inputs = tokenizer(input_text, return_tensors="pt")

print("=== 입력 텍스트 토큰화 ===")
print(f"입력 텍스트: {input_text}")
print(f"토큰화된 결과: {inputs['input_ids']}")
print(f"토큰 개수: {inputs['input_ids'].shape[1]}")

# GPU 사용 가능하면 GPU로 이동
if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    print("GPU로 이동 완료")
else:
    print("CPU 사용 중")




# SKT GPT2 텍스트 생성
# generate 메서드의 주요 파라미터들:
# - max_length: 생성할 최대 토큰 수
# - num_return_sequences: 생성할 시퀀스 수
# - temperature: 생성 다양성 조절 (높을수록 다양함)
# - do_sample: 샘플링 사용 여부
# - pad_token_id: 패딩 토큰 ID

# 패딩 토큰 설정 (GPT2는 기본적으로 패딩 토큰이 없음)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

output_ids = model.generate(
    **inputs,
    max_length=200,
    num_return_sequences=1,
    temperature=0.8,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# 결과 출력
generated_text = tokenizer.batch_decode(output_ids)[0]
print("=== 생성된 텍스트 ===")
print(generated_text)

# 원본 입력과 새로 생성된 부분 분리
input_length = inputs['input_ids'].shape[1]
new_tokens = output_ids[0][input_length:]
new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(f"\n=== 새로 생성된 부분만 ===")
print(new_text)


# SKT GPT2는 자체적으로 autoregressive 모델입니다. "Autoregressive"란, 이전에 생성된 토큰들을 기반으로 다음 토큰을 생성하는 모델을 의미합니다.
# 
# 위의 코드에서 `model.generate` 메서드는 이미 autoregressive한 방식으로 문장을 생성합니다. 그러나 이를 명시적으로 보여주기 위해 각 단계에서 토큰을 하나씩 생성하는 autoregressive한 코드를 아래에 작성하겠습니다:
# 
# ## 다양한 한국어 프롬프트로 실험해보기
# 
# 다음 셀에서 다양한 한국어 프롬프트로 텍스트 생성을 시도해볼 수 있습니다:



# 다양한 한국어 프롬프트로 텍스트 생성 실험
prompts = [
    "인공지능의 미래는",
    "기술이 발달한 세상에서",
    "인생에서 가장 중요한 것은"
]

print("=== 다양한 한국어 프롬프트로 텍스트 생성 실험 ===\n")

for i, prompt in enumerate(prompts, 1):
    print(f"{i}. 프롬프트: '{prompt}'")

    # 입력 텍스트 토큰화
    inputs = tokenizer(prompt, return_tensors="pt")

    # GPU 사용 가능하면 GPU로 이동
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # 텍스트 생성
    output_ids = model.generate(
        **inputs,
        max_length=150,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # 결과 출력
    generated_text = tokenizer.batch_decode(output_ids)[0]
    input_length = inputs['input_ids'].shape[1]
    new_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
    print(f"\n   생성된 텍스트: {new_text}")
    print("-" * 200)




# SKT GPT2 Autoregressive 생성 과정 분석
# 단순한 텍스트 입력 구성
input_text = "옛날 옛적에"

# 입력 텍스트 토큰화
inputs = tokenizer(input_text, return_tensors="pt")

print(f"원본 프롬프트: {input_text}")
print(f"토큰화 후 토큰 수: {inputs['input_ids'].shape[1]}")
print(f"토큰화된 결과: {inputs['input_ids']}")

if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}




# SKT GPT2 Autoregressive한 방식으로 문장 생성
# 각 단계에서 이전 토큰들을 기반으로 다음 토큰을 예측하는 과정을 시각화

# 현재 입력 토큰 수 확인
current_length = inputs['input_ids'].shape[1]
print(f"현재 입력 토큰 수: {current_length}")

max_length = current_length + 30  # 현재 길이에서 30개 토큰 추가 생성
input_ids_concat = inputs['input_ids'].clone()

print("=== SKT GPT2 Autoregressive 생성 과정 ===\n")
print(f"시작 프롬프트: '{input_text}'\n")
print(f"목표 길이: {max_length} 토큰\n")

# 원본 입력 길이 저장 (새로 생성된 부분만 추출하기 위해)
original_length = current_length

step = 0
while input_ids_concat.shape[1] < max_length:
    step += 1

    # 다음 토큰 예측
    model_inputs = {"input_ids": input_ids_concat}
    if "attention_mask" in inputs:
        model_inputs["attention_mask"] = torch.ones_like(input_ids_concat)

    predictions = model(**model_inputs)
    logits = predictions.logits
    predicted_token = torch.argmax(logits[0, -1]).item()

    # 생성된 토큰을 입력 토큰 뒤에 추가 (같은 장치에 맞춤)
    new_token_tensor = torch.tensor([[predicted_token]], device=input_ids_concat.device)
    input_ids_concat = torch.cat([input_ids_concat, new_token_tensor], dim=1)

    # 새로 생성된 부분만 추출
    new_tokens = input_ids_concat[0][original_length:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    new_token = tokenizer.decode(predicted_token, skip_special_tokens=True)
    print(f"Step {step}: 새로 생성된 토큰: '{new_token}' -> 전체 생성 텍스트: '{new_text}'")

    # 너무 길어지면 중단
    if step > 30:
        break

print("생성 완료!")

