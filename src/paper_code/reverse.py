import os
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

adapter_path = "./model_v0"
peft_config = PeftConfig.from_pretrained(adapter_path)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

prompt = f"""
<bos><start_of_turn>user
You are given a Gorgias logic program that defines possible actions, conditions, and preferences.

Only output the Gorgias code.
Do NOT output any explanation.
Follow exactly the previous examples format.
Do not add any extra sentences.
Do not forget to add the default preferences when there is 2 possible actions.

---

English:
If the weather is nice, I can either go out or stay home. 
Usually, I go out. 
But if there's a nice movie on TV, I stay home instead. 
However, if a friend invites me, I go out again. 
I can't go out and stay home at the same time.

Gorgias code:
:- dynamic nice_weather/0, nice_movie_tv/0, invitation_from_friend/0.
rule(r1, go_out, []) :- nice_weather.
rule(r2, stay_home, []) :- nice_weather.
rule(p1, prefer(r1,r2), []).
rule(p2, prefer(r2,r1), []) :- nice_movie_tv.
rule(c1, prefer(p2,p1), []).
rule(c2, prefer(p1,p2), []) :- invitation_from_friend.
rule(c3, prefer(c2,c1), []).
complement(go_out, stay_home).
complement(stay_home, go_out).

English:
If I have a phone call, I can either answer it or ignore it.
However if I am at work, I prefer to deny it.
But if it's a family member calling, I prefer to answer it.
However if I am at a meeting, I prefer to deny it.
I can't at the same time answer and deny a call.

Gorgias code:
:- dynamic phone_call/0, at_work/0, family_member/0, at_meeting/0.
rule(r1, allow, []):- phone_call.
rule(r2, deny, []):- phone_call.
rule(p1, prefer(r1, r2), []).
rule(p2, prefer(r2, r1), []):- at_work.
rule(c1, prefer(p2, p1), []).
rule(c2, prefer(p1, p2), []):- family_member.
rule(c3, prefer(c2, c1), []).
rule(c4, prefer(c1, c2), []):- at_meeting.
rule(c5, prefer(c4, c3), []).
complement(deny, allow).
complement(allow, deny).

---

Now convert the following English text into Gorgias code:

'If there is a family emergency, I can either attend a workshop or finish my report. Generally, I choose to attend the workshop. I can’t attend the workshop and finish my report at the same time.'

<end_of_turn><start_of_turn>assistant
"""
supp = """If i have medical appointment I go to the cinema. 
If there is a new music album, i go to the park. 
But if i have both, i prefer to go to the cinema. 
However if it's sunny, I prefer to go to the park.
However if it's noisy outside i prefer to go to the cinema over go to the park.
But if there is blue sky, i prefer to go to the park.
I can't at the same time go to the cinema and go to the park."""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0])

if "<start_of_turn>assistant" in response:
    _, model_output = response.split("<start_of_turn>assistant", 1)
else:
    model_output = response

model_output = model_output.split("<end_of_turn>")[0]

model_output = model_output.strip()

print("Model output:")
print(model_output)
