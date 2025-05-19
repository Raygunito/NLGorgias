import csv
import json
import random
import unicodedata
# final_test_dataset.json
# final_train_dataset.json
# final_validation_dataset.json
input_json = ["final_test_dataset.json",
              "final_train_dataset.json", "final_validation_dataset.json"]
output_json = ["test_g.json", "train_g.json", "val_g.json"]
prompt_template = """
You are given a Gorgias logic program that defines possible actions, conditions, and preferences.

First, read and internally understand the logic of the program.
A few example explanations are included to help you understand how these structures work.
**Do NOT include any explanation in your final answer. Only output the natural English translation.**

Your translation must:
- Accurately reflect the logic of the program.
- Be fluent and natural — like everyday speech.
- Avoid technical terms such as "preference", "priority", "override", or "conflict".
- Do not use the word "prefer".
- Use natural phrases like “I usually…”, “I tend to…”, “I choose to…”, “I go back to…”.
- Do not invent or simplify anything — just restate what is in the code using clear language.
- Only output the English translation, nothing else.

---

Example 1:
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
If the weather is nice, I can either go out or stay home. Usually, I go out. But if there’s a nice movie on TV, I stay home instead. However, if a friend invites me, I go out again. I can’t go out and stay home at the same time.
---
Example 2:
Gorgias code:
:- dynamic from_family_member/0, from_work/0.
rule(r1, deny_call, []).
rule(r2, accept_call, []).
rule(p1, prefer(r1, r2), []).
rule(p2, prefer(r2, r1), []) :- from_family_member.
rule(c1, prefer(p2, p1), []).
rule(c2, prefer(p1, p2), []) :- from_work.
rule(c3, prefer(c2, c1), []).
complement(accept_call, deny_call).
complement(deny_call, accept_call).

English:
I can either accept or deny the call. Usually, I deny it. But if it’s from a family member, I accept it instead. But if it's from work, I deny it. I can’t accept and deny the call at the same time.
---
Example 3:
Gorgias code:
:- dynamic team_project_due/0, night_time/0, rainy_day/0, important_meeting/0, new_restaurant_to_try/0.
rule(r1, go_shopping, []) :- team_project_due.
rule(r2, visit_doctor, []) :- night_time.
rule(p1, prefer(r1, r2), []) :- team_project_due, night_time.
rule(p2, prefer(r2, r1), []) :- team_project_due, night_time, rainy_day.
rule(c1, prefer(p2, p1), []).
rule(c2, prefer(p1, p2), []) :- important_meeting.
rule(c3, prefer(c2, c1), []).
rule(c4, prefer(c1, c2), []) :- new_restaurant_to_try.
rule(c5, prefer(c4, c3), []).
complement(visit_doctor, go_shopping).
complement(go_shopping, visit_doctor).

English:
If I have a team project due, I go shopping. If it’s nighttime, I visit the doctor. But if it’s both nighttime and I have a project due, I usually go shopping. If it’s also a rainy day, I visit the doctor instead. However if I have an important meeting, I go shopping again. But if I have a new restaurant to try, I visit the doctor. I can’t go shopping and visit the doctor at the same time.
---
Now translate the following Gorgias program into clear, natural English.
Again, **do NOT output any explanation** — just the final translation.

{gorgias_code}
"""
# Now translate the following Gorgias program into clear, natural English.


def normalize_text(text):
    text = unicodedata.normalize('NFKC', text)
    return text.strip()


for i in range(len(input_json)):
    data = []
    reader = json.load(open(input_json[i], 'r', encoding='utf-8'))
    for row in reader:
        translation = normalize_text(row['NL Translation'].strip())
        code = normalize_text(row['Gorgias Code'].strip().strip('"'))

        user_prompt = prompt_template.format(gorgias_code=code)

        item = {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": translation
                }
            ]
        }
        data.append(item)

    with open(output_json[i], 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        print("Number of items in the dataset:", len(data))
    print(f"Data saved to {output_json[i]}")
