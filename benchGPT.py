import os
import textwrap
from src.app import gorgiasController
import json
import csv
from openai import OpenAI

prompt = textwrap.dedent("""
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
        {english_text}
        """)

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

csv_file = open('gorgias_codes_GPT_3.5.csv', mode='w',
                newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Index", "Original Prompt",
                    "Generated Gorgias Code", "Query Result"])

json_file = open("test_g.json",encoding="utf-8")
json_data = json.load(json_file)
# shuffle and take the first 50 items
# random.shuffle(json_data)
json_data = json_data[:100]

# number of item in json
num_items = len(json_data)

count = 0

for i in range(num_items):
    print(f"Item {i+1}:")
    item = json_data[i]['messages'][1]['content']
    text = prompt.format(english_text=item)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
        
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=512
    )
    gorgias_code = completion.choices[0].message.content.strip()
    gorgiasCloud = gorgiasController.GorgiasController("gpt")
    gorgiasCloud.sendContent(gorgias_code)
    facts, query = gorgiasController.createQueryGorgias(gorgias_code)
    response = gorgiasCloud.query(facts=facts, query=query)
    gorgiasCloud.deleteProject()
    print(gorgias_code)
    result = "Success" if response['hasResult'] else "Fail"
    csv_writer.writerow([i+1, item, gorgias_code, result])
    if response['hasResult']:
        count += 1
        print("Success")
    else:
        print("Fail")
csv_file.close()
print(f"Success rate: {count}/{num_items}")
