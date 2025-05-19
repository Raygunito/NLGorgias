# use METEOR and do it on eval_results.json
# import nltk
# nltk.download('wordnet')
# from nltk.translate.meteor_score import meteor_score
# from nltk.tokenize import word_tokenize

# import json
# import os

# def load_json_file(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data


# if __name__ == "__main__":
#     data = load_json_file("eval_result.json")
#     # json structure : [ {"pred_answer": "model answer", "answer": "real answer", "bleu_score": 0.5},...]
#     # compute METEOR score
#     for i, item in enumerate(data):
#         pred_answer = item["pred_answer"]
#         answer = item["answer"]
#         bleu_score = item["bleu_score"]
#         meteor_score_value = meteor_score([word_tokenize(answer)], word_tokenize(pred_answer))
#         data[i]["meteor_score"] = meteor_score_value
#     # save the data
#     with open("eval_results_meteor.json", 'w') as file:
#         json.dump(data, file, indent=4)
#     print("METEOR scores computed and saved to eval_results_meteor.json")
#     # print the mean METEOR score
#     mean_meteor_score = sum(item["meteor_score"] for item in data) / len(data)
#     print(f"Mean METEOR score: {mean_meteor_score}")
    
    
import nltk
nltk.download('wordnet')
import os
import json
from openai import OpenAI
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.corpus import wordnet
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
wordnet.ensure_loaded()
def process_item(index, item):
    prompt = item['messages'][0]['content']
    true_answer = item['messages'][1]['content']
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=512
    )
    translation = completion.choices[0].message.content.strip()
    meteor_score_value = meteor_score([word_tokenize(true_answer)], word_tokenize(translation))
    return index, {
        "answer": true_answer,
        "pred_answer": translation,
        "meteor_score": meteor_score_value
    }

def main():
    with open("test_g.json", "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    results = [None] * len(json_data)
    mean = 0.0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_item, idx, item) for idx, item in enumerate(json_data)]
        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result
            mean += result["meteor_score"]

    with open("meteor_score.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        f.write("\n")

    mean /= len(results)
    print(f"Mean Meteor score: {mean}")

if __name__ == "__main__":
    main()
