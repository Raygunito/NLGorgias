import random
import gorgiasController
import NLtoGorgias
import json
import csv

csv_file = open('gorgias_codes.csv', mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Index", "Original Prompt", "Generated Gorgias Code", "Query Result"])

json_file = open("test_g.json")
json_data = json.load(json_file)
json_data = json_data[:100]

# number of item in json
num_items = len(json_data)
llm = NLtoGorgias.NLtoGorgias("./fine_tuned_gemma.pt_ft")

count = 0

for i in range(num_items):  
    print(f"Item {i+1}:")
    item = json_data[i]['messages'][1]['content']
    gorgias_code = llm.generate_gorgias_code(item)
    gorgiasCloud = gorgiasController.GorgiasController()
    gorgiasCloud.sendContent(gorgias_code)
    facts,query = gorgiasController.createQueryGorgias(gorgias_code)
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
    
    
