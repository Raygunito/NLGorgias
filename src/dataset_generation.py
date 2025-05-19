from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
import re
import time
from openai import OpenAI
import json

from tqdm import tqdm

actions = [
    # Work-related actions
    "attend_meeting", "finish_report", "reply_emails", "give_presentation",
    "prepare_report", "schedule_meeting", "call_client", "update_project_plan", "submit_timesheet", "review_documents",

    # Social & leisure activities
    "go_to_restaurant", "go_to_cinema", "go_to_park", "go_to_theater",
    "visit_family", "attend_concert", "travel_abroad", "go_shopping",
    "visit_museum", "attend_workshop", "host_dinner_party", "go_to_bar", "take_picnic", "explore_city",

    # Health & exercise
    "go_gym", "morning_run", "yoga_session", "visit_doctor",
    "evening_walk", "swim_session", "cycling_session", "meditate", "join_fitness_class",

    # Daily tasks
    "buy_groceries", "clean_house", "cook_dinner", "read_book",
    "do_laundry", "water_plants", "pay_bills", "plan_meals", "organize_workspace", "make_coffee", "dispose_trash",

    # Transportation
    "take_bus", "ride_bike", "drive_car", "book_flight",
    "take_train", "rent_car", "order_taxi", "book_ride_share", "cycle_to_work", "use_subway"
]

facts = [
    # Work-related facts
    "urgent_deadline", "important_meeting", "boss_in_office", "team_project_due",
    "project_extension", "client_feedback", "deadline_missed", "extended_work_hours",

    # Personal situations
    "feeling_sick", "birthday_today", "wedding_anniversary", "friend_in_town", "medical_appointment",
    "job_interview", "relationship_break", "moving_house", "family_emergency", "vacation_planned",

    # Weather conditions
    "good_weather", "rainy_day", "snowstorm", "hot_day",
    "cloudy_day", "windy_day", "hail_storm", "humid_day",

    # Time-based events
    "weekend", "holiday_season", "morning_rush", "night_time",
    "afternoon", "lunch_time", "early_morning", "dusk",

    # Social dynamics
    "invitation_from_friend", "family_gathering", "new_restaurant_to_try", "concert_nearby",
    "party_invitation", "new_neighbor", "unexpected_guest", "community_event", "networking_event", "school_reunion",

    # Financial considerations
    "low_budget", "got_bonus", "discount_on_flight", "expensive_event",
    "unexpected_expense", "financial_aid", "good_investment", "tax_refund", "subscription_due"
]


class GorgiasDataset:
    def __init__(self, custom_actions=None, custom_facts=None):
        """
        Initialize the GorgiasDataset with optional custom actions and facts.
        :param custom_actions: List of custom actions to use instead of the default ones.
        :param custom_facts: List of custom facts to use instead of the default ones.
        """
        self.actions = actions
        self.facts = facts
        if custom_actions:
            self.actions = custom_actions
        if custom_facts:
            self.facts = custom_facts

    @staticmethod
    def convert_to_dynamic(term):
        match = re.match(r"(\w+)(\((.*?)\))?", term)
        if match:
            predicate = match.group(1)
            args = match.group(3)
            arity = len(args.split(',')) if args else 0
            return f"{predicate}/{arity}"
        return None

    def generate_beginner_gorgias(self, forcedType: int = None) -> str:
        """
        Generate a Gorgias code with a beginner level of complexity.
        :param forcedType: Optional parameter to force a specific type of the generated code. (0, 1, 2)
        If None, a random depth between 1 and 3 will be chosen.
        :return: A string representing the Gorgias code.
        """
        functionArray = [GorgiasDataset.generate_beginner_gorgias_common,
                         GorgiasDataset.generate_beginner_gorgias_conflict, GorgiasDataset.generate_beginner_gorgias_no_common]
        depth = random.randint(1, 3)
        if forcedType is not None:
            return functionArray[forcedType](depth, self.facts, self.actions)
        return functionArray[random.randint(0, 2)](depth, self.facts, self.actions)

    @staticmethod
    def generate_beginner_gorgias_common(depth: int, fact: list, action: list) -> str:
        tmp = fact[:]
        action1, action2 = random.sample(action, 2)
        shared_condition = random.choice(tmp)
        tmp.remove(shared_condition)
        dynamic = f":- dynamic {GorgiasDataset.convert_to_dynamic(shared_condition)}"

        lists = [
            f"rule(r1, {action1}, []) :- {shared_condition}.",
            f"rule(r2, {action2}, []) :- {shared_condition}.",
            f"rule(p1, prefer(r1, r2), [])."
        ]

        if depth == 1:
            dynamic += f"."
            complement1 = f"complement({action2}, {action1})."
            complement2 = f"complement({action1}, {action2})."
            lists.extend([complement1, complement2])
            lists.insert(0, dynamic)
            return "\n".join(lists)

        if depth == 2:
            p2_condition = random.choice(tmp)
            tmp.remove(p2_condition)
            txt = f"rule(p2, prefer(r2, r1), []) :- {p2_condition}."
            lists.extend([txt])
            txt = f"rule(c1, prefer(p2, p1), [])."
            lists.extend([txt])
            complement1 = f"complement({action2}, {action1})."
            complement2 = f"complement({action1}, {action2})."
            lists.extend([complement1, complement2])
            dynamic += f", {GorgiasDataset.convert_to_dynamic(p2_condition)}"

        if depth == 3:
            p2_condition = random.choice(tmp)
            tmp.remove(p2_condition)
            c2_condition = random.choice(tmp)
            txt = f"rule(p2, prefer(r2, r1), []) :- {p2_condition}."
            lists.extend([txt])
            txt = f"rule(c1, prefer(p2, p1), [])."
            lists.extend([txt])
            txt = f"rule(c2, prefer(p1, p2), []) :- {c2_condition}."
            lists.extend([txt])
            txt = f"rule(c3, prefer(c2, c1), [])."
            lists.extend([txt])
            complement1 = f"complement({action2}, {action1})."
            complement2 = f"complement({action1}, {action2})."
            lists.extend([complement1, complement2])
            dynamic += f", {GorgiasDataset.convert_to_dynamic(p2_condition)}, {GorgiasDataset.convert_to_dynamic(c2_condition)}"
        lists.insert(0, dynamic)
        return "\n".join(lists)

    def generate_beginner_gorgias_no_common(depth: int, fact: list, action: list) -> str:
        tmp = fact[:]
        action1, action2 = random.sample(action, 2)
        shared_condition = random.choice(tmp)
        tmp.remove(shared_condition)
        dynamic = f":- dynamic "

        lists = [
            f"rule(r1, {action1}, []).",
            f"rule(r2, {action2}, []).",
            f"rule(p1, prefer(r1, r2), [])."
        ]

        if depth == 1:
            dynamic += f"."
            complement1 = f"complement({action2}, {action1})."
            complement2 = f"complement({action1}, {action2})."
            lists.extend([complement1, complement2])
            lists.insert(0, dynamic)
            return "\n".join(lists)

        if depth == 2:
            p2_condition = random.choice(tmp)
            tmp.remove(p2_condition)
            txt = f"rule(p2, prefer(r2, r1), []) :- {p2_condition}."
            lists.extend([txt])
            txt = f"rule(c1, prefer(p2, p1), [])."
            lists.extend([txt])
            complement1 = f"complement({action2}, {action1})."
            complement2 = f"complement({action1}, {action2})."
            lists.extend([complement1, complement2])
            dynamic += f"{GorgiasDataset.convert_to_dynamic(p2_condition)}"

        if depth == 3:
            p2_condition = random.choice(tmp)
            tmp.remove(p2_condition)
            c2_condition = random.choice(tmp)
            txt = f"rule(p2, prefer(r2, r1), []) :- {p2_condition}."
            lists.extend([txt])
            txt = f"rule(c1, prefer(p2, p1), [])."
            lists.extend([txt])
            txt = f"rule(c2, prefer(p1, p2), []) :- {c2_condition}."
            lists.extend([txt])
            txt = f"rule(c3, prefer(c2, c1), [])."
            lists.extend([txt])
            complement1 = f"complement({action2}, {action1})."
            complement2 = f"complement({action1}, {action2})."
            lists.extend([complement1, complement2])
            dynamic += f"{GorgiasDataset.convert_to_dynamic(p2_condition)}, {GorgiasDataset.convert_to_dynamic(c2_condition)}"
        lists.insert(0, dynamic)
        return "\n".join(lists)

    def generate_beginner_gorgias_conflict(depth: int, fact: list, action: list) -> str:
        tmp = fact[:]
        action1, action2 = random.sample(action, 2)
        condition1 = random.choice(tmp)
        tmp.remove(condition1)
        condition2 = random.choice(tmp)
        tmp.remove(condition2)
        conditionConflict = random.choice(tmp)
        tmp.remove(conditionConflict)
        conditionConflict2 = random.choice(tmp)
        tmp.remove(conditionConflict2)
        dynamic = f":- dynamic {GorgiasDataset.convert_to_dynamic(condition1)}, {GorgiasDataset.convert_to_dynamic(condition2)}, {GorgiasDataset.convert_to_dynamic(conditionConflict)}"

        lists = [
            f"rule(r1, {action1}, []) :- {condition1}.",
            f"rule(r2, {action2}, []) :- {condition2}.",
            f"rule(p1, prefer(r1, r2), []) :- {condition1}, {condition2}.",
            f"rule(p2, prefer(r2, r1), []) :- {condition1}, {condition2}, {conditionConflict}.",
            f"rule(c1, prefer(p2, p1), [])."
        ]

        if depth > 1:
            p2_condition = random.choice(tmp)
            tmp.remove(p2_condition)
            txt = f"rule(c2, prefer(p1, p2), []) :- {p2_condition}."
            lists.extend([txt])
            txt = f"rule(c3, prefer(c2, c1), [])."
            lists.extend([txt])

            dynamic += f", {GorgiasDataset.convert_to_dynamic(p2_condition)}"

            if depth == 3:
                c2_condition = random.choice(tmp)
                txt = f"rule(c4, prefer(c1, c2), []) :- {c2_condition}."
                lists.extend([txt])
                txt = f"rule(c5, prefer(c4, c3), [])."
                lists.extend([txt])
                dynamic += f", {GorgiasDataset.convert_to_dynamic(c2_condition)}"

        complement1 = f"complement({action2}, {action1})."
        complement2 = f"complement({action1}, {action2})."
        lists.extend([complement1, complement2])
        dynamic += f"."
        lists.insert(0, dynamic)
        return "\n".join(lists)



ROOT = "data/"
def createDatasetJSON(amount: int = 10):
    dataset = []
    gorgias_dataset = GorgiasDataset()

    prompt_files = ["prompt_common.txt", "prompt_conflict.txt", "prompt_no_common.txt"]
    prompt_contents = []
    for file_name in prompt_files:
        with open(ROOT + file_name, 'r', encoding='utf-8') as f:
            prompt_contents.append(f.read())

    def generate_single_example(i):
        gorgias_code = gorgias_dataset.generate_beginner_gorgias(1 + (i % 3))
        prompt = prompt_contents[(i % 3)]
        english_translation = convert_gorgias_to_english(gorgias_code, prompt)
        return {
            "id": i,
            "gorgias_code": gorgias_code,
            "english_translation": english_translation
        }

    results = [None] * amount
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(generate_single_example, i): i for i in range(amount)}
        for future in tqdm(as_completed(futures), total=amount, desc="Generating examples"):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"Error generating example {i}: {e}")

    dataset = [item for item in results if item is not None]
    return dataset

def convert_gorgias_to_english(gorgias_code, prompt: str):

    prompt = prompt + gorgias_code

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=512
        )
        translation = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        translation = "Error in translation."

    return translation


if __name__ == "__main__":
    start_time = time.time()

    jsonToDump = createDatasetJSON(3)

    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(jsonToDump, f, indent=4)

    duration = time.time() - start_time
    print(f"Dataset generated and saved in {duration:.2f} seconds.")