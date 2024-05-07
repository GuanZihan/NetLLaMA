import openai
import os
import json
# from dotenv import load_dotenv
import matplotlib.pyplot as plt

# load_dotenv()
# API_KEY = os.getenv("OPENAI_API_KEY")
# openai.api_key = API_KEY

# Define the path to the dataset
DATA_PATH = '/Users/yanxi/Documents/GitHub/NetLLaMA/datasets/TeleQnA.json'

# Load the JSON file
with open(DATA_PATH, 'r') as file:
    all_data = json.load(file)
    # Only analyze the last 2000 entries of the dataset
    data = all_data[-2000:]

# Function to query the OpenAI API
def query_openai_api(question, options, model_name):
    # Construct the complete text including the question and options
    options_text = ', '.join([f"{key}: {value}" for key, value in options.items() if value is not None])
    full_question = f"{question} Options: {options_text}"
    
    try:
        # Make the API call
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "user", "content": full_question}
            ]
        )
        # Extract and return the response content
        return response.choices[0].message['content'].strip()
    except Exception as e:
        # Print any errors that occur
        print(f"An error occurred with model {model_name}: {e}")
        return None


# Models to test
models = ["gpt-3.5-turbo", "gpt-4"]

# Iterate through each model
for model_name in models:
    # Initialize counters and accumulators
    correct_count = 0
    category_correct = {}
    category_total = {}
    length_accuracy = []

    print(f"\nTesting with model: {model_name}")
    
    # Iterate through each entry in the data
    for item in data:
        # Extract the question and category
        question = item['question']
        category = item.get('category', 'Unknown')
        # Calculate the length of the question in words
        question_length = len(question.split())
        # Construct the options dictionary
        options = {
            'option_1': item.get('option_1'),
            'option_2': item.get('option_2'),
            'option_3': item.get('option_3'),
            'option_4': item.get('option_4'),
            'option_5': item.get('option_5')
        }
        # Extract the correct answer
        correct_answer = item['answer'].split(': ')[1].lower()

        print(f"Querying the API for the question: {question[:50]}...")
        response = query_openai_api(question, options, model_name)

        # Check if the response is None
        if response is None:
            print("Received no response for the question, skipping...")
            continue  # Skip to the next iteration

        # Format the API response
        if response.lower().startswith("option"):
            response = response.split(': ')[1].lower() if ': ' in response else response.lower()

        # Check if the response matches the correct answer
        is_correct = correct_answer in response
        if is_correct:
            correct_count += 1
            category_correct[category] = category_correct.get(category, 0) + 1
        category_total[category] = category_total.get(category, 0) + 1
        length_accuracy.append((question_length, is_correct))
    
    # Calculate and print the accuracy rate for each model
    total_count = len(data)
    accuracy = (correct_count / total_count) * 100
    print(f"Correct response rate for {model_name}: {accuracy:.2f}%")

    # Plot 1: Ratio of correct to wrong answers
    plt.figure(figsize=(8, 4))
    plt.bar(['Correct', 'Wrong'], [correct_count, total_count - correct_count], color=['green', 'red'])
    plt.title(f'Correct vs. Wrong Answers - {model_name}')
    plt.show()

    # Plot 2: Correct rates by question category
    plt.figure(figsize=(10, 6))
    categories = list(category_correct.keys())
    correct_rates = [(category_correct[c] / category_total[c]) * 100 for c in categories]
    plt.bar(categories, correct_rates)
    plt.title(f'Correct Rates by Category - {model_name}')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel('Correct Rate (%)')
    plt.tight_layout()
    plt.show()

    # Plot 3: Relationship between question length and correctness
    lengths, correctness = zip(*length_accuracy)
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, [1 if is_correct else 0 for is_correct in correctness], alpha=0.5)
    plt.title(f'Question Length vs. Correctness - {model_name}')
    plt.xlabel('Question Length (words)')
    plt.ylabel('Correctness (1=Correct, 0=Incorrect)')
    plt.show()

