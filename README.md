# NetLLaMA

```bash
pip install pandarallel pandas jupyter numpy datasets sentencepiece openai wandb accelerate tiktoken transformers==4.32.0
```

To train the model, simply run

```
bash run.sh
```

# For query_gpt.py file:

This file utilizes the OpenAI API, specifically the GPT-3.5-turbo and GPT-4 models, to analyze questions from TeleQnA.json dataset and evaluate the accuracy of the model's responses. By comparing different models' responses to the same set of questions, this file aims to assess and compare their performance in terms of question response accuracy, category-specific accuracy, and the relationship between question length and correctness.

**Step 1: Environment Setup**(https://platform.openai.com/docs/quickstart/step-1-setting-up-python)
**_Install Python_**
Python Environment: Requires Python 3.6 or higher.

**_Set up a virtual environment_**
To create a virtual environment, Python supplies a built in venv module which provides the basic functionality needed for the virtual environment. Running the command below will create a virtual environment named "openai-env" inside the current folder you have selected in your terminal / command line:

```
python -m venv openai-env
```

Once you’ve created the virtual environment, you need to activate it. On Windows, run:

```
openai-env\Scripts\activate
```

On Unix or MacOS, run:

```
source openai-env/bin/activate
```

**_Install the OpenAI Python library_**
Once you have Python 3.7.1 or newer installed and (optionally) set up a virtual environment, the OpenAI Python library can be installed. From the terminal / command line, run:

```
pip install --upgrade openai
```

Once this completes, running pip list will show you the Python libraries you have installed in your current environment, which should confirm that the OpenAI Python library was successfully installed.

**_Dependencies Installation_**
The openai library is needed for API calls, along with the matplotlib library for data processing and visualization.

Installation command:

```
pip install openai matplotlib
```

**Step 2: Set up your API key**

Set the correct API key in your current terminal session manually:
On Linux or macOS:

```
export OPENAI_API_KEY="your_openai_api_key_here"
```

On Windows Command Prompt:

```
set OPENAI_API_KEY=your_openai_api_key_here
```

On Windows PowerShell:

```
$Env:OPENAI_API_KEY="your_openai_api_key_here"
```

**Dataset**
The dataset file should be placed in the specified path and be in JSON format.
The dataset file path should be assigned to the 'DATA_PATH' variable.

**Step 3: run the file**
eg. run

```
python /Users/yanxi/Documents/GitHub/NetLLaMA/datasets/query_gpt.py
```
