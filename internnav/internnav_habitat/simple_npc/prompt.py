TEMPLATE = {
    'one_turn_prompt': """
You are a helpful assistant in helping agent to finish its navigation task.

## Here is the ground truth information you know more than the agent
'TASK DONE' shows if the agent has finished the task, if it is false, you need to know that the agent hasn't found the goal object.
'GOAL INFORMATION' shows the goal object's information.
'CORRECT PATH' shows the correct path description to the goal object.

TASK DONE:
{task_done}

GOAL INFORMATION:
{goal_information}

CORRECT PATH:
{path_description}

## Some constraints you MUST follow:
1. Only output the answer to the question.
2. Don't be verbose.

## Here is the question you need to answer
QUESTION: {question}
""",
    "two_turn_prompt_0": """
You are a helpful assistant in helping agent to finish its navigation task. You will be given a question among the following three types:
1. Disambiguation: This question is asked to check whether the agent has found the goal object. Like "Is it the object you are looking for?"
2. Path: This question is asked to get the path to the goal object. Like "Where should I go now?"
3. Information: This question is asked to get more information about the goal object. Like "Where is the goal object?", "What is the color of the goal object?"

You need to classify the question into one of the three types. Only output the name of the type(disambiguation, path, information). Don't be verbose.

## Here is the question you need to answer
QUESTION: {question}
""",
    "two_turn_prompt_1": """
You are a helpful assistant in answering the question. Here follows the ground truth information about the goal object. You need to answer the question based on the ground truth information.

## Here is the ground truth information about the goal object
GOAL INFORMATION:
{goal_information}

## Here is the question you need to answer
QUESTION: {question}
""",
}

DISAMBIGUATION_PROMPT = {
    'yes': [
        "Yes, you are in the correct position.",
        "That's right, you are at the intended location.",
        "Yes, you have reached the right spot.",
        "Correct, you are in the proper place.",
        "Yes, you are exactly where you need to be.",
        "Yes, you are aligned correctly.",
        "Yes, you are positioned accurately.",
        "Everything looks good, you are at the correct location.",
        "You are in the right area.",
        "Yes, you are currently at the correct position.",
        "That's perfect, you are in the right spot.",
        "Yes, your position is accurate.",
        "You have reached the proper location.",
        "Yes, you are at the specified position.",
        "Everything is aligned properly, you're in the correct spot.",
        "Yes, you are where you should be.",
        "Yes, this is the right place.",
    ],
    'no': [
        "This is not the intended location.",
        "You are not in the proper place.",
        "No, you are not where you need to be.",
        "No, you are not aligned correctly.",
        "No, you are positioned incorrectly.",
        "You are not at the correct location.",
        "No, you are situated incorrectly.",
        "You are in the wrong area.",
        "No, you are not currently at the correct position.",
        "That's not the right spot.",
        "No, you are not at the intended destination.",
        "Your position is inaccurate.",
        "You haven't reached the proper location.",
        "No, you are not at the specified position.",
        "The alignment is off, you are in the wrong spot.",
        "This is not the right place.",
    ],
}
