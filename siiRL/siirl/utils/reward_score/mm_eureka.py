import os
import re
from datetime import datetime

from loguru import logger
import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse
from siirl.utils.extras.patch import verify

choices = ["a", "b", "c", "d"]


def extract_answer_with_tags(text):
    match = re.search(r"(<answer>.*?</answer>)", text)
    if match:
        return match.group(1)
    return None


def accuracy_reward_func(completion, answer):
    reward = 0.0
    response = extract_answer_with_tags(completion)
    if response != None:
        response = response
    else:
        try:
            response = completion.split("<answer>")[-1]
        except:
            response = completion.split("\n")[-1]

    content, sol = response, answer
    answer_parsed = content
    gold_parsed = parse(sol)
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            content,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception:
            pass

        if reward == 0.0:
            try:
                content_match = re.search(r"<answer>(.*?)</answer>", completion)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = student_answer.replace("</answer>", "").replace("<answer>", "").strip()
                for answer in gold_parsed:
                    if str(answer).lower() in choices:
                        if str(answer).lower() in student_answer.lower():
                            choices_other = [choice for choice in choices if choice != str(answer).lower()]
                            if all(choice not in student_answer.lower() for choice in choices_other):
                                reward = 1.0
            except Exception:
                pass
    else:
        reward = 1.0
        print("Failed to parse gold solution: ", sol)

    return reward, answer_parsed


def format_reward_func(completion, **kwargs):
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>.+?</answer>.*$"
    )
    matches = re.search(pattern, completion, re.DOTALL)
    return 0.5 if matches else 0.0


def compute_score(predict_str: str, ground_truth: str) -> float:
    try:
        accuracy_reward, answer_parsed = accuracy_reward_func(predict_str, ground_truth)
        format_reward = format_reward_func(predict_str)
    except:
        logger.warning(f"Error in computing rewards for prediction: {predict_str}")
        accuracy_reward = 0.0
        format_reward = 0.0
        answer_parsed = ""
    # LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")
    # with open(LOG_PATH, "a") as f:
    #     f.write(f"===============================================================\n")
    #     f.write("【Predict Str】: " + predict_str + "\n")
    #     f.write("【Answer】: " + ground_truth + "\n")
    #     f.write(f"【Accuracy Reward】: {accuracy_reward}\tFormat Reward: {format_reward}\n")
    #     f.write(f"===============================================================\n")
    return 0.9 * accuracy_reward + 0.1 * format_reward
