import re


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(data_sources, solution_strs, ground_truths, extra_infos, method="strict", format_score=0.0, score=1.0, **kwargs):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        data_sources: a list of data sources
        solution_strs: a list of solution texts
        ground_truths: a list of ground truths
        extra_infos: a list of extra infos
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    scores = []
    for solution_str, ground_truth in zip(solution_strs, ground_truths):
        answer = extract_solution(solution_str=solution_str, method=method)
        if answer is None:
            scores.append(0)
        else:
            if answer == ground_truth:
                scores.append(score)
            else:
                scores.append(format_score)
    return scores