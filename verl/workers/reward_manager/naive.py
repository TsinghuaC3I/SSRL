# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
from collections import Counter
import numpy as np


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def cut_and_normalize_strs(s):
    if s:
        s = s.strip().lower()
        s = s.split('\n')[0]
        s = s.split('.')[0]
        s = s.split(',')[0]
        if 'answer is' in s:
            s = s.split('answer is')[-1]
        if 'The answer is' in s:
            s = s.split('The answer is')[-1]
        # Cut off the first newline, period, or comma
        truncated_text = re.split(r'[\n.,]', s, 1)[0]

        # Remove punctuation
        no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

        # Remove article
        no_articles = re.sub(r'\b(an|the)\b',
                            '',
                            no_punctuation,
                            flags=re.IGNORECASE)

        # Remove duplicated blank spaces
        cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()
    else:
        cleaned_text = ''
    return cleaned_text



def f1_score_cal(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def em_check(pred, answer):
    if isinstance(answer, str):
        answer = [answer]

    em_score = np.max([int(normalize_answer(pred) == normalize_answer(answer[index])) for index in range(len(answer))])
    f1_score = np.max([f1_score_cal(normalize_answer(pred), normalize_answer(str(answer[index]))) for index in range(len(answer))])

    return em_score, f1_score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) == 0:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def extract_search(solution_str):
    """Extract the search query from the solution string."""
    search_pattern = r'<search>(.*?)</search>'
    match = re.finditer(search_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 2 or more matches, return the last one
    return len(matches)

def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    search_num = extract_search(solution_str=solution_str)
        
    if answer is None:
        return 0,0, search_num, 0
    answer_length = len(answer.split())
    em, f1 = em_check(answer, ground_truth['target'])
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"EM: {em}, F1: {f1}, Search Num: {search_num}, Answer Length: {answer_length}")
    return em, f1, search_num, answer_length

def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score

def is_valid_sequence(text):
    # Find the position of "<|im_start|>assistant" with potential whitespace
    content = text
    
    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?: think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
        
        # Check if this is a tag
        if re.match(r"</?(?: think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
    
    return True, "Valid sequence format"

def format_reward(response: str) -> float:
    """
    Simple format reward: 1.0 if correct format, 0.0 otherwise.
    
    Required format:
    1. Start with <think>...</think>
    2. Each <search> must be paired with <information>
    3. End with <answer>...</answer>
    """
    
    response = response.strip()
    
    # Check if any tag content contains disallowed tags
    allowed_tags = {'think', 'search', 'information', 'answer', '/think', '/search', '/information', '/answer'}
    all_tags = re.findall(r'<([^>]+)>', response)
    for tag in all_tags:
        if tag not in allowed_tags:
            return 0.0
    
    # if not is_valid_sequence(response)[0]:
    #     return 0.0
    
    # Must start with <think> and end with </answer>
    if not (response.startswith('<think>') and response.endswith('</answer>')):
        return 0.0

    # Extract all tags in order
    tags = re.findall(r'<(/?(?:think|search|information|answer))>', response)
    
    # Check if any tag content is empty
    tag_contents = {
        'think': re.findall(r'<think>(.*?)</think>', response, re.DOTALL),
        'search': re.findall(r'<search>(.*?)</search>', response, re.DOTALL),
        'information': re.findall(r'<information>(.*?)</information>', response, re.DOTALL),
        'answer': re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
    }
    
    
    if len(tags) < 4:  
        return 0.0
    # Return 0 if any tag has empty content
    for tag_type, contents in tag_contents.items():
        for content in contents:
            if not content.strip():
                return 0.0
            if tag_type == 'search' and len(content.split('\n')) != 1:
                return 0.0
            if tag_type == 'search' and 'your query' in content.lower():
                return 0.0
            if tag_type == 'think' and 'your thoughts' in content.lower():
                return 0.0
            if tag_type == 'answer' and 'your answer' in content.lower():
                return 0.0
            if tag_type == 'information' and 'your information' in content.lower():
                return 0.0
            # if tag_type == 'information':
            #     documents = []
            #     documents.append(content.split('Document 1: ')[-1].split('Document 2: ')[0])
            #     documents.append(content.split('Document 2:')[-1].split('Document 3: ')[0])
            #     documents.append(content.split('Document 3: ')[-1])
            #     documents = [doc.strip() for doc in documents if doc.strip()]
            #     documents = list(set(documents))
            #     if len(documents) != 3:
            #         return 0.0
            #     for line in documents:
            #         if len(line.split(' ')) < 15 or len(line.split(' ')) > 70:
            #             return 0.0  # 

    # Check structure
    if tags[0] != 'think' or tags[1] != '/think':
        return 0.0
    
    if tags[-2] != 'answer' or tags[-1] != '/answer':
        return 0.0
    
    # Check search-information pairing in the middle
    middle_tags = tags[2:-2]  # Exclude initial think and final answer
    
    i = 0
    while i < len(middle_tags):
        if middle_tags[i] == 'search':
            # Must be followed by /search, information, /information
            if (i + 3 >= len(middle_tags) or 
                middle_tags[i + 1] != '/search' or
                middle_tags[i + 2] != 'information' or 
                middle_tags[i + 3] != '/information'):
                return 0.0
            i += 4
        else:
            i += 1

    think_num = response.count('<think>')
    search_num = response.count('<search>')
    information_num = response.count('<information>')
    if search_num != information_num:
        return 0.0
    
    max_turn = 2
    score = 1.0 / max_turn * think_num
    ratio = 1.0
    
    upper_bound = 8
    if think_num != search_num + 1:
        ratio = min(think_num, search_num + 1) / max(think_num, search_num + 1)
        
    return min(score, 1.0) * ratio if think_num <= upper_bound else 0.0

class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score_em
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_tensor_f1 = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_tensor_all = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_formats = []
        reward_extra_info = defaultdict(list)
        length_answer = []
        num_search = []
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            em_score, f1_score, s_num, a_length = self.compute_score(
                solution_str=response_str,
                ground_truth=ground_truth,
            )
            
            format_score = format_reward(response_str)
            length_answer.append(a_length)
            num_search.append(s_num)

            score = em_score
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
                    
            # new york new york city
            score_f1 = f1_score
            if isinstance(score_f1, dict):
                reward_f1 = score_f1["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info['f1'].append(value)
            else:
                reward_f1 = score_f1

            reward_tensor_f1[i, valid_response_length - 1] = reward_f1
            reward_formats.append(format_score)
            
            all_score = em_score * 0.9 + format_score * 0.1
            if isinstance(all_score, dict):
                reward_all = all_score["score"]
                # Store the information including original reward
                for key, value in all_score.items():
                    reward_extra_info['all'].append(value)
            else:
                reward_all = all_score
            reward_tensor_all[i, valid_response_length - 1] = reward_all
            
        reward_format = sum(reward_formats) / len(reward_formats) if reward_formats else 0.0
        
        if return_dict:
            return {
                "reward_tensor_all": reward_tensor_all,
                "reward_format": reward_format,
                "reward_tensor": reward_tensor,
                "reward_tensor_f1": reward_tensor_f1,
                "reward_extra_info": reward_extra_info,
                "length_answer": torch.mean(torch.tensor(length_answer, dtype=torch.float32)),
                "num_search": torch.mean(torch.tensor(num_search, dtype=torch.float32)),
            }
        else:
            return reward_tensor_all, reward_format, reward_tensor, reward_tensor_f1, torch.mean(torch.tensor(length_answer, dtype=torch.float32)), torch.mean(torch.tensor(num_search, dtype=torch.float32))
