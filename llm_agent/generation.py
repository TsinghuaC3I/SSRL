import torch
import re
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import random
import time
import json
import requests
import hashlib
import pickle
from pathlib import Path
import numpy as np
import torch.nn.functional as F


def format_search_snippets(snippets):
    """
    Process search result snippets to remove ellipses and improve readability.
    
    Args:
        snippets: List of snippet strings or a single multi-line string
    
    Returns:
        List of formatted snippets
    """
    # If input is a string, split by newlines
    if isinstance(snippets, str):
        snippets = [s.strip() for s in snippets.split('\n') if s.strip()]
    
    formatted_snippets = []
    
    for snippet in snippets:
        # Remove leading/trailing whitespace
        cleaned = snippet.strip()
        
        # Remove ellipses at the beginning or end
        cleaned = re.sub(r'^\.\.\.+\s*', '', cleaned)
        cleaned = re.sub(r'\s*\.\.\.+$', '', cleaned)
        
        # Replace internal ellipses with proper punctuation
        # If ellipsis appears between sentences, replace with period
        cleaned = re.sub(r'\.\s*\.\.\.+\s*([A-Z])', r'. \1', cleaned)
        
        # If ellipsis appears mid-sentence, replace with comma or remove
        cleaned = re.sub(r'\s+\.\.\.+\s+', ' ', cleaned)
        
        # Ensure proper sentence ending
        if cleaned and not cleaned[-1] in '.!?':
            cleaned += '.'
        
        # Capitalize first letter if needed
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        formatted_snippets.append(cleaned)
    
    return formatted_snippets

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    temperature: float = 0.8
    topk: int = 5
    # Cache configuration
    cache_enabled: bool = False
    cache_dir: str = "./search_cache"
    cache_ttl_days: int = 7  # Cache time-to-live in days
    stop_on_first_answer: bool = True  # New flag to control stopping behavior
    # Entropy-guided search configuration
    entropy_guided: bool = False
    entropy_threshold: float = 0.1
    entropy_window_size: int = 5
    type: str = 'llama-3b'


class SearchCache:
    """Simple file-based cache for search results."""
    
    def __init__(self, cache_dir: str, ttl_days: int = 7):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_days * 24 * 60 * 60
        self._stats = {"hits": 0, "misses": 0}
    
    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key from the query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, query: str, type: str) -> Optional[str]:
        """Retrieve cached result if available and not expired."""
        cache_key = self._get_cache_key(query)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                
                # Load cached result
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                self._stats["hits"] += 1
                # llama specific
                result = cached_data.get('result').replace('Doc ', '')
                result_list = result.split('\n')
                result_list = [line.strip() for line in result_list if line.strip()]
                result_list = [res.replace(':', '.', 1) for res in result_list if res] 
                
                formatted_list = []
                for res in result_list:
                    if res:
                        # Remove any numbered list format (1., 2., etc.) and leading dashes
                        cleaned = re.sub(r'^\d+\.\s*', '', res)  # Remove "1. ", "2. ", etc.
                        cleaned = re.sub(r'^-\s*', '', cleaned)  # Remove existing "- "
                        if cleaned:
                            if type == 'qwen-7b':
                                formatted_list.append(f"- {cleaned}")
                if 'llama' in type:
                    formatted_list = result_list
                search_number = random.randint(0, len(formatted_list) - 1)
                formatted_list = formatted_list[:search_number + 1]  # Limit to random number of results
                result_list = format_search_snippets(formatted_list)
                if type == 'qwen-3b':
                    return ' '.join(result_list)
                return '\n'.join(result_list)
            except Exception as e:
                print(f"Error loading cache for query '{query}': {e}")
                self._stats["misses"] += 1
                return None
        
        self._stats["misses"] += 1
        return None
    
    def set(self, query: str, result: str) -> None:
        """Store search result in cache."""
        cache_key = self._get_cache_key(query)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cached_data = {
                'query': query,
                'result': result,
                'timestamp': time.time()
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            print(f"Error saving cache for query '{query}': {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"Error deleting cache file {cache_file}: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "total": total,
            "hit_rate": hit_rate
        }
    
    def clean_expired(self) -> int:
        """Remove expired cache entries."""
        removed = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > self.ttl_seconds:
                    cache_file.unlink()
                    removed += 1
            except Exception as e:
                print(f"Error cleaning cache file {cache_file}: {e}")
        
        return removed


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        # logger: Tracking,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        # self.logger = logger
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        
        # Initialize search cache
        if config.cache_enabled:
            self.search_cache = SearchCache(
                cache_dir=config.cache_dir,
                ttl_days=config.cache_ttl_days
            )
            print(f"Search cache initialized at {config.cache_dir} with TTL of {config.cache_ttl_days} days")
        else:
            self.search_cache = None
            print("Search cache disabled")

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses with validation."""
        try:
            tokenized = self.tokenizer(
                responses, 
                add_special_tokens=False, 
                return_tensors='pt', 
                padding="longest"
            )['input_ids']
            
            # Validate token IDs
            vocab_size = self.tokenizer.vocab_size
            if torch.any(tokenized >= vocab_size) or torch.any(tokenized < 0):
                # Filter out invalid tokens by replacing them with pad token
                tokenized = torch.where(
                    (tokenized >= 0) & (tokenized < vocab_size),
                    tokenized,
                    self.tokenizer.pad_token_id
                )
            
            return tokenized
            
        except Exception as e:
            print(f"Error in batch tokenization: {e}")
            # Fallback: return tensor of pad tokens
            max_len = max(len(r) for r in responses) if responses else 1
            return torch.full((len(responses), max_len), self.tokenizer.pad_token_id, dtype=torch.long)

    def _postprocess_responses(self, responses: torch.Tensor, is_last_turn: bool = False, no_process: bool = False) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        
        if no_process:
            return responses, responses_str

        # In the last turn with stop_on_first_answer enabled, prioritize stopping at answer
        if is_last_turn and getattr(self.config, 'stop_on_first_answer', True):
            processed_responses = []
            for resp in responses_str:
                if '</answer>' in resp:
                    # Stop at answer even if search appears first
                    processed_responses.append(resp.split('</answer>')[0] + '</answer>')
                elif '</search>' in resp:
                    # Only stop at search if no answer is present
                    processed_responses.append(resp.split('</search>')[0] + '</search>')
                else:
                    processed_responses.append(resp)
            responses_str = processed_responses
        else:
            # Normal behavior: stop at first occurrence of either tag
            responses_str = [resp.split('</search>')[0] + '</search>'
                     if '</search>' in resp 
                     else resp.split('</answer>')[0] + '</answer>'
                     if '</answer>' in resp 
                     else resp
                     for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)

        return new_rollings

    def _concatenate_with_padding(self,
                prompt: torch.Tensor,
                response: torch.Tensor,
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        if info is not None:
            tensors.append(info)

        concatenated = torch.cat(tensors, dim=1)
        
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)

        return padded_tensor

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids is not None:
            responses = self._concatenate_with_padding(
                right_side['responses'],
                cur_responses,
                next_obs_ids,
                pad_to_left=False
            )
        else:
            responses = self._concatenate_with_padding(
                right_side['responses'],
                cur_responses,
                pad_to_left=False
            )
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        input_batch_size = active_batch.batch['input_ids'].shape[0]
        
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        padded_active_batch.meta_info = active_batch.meta_info
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, search_mode, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, :]}
        original_right_side = {'responses': initial_input_ids[:, []]}
        trajectory_turns = [0 for _ in range(gen_batch.batch['input_ids'].shape[0])]
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        
        # Track if we're in the last turn (after max_turns iterations)
        is_last_turn = False
        
        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            rollings_active.meta_info = rollings.meta_info
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info            
            log_probs = gen_output.batch['rollout_log_probs']
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'], is_last_turn=False)
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, active_mask, self.config.type
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )

            for idx in range(len(dones)):
                if trajectory_turns[idx] == 0 and dones[idx] == 1:
                    trajectory_turns[idx] = step + 1

        # Mark that we're entering the last turn
        is_last_turn = True
        
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            rollings_active.meta_info = rollings.meta_info
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'], is_last_turn=True)
            
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, active_mask, self.config.type
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )

            meta_info['turns_stats'] = turns_stats.tolist()
            meta_info['active_mask'] = active_mask.tolist()
            meta_info['valid_action_stats'] = valid_action_stats.tolist()
            meta_info['valid_search_stats'] = valid_search_stats.tolist()

            # Record completion turns for remaining active samples
            for idx in range(len(dones)):
                if trajectory_turns[idx] == 0:
                    trajectory_turns[idx] = step + 2
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        print("Interaction Turns Statistics:")
        for turns in range(1, self.config.max_turns + 2):
            count = (torch.tensor(trajectory_turns) == turns).sum().item()
            print(f"Finish at the {turns}-th turn: {count}")

        return self._compose_final_output(original_left_side, original_right_side, meta_info), trajectory_turns

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions, active_mask=None, type='llama-3b', do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries, type)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    if type == 'llama-8b':
                        next_obs.append(f'\n<information>\n{search_results.pop(0).strip()}\n</information>\n')
                    if type == 'llama-3b':
                        next_obs.append(f'\n\n<information>\n{search_results.pop(0).strip()}\n</information>\n\n')
                    if type == 'qwen-3b':
                        next_obs.append(f'<information>{search_results.pop(0).strip()}</information>')
                    if type == 'qwen-7b':
                        next_obs.append(f'\n\n<information>\n{search_results.pop(0).strip()}\n</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)

        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries, type) -> str:
        """
        Batchified search for queries with caching support.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        all_search_result = ['No information available' for _ in range(len(queries))]
        queries_to_search = []
        query_indices = []
        
        # Check cache first
        if self.search_cache:
            for idx, query in enumerate(queries):
                cached_result = self.search_cache.get(query, type)
                if cached_result is not None:
                    all_search_result[idx] = cached_result
                else:
                    queries_to_search.append(query)
                    query_indices.append(idx)
        else:
            queries_to_search = queries
            query_indices = list(range(len(queries)))
        
        # Search for uncached queries
        if queries_to_search:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self._search, queries_to_search[i], type, i) for i in range(len(queries_to_search))]
                for future in as_completed(futures):
                    try:
                        result, search_idx = future.result()
                        original_idx = query_indices[search_idx]
                        all_search_result[original_idx] = result
                        
                        # Cache the result
                        if self.search_cache:
                            self.search_cache.set(queries_to_search[search_idx], result)
                    except Exception as e:
                        continue

        return all_search_result

    def search_text_by_text(self, query, type, retry_attempt=3):
        """Perform web search with retry logic."""
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query,
            "num": 3
        })
        headers = {
            'X-API-KEY': 'Your key',
            'Content-Type': 'application/json'
        }
        
        for i in range(retry_attempt):
            try:
                search = requests.request("POST", url, headers=headers, data=payload)
                search = json.loads(search.text)
                search_result = search["organic"]
                search_texts = []
                for item in search_result:
                    text_data = ''
                    if 'snippet' in item:
                        text_data += format_search_snippets(item['snippet'])[0]
                    print(f"Text: {text_data}")
                    print(f"Type: {type}")
                    search_texts.append(text_data)
                search_number = random.randint(0, len(search_texts) - 1)
                # qwen 25 3b inst
                # return ' '.join([f"{i}. {doc}" for i, doc in enumerate(search_texts[:search_number + 1]) if doc.strip()])
                if 'llama' in type:
                    return '\n'.join([f"{i+1}. {doc}" for i, doc in enumerate(search_texts[:search_number + 1]) if doc.strip()])
                elif type == 'qwen-3b':
                    return ' '.join([f"{doc}" for i, doc in enumerate(search_texts[:search_number + 1]) if doc.strip()])
                elif type == 'qwen-7b':
                    return '\n'.join([f"- {doc}" for i, doc in enumerate(search_texts[:search_number + 1]) if doc.strip()])
            except Exception as e:
                print(f"Attempt {i + 1} failed: {e}")
                if i < retry_attempt - 1:
                    time.sleep(2)  # Wait 2 seconds before retry
                else:
                    print("All retries failed.")
                    return 'No information available'
                
    def _search(self, query, type, index):
        """Wrapper for search with index tracking."""
        doc_texts = self.search_text_by_text(query, type)
        return doc_texts, index

    def clear_cache(self):
        """Clear the search cache."""
        if self.search_cache:
            self.search_cache.clear()
            print("Search cache cleared")
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.search_cache:
            return self.search_cache.get_stats()
        return None

    def run_llm_loop_entropy(self, gen_batch, search_mode, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Simplified entropy-guided LLM loop."""
        
        # Initialize
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        entropy_histories = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        rollings = gen_batch
        original_left_side = {'input_ids': initial_input_ids[:, :]}
        original_right_side = {'responses': initial_input_ids[:, []]}
        trajectory_turns = [0 for _ in range(gen_batch.batch['input_ids'].shape[0])]
        total_search = 0
        real_search = 0
        
        # Main loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
                
            # Cut to effective length before generation
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # Generate full trajectory for active samples
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            rollings_active.meta_info = rollings.meta_info
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            
            # Get full responses and log probs (only for active samples)
            full_responses_active = self.tokenizer.batch_decode(gen_output.batch['responses'], skip_special_tokens=True)
            log_probs_active = gen_output.batch.get('rollout_log_probs')
            
            # Process each response
            truncated_responses_active = []
            next_observations = []
            dones = []
            valid_action = []
            is_search = []
            
            # Keep track of active index
            active_idx = 0
            
            for idx, active in enumerate(active_mask):
                if not active:
                    next_observations.append('')
                    dones.append(1)
                    valid_action.append(0)
                    is_search.append(0)
                    continue
                    
                # Get response for this active sample
                response = full_responses_active[active_idx]
                sample_log_probs = log_probs_active[active_idx] if log_probs_active is not None else None
                active_idx += 1
                
                # Parse action
                action_match = re.search(r'<(search|answer)>(.*?)</\1>', response, re.DOTALL)
                
                if not action_match:
                    # Invalid action
                    truncated_responses_active.append(response)
                    next_observations.append(f'\nMy previous action is invalid. \
    If I want to search, I should put the query between <search> and </search>. \
    If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    continue
                
                action_type = action_match.group(1)
                action_content = action_match.group(2).strip()
                
                # Truncate at action end
                truncated = response[:action_match.end()]
                truncated_responses_active.append(truncated)
                
                if action_type == 'answer':
                    next_observations.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action_type == 'search':
                    total_search += 1
                    # Check entropy for search
                    info_match = re.search(r'<information>(.*?)</information>', response[action_match.end():], re.DOTALL)
                    next_info = info_match.group(1).strip() if info_match else 'No Info'
                    if self._should_use_real_search(action_content, response, sample_log_probs, entropy_histories[idx]):
                        real_search += 1
                        real_results = self.batch_search([action_content])
                        info_content = real_results[0] if real_results else 'No information available'
                        if type == 'qwen-3b':
                            next_observations.append(f'<information>{info_content.strip()}</information>')
                        elif type == 'llama-3b':
                            next_observations.append(f'\n\n<information>\n{info_content.strip()}\n</information>\n\n')
                        else:
                            next_observations.append(f'\n<information>\n{info_content.strip()}\n</information>\n')
                    else:
                        info_match = re.search(r'<information>(.*?)</information>', response[action_match.end():], re.DOTALL)
                        if info_match:
                            if type == 'qwen-3b':
                                next_observations.append(f'<information>{info_match.group(1).strip()}</information>')
                            elif type == 'llama-3b':
                                next_observations.append(f'\n\n<information>\n{info_match.group(1).strip()}\n</information>\n\n')
                            else:
                                next_observations.append(f'\n<information>\n{info_match.group(1).strip()}\n</information>\n')
                        else:
                            if type == 'qwen-3b':
                                next_observations.append(f'\n<information>\nNo information available\n</information>\n')
                            elif type == 'llama-3b':
                                next_observations.append(f'\n\n<information>\nNo information available\n</information>\n\n')
                            else:
                                next_observations.append(f'\n<information>\nNo information available\n</information>\n')
                    
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
            
            # Tokenize only active truncated responses
            truncated_ids_active = self._batch_tokenize(truncated_responses_active)
            
            # Use _example_level_pad to expand to full batch size
            truncated_ids, truncated_responses = self.tensor_fn._example_level_pad(
                truncated_ids_active, truncated_responses_active, active_mask
            )
            
            # Process observations
            next_obs_ids = self._process_next_obs(next_observations)
            
            # Update rolling state with truncated response + observation
            rollings = self._update_rolling_state(rollings, truncated_ids, next_obs_ids)
            
            # Update right side for final output
            original_right_side = self._update_right_side(original_right_side, truncated_ids, next_obs_ids)
            
            # Update active mask
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            
            # Update trajectory turns
            for idx in range(len(dones)):
                if trajectory_turns[idx] == 0 and dones[idx] == 1:
                    trajectory_turns[idx] = step + 1
            
            print(f"Step {step + 1}: Active samples = {active_mask.sum().item()}")
        
        # Final generation for remaining active samples
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            rollings_active.meta_info = rollings.meta_info
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            
            # Process final responses (only active samples)
            final_responses_active = self.tokenizer.batch_decode(gen_output.batch['responses'], skip_special_tokens=True)
            
            # Process active responses
            processed_responses_active = []
            for response in final_responses_active:
                # Prioritize answer in final turn
                if '</answer>' in response:
                    processed = response.split('</answer>')[0] + '</answer>'
                elif '</search>' in response:
                    processed = response.split('</search>')[0] + '</search>'
                else:
                    processed = response
                processed_responses_active.append(processed)
            
            # Tokenize only active responses
            final_ids_active = self._batch_tokenize(processed_responses_active)
            
            # Use _example_level_pad to expand to full batch size
            final_ids, _ = self.tensor_fn._example_level_pad(final_ids_active, processed_responses_active, active_mask)
            
            # Update final output
            original_right_side = self._update_right_side(original_right_side, final_ids)
            
            # Record completion turns
            for idx in range(len(active_mask)):
                if trajectory_turns[idx] == 0:
                    trajectory_turns[idx] = step + 2
        
        # Print statistics
        print("Interaction Turns Statistics:")
        for turns in range(1, self.config.max_turns + 2):
            count = (torch.tensor(trajectory_turns) == turns).sum().item()
            print(f"Finish at the {turns}-th turn: {count}")
        
        print(f"Total searches: {total_search}, Real searches: {real_search}")
        return self._compose_final_output(original_left_side, original_right_side, meta_info), trajectory_turns

    def _should_use_real_search(self, search_content: str, full_response: str, log_probs: torch.Tensor, entropy_history: List[float]) -> bool:
        """Simplified entropy check."""
    
        if log_probs is None:
            return False
            
        # Calculate entropy for search content
        entropy = self._calculate_search_entropy(log_probs, full_response, search_content)
        if len(entropy_history) == 0:
            entropy_history.append(entropy)
            return True
        entropy_history.append(entropy)
        
        # Keep window size
        if len(entropy_history) > self.config.entropy_window_size:
            entropy_history.pop(0)
        
        # Check if entropy is increasing
        if len(entropy_history) >= 2:
            increase = entropy_history[-1] - entropy_history[-2]
            return increase > 0
        
        return False


    def _calculate_search_entropy(self, log_probs: torch.Tensor, full_response: str, search_content: str) -> float:
        """Calculate entropy for search content tokens."""
        # Find search content position in response
        search_match = re.search(r'<search>(.*?)</search>', full_response, re.DOTALL)
        if not search_match:
            return 0.0
        
        # Tokenize to find positions
        text_before_search = full_response[:search_match.start(1)]
        text_with_search = full_response[:search_match.end(1)]
        
        tokens_before = self.tokenizer(text_before_search, return_tensors='pt', add_special_tokens=False)['input_ids']
        tokens_with = self.tokenizer(text_with_search, return_tensors='pt', add_special_tokens=False)['input_ids']
        
        start_idx = tokens_before.shape[1] if tokens_before.shape[1] > 0 else 0
        end_idx = tokens_with.shape[1]
        
        # Extract and calculate entropy
        if start_idx < log_probs.shape[0] and end_idx <= log_probs.shape[0]:
            search_log_probs = log_probs[start_idx:end_idx]
            
            # Convert to probabilities and calculate entropy
            probs = torch.exp(search_log_probs)
            probs = torch.clamp(probs, min=1e-10, max=1.0)
            entropy_values = -probs * torch.log2(probs)
            
            return entropy_values.mean().item()
        
        return 0.0