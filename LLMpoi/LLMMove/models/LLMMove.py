import openai
import random
import os
import json
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except (ValueError, TypeError):
        return float('inf')
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 6371.0 * c


def parse_response(text):
    """Parse LLM response: try json.loads first, fallback to eval.
    Handles markdown code blocks like ```json ... ```."""
    text = text.strip()

    # Strip markdown code block if present
    if text.startswith('```'):
        first_nl = text.find('\n')
        if first_nl > 0 and first_nl < 20:
            text = text[first_nl:].strip()
        else:
            text = text[3:].strip()
        if text.endswith('```'):
            text = text[:-3].strip()

    if text.startswith('{') or text.startswith('['):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    try:
        return eval(text)
    except:
        raise ValueError(f"Cannot parse response: {text[:200]}")


def normalize_prediction(pred_list):
    """Ensure all POIIDs in the prediction list are strings for consistent comparison."""
    normalized = []
    for item in pred_list:
        if isinstance(item, (int, float)):
            normalized.append(str(int(item)))
        elif isinstance(item, str):
            normalized.append(item.strip())
        else:
            normalized.append(str(item))
    return normalized


class LLMMove():
    def __init__(self, llm_model='gpt-3.5-turbo', api_base=None, output_dir='./output'):
        self.llm_model = llm_model
        self.api_base = api_base
        self.output_dir = output_dir

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable")

        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        # Set explicit timeout to prevent hanging; disable openai-internal retries (we use tenacity)
        client_kwargs["timeout"] = 300.0
        client_kwargs["max_retries"] = 0

        from openai import OpenAI
        self.client = OpenAI(**client_kwargs)

    def _call_llm(self, messages):
        """Call the LLM directly (no retry wrapper, which can cause issues on some endpoints)."""
        return self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0,
            timeout=300,
        )

    def run(self, data, datasetName):
        self.datasetName = datasetName
        self.longs, self.recents, self.targets, self.poiInfos, self.traj2u = data
        poiList = list(self.poiInfos.keys())
        hit1 = 0
        hit5 = 0
        hit10 = 0
        rr = 0
        valid = 0
        err = list()
        api_errors = 0

        for trajectory, groundTruth in tqdm(self.targets.items()):
            try:
                seed_value = int(trajectory)
            except ValueError:
                continue
            random.seed(seed_value)
            negSample = random.sample(poiList, 100)
            candidateSet = negSample + [groundTruth[0]]
            try:
                prediction = self.runeach(trajectory, candidateSet, groundTruth)
                gt_id = str(groundTruth[0]).strip()
                pred_ids = normalize_prediction(prediction)

                if gt_id in pred_ids:
                    index = pred_ids.index(gt_id) + 1
                    if index == 1:
                        hit1 += 1
                    if index <= 5:
                        hit5 += 1
                    hit10 += 1
                    rr += 1 / index
                else:
                    err.append(int(trajectory))

                # ValidRatio: all predicted POIIDs must exist in the POI set
                if all(pid in poiList for pid in pred_ids):
                    valid += 1
            except Exception as e:
                api_errors += 1
                if api_errors <= 5:
                    print(f"Error on trajectory {trajectory}: {repr(e)}")

        if api_errors > 5:
            print(f"... and {api_errors - 5} more API errors (suppressed)")

        num_trajectories = len(self.targets)
        acc1 = hit1 / num_trajectories if num_trajectories > 0 else 0
        acc5 = hit5 / num_trajectories if num_trajectories > 0 else 0
        acc10 = hit10 / num_trajectories if num_trajectories > 0 else 0
        mrr = rr / num_trajectories if num_trajectories > 0 else 0
        valid_ratio = valid / num_trajectories if num_trajectories > 0 else 0
        print(f'=== Results ({self.datasetName}) ===')
        print(f'  acc@1:      {acc1:.4f}')
        print(f'  acc@5:      {acc5:.4f}')
        print(f'  acc@10:     {acc10:.4f}')
        print(f'  mrr:        {mrr:.4f}')
        print(f'  validRatio: {valid_ratio:.4f}')
        print(f'  total test trajectories: {num_trajectories}')
        print(f'  API errors: {api_errors}')
        return acc1, acc10, mrr, valid_ratio

    def runeach(self, trajectory, candidateSet, groundTruth):
        u = self.traj2u[trajectory]
        long = self.longs[u]
        rec = self.recents[trajectory]
        path = '{}/LLMMove/{}/{}'.format(self.output_dir, self.datasetName, trajectory)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                saved = json.load(file)
                prediction = saved["response"]["recommendation"]
                return normalize_prediction(prediction)

        output = dict()
        mostrec = rec[-1][0]

        longterm = [(poi, self.poiInfos[poi]["category"]) for poi, _ in long]
        longterm = longterm[-40:]

        recent = [(poi, self.poiInfos[poi]["category"]) for poi, _ in rec]
        recent = recent[-5:]

        candidates = []
        for poi in candidateSet:
            dist = haversine_distance(
                self.poiInfos[poi]["latitude"],
                self.poiInfos[poi]["longitude"],
                self.poiInfos[mostrec]["latitude"],
                self.poiInfos[mostrec]["longitude"]
            )
            candidates.append((poi, dist, self.poiInfos[poi]["category"]))
        candidates.sort(key=lambda x: x[1])

        prompt = f"""\
<long-term check-ins> [Format: (POIID, Category)]: {longterm}
<recent check-ins> [Format: (POIID, Category)]: {recent}
<candidate set> [Format: (POIID, Distance, Category)]: {candidates}
Your task is to recommend a user's next point-of-interest (POI) from <candidate set> based on his/her trajectory information.
The trajectory information is made of a sequence of the user's <long-term check-ins> and a sequence of the user's <recent check-ins> in chronological order.
Now I explain the elements in the format. "POIID" refers to the unique id of the POI, "Distance" indicates the distance (kilometers) between the user and the POI, and "Category" shows the semantic information of the POI.

Requirements:
1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.
2. Consider the recent check-ins to extract users' current perferences.
3. Consider the "Distance" since people tend to visit nearby pois.
4. Consider which "Category" the user would go next for long-term check-ins indicates sequential transitions the user prefer.

Please organize your answer in a JSON object containing following keys:
"recommendation" (10 distinct POIIDs of the ten most probable places in <candidate set> in descending order of probability), and "reason" (a concise explanation that supports your recommendation according to the requirements). Do not include line breaks in your output.
"""
        output["prompt"] = prompt
        messages = [{"role": "user", "content": prompt}]

        # DEBUG: show prompt stats
        prompt_tokens = len(prompt) // 4  # rough estimate
        print(f"  [{trajectory}] Calling {self.llm_model}... (prompt ~{prompt_tokens} tokens)", flush=True)
        import time
        t0 = time.time()

        response = self._call_llm(messages)

        elapsed = time.time() - t0
        res_content = response.choices[0].message.content
        print(f"  [{trajectory}] Got response ({elapsed:.1f}s, ~{len(res_content)} chars)")
        parsed = parse_response(res_content)
        output["response"] = parsed

        if "recommendation" not in parsed:
            raise ValueError(f"LLM response missing 'recommendation' key: {res_content[:200]}")
        prediction = parsed["recommendation"]
        if not isinstance(prediction, list):
            raise ValueError(f"LLM 'recommendation' is not a list: {prediction}")
        if len(prediction) == 0:
            raise ValueError(f"LLM returned empty recommendation list")

        prediction = normalize_prediction(prediction)
        output["groundTruth"] = groundTruth[0]
        self.outputResponse(output, trajectory)
        return prediction

    def outputResponse(self, response, trajectory):
        path = '{}/LLMMove/{}/{}'.format(self.output_dir, self.datasetName, trajectory)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(response, indent='\t', ensure_ascii=False))
