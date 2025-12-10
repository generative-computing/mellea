import asyncio
from datetime import datetime
import json
import os
import re
from tqdm.auto import tqdm

from utils.utils import *

INSTRUCTIONS = """Assume you are a human expert in grading predictions given by a model. You are given a question and a model prediction. Judge if the prediction matches the ground truth answer by following these steps:
1: Take it as granted that the Ground Truth is always correct.
2: If the Prediction indicates it is not sure about the answer, "score" should be "0"; otherwise, go the next step.
3: If the Prediction exactly matches the Ground Truth, "score" is 1.
4: If the Prediction does not exactly match the Ground Truth, go through the following steps and likely give a score as 0.
5: If the Ground Truth is a number, "score" is 1 if and only if the Prediction gives a number that almost exactly matches the ground truth.
6: If the Prediction is self-contradictory, "score" must be 0.
7: If the prediction is not answering the question, "score" must be 0.
8: If the prediction is a concise and correct summary of the ground truth, "score" is 1.
9: If ground truth contains a set of items, prediction must contain exactly same items for the score to be 1.
10: Otherwise, "score" is 0.

### Output a JSON blob with an "explanation" field explaining your answer as short as possible and an "score" field with value 1 or 0."""

IN_CONTEXT_EXAMPLES = """You should make the judgment based on provided examples.
Examples:
Question: "which company has higher eps, btu or cma?"
Ground Truth: "cma"
Prediction: "it is not possible to determine which company has a higher eps."
Output: {"score": 0, "explanation": "The prediction is not sure about the answer."}

Question: "who is taller, a or b?"
Ground Truth: "a"
Prediction: "The answer is a. a is of height 1.75 m and b is of height 1.82 m. So b is taller."
Output: {"score": 0, "explanation": "The prediction is self-contradictory."}

Question: "who among patricia place and lana clarkson is younger?"
Ground Truth: "lana clarkson"
Prediction: "lana clarkson (born april 5, 1962) is younger than patricia clarkson (born december 29, 1959)."
Output: {"score": 1, "explanation": "The prediction gives the correct answer."}

Question: "did lakers win more games than warriors in 2024-4?"
Ground Truth: "yes"
Prediction: "yes, lakers won more games than warriors in 2023-2024 season."
Output: {"score": 0, "explanation": "The prediction is not answering the question."}

Question: "what is the height of the building where citigroup is headquartered?"
Ground Truth: "151 m"
Prediction: "the height is 915 feet (279 m)."
Output: {"score": 0, "explanation": "The prediction, 151 m, does not match the ground truth, 279 m."}

Question: "who were the last 3 prime ministers of uk?"
Ground Truth: "rishi sunak, liz truss, and boris johnson"
Prediction: "1. liz truss, 2. boris johnson, 3.theresa may"
Output: {"score": 0, "explanation": "The prediction does not contain item, rishi sunak, that is in the grount truth."}

Question: "who authored the taming of the shrew?"
Ground Truth: "william shakespeare"
Prediction: "william shakespeare"
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "who authored The Taming of the Shrew?"
Ground Truth: "william shakespeare"
Prediction: "w shakespeare"
Output: {"score": 1, "explanation": "The prediction matches the ground truth as w is the abbreviation of william."}

Question: "what is the state bird of california?"
Ground Truth: "california quail"
Prediction: "california valley quail"
Output: {"score": 1, "explanation": "The prediction matches the ground truth as california quail is also known as california valley quail."}

Question: "what country is maharashtra metro rail corporation limited located in?"
Ground truth: "india"
Prediction: "indianapolis"
Output: {"score": 0, "explanation": "The prediction doesn't match the ground truth answers."}

Question: "what was wolves's record in their last game in eng-premier league?"
Ground Truth: "w"
Prediction: "wolves's record in their last game in the eng-premier League was a 2-1 win over fulham."
Output: {"score": 1, "explanation": "The prediction correctly answers the result is a win."}

Question: "how many seconds is 3 minutes 15 seconds?"
Ground Truth: "195 seconds"
Prediction: "3 minutes 15 seconds is 195 seconds."
Output: {"score": 1, "explanation": "The prediction matches ground truth as it correctly states the number of seconds asked in the question."}

Question: "on which days did am distribute dividends in the last year?"
Ground Truth: "2023-01-24, 2023-04-25, 2023-07-25, 2023-10-24"
Prediction: "the company distributed dividends on october 24, 2023 and january 24, 2023."
Output: {"score": 0, "explanation": "The prediction partially matches the ground truth, but it is incomplete."}

Question: "on which days did xxx distribute dividends in the last year?"
Ground Truth: "2023-01-13, 2023-03-25, 2023-11-21"
Prediction: "xxx distributed dividends on 1. 2023-01-13, 2. 2023-03-25, 3. 2023-10-21."
Output: {"score": 0, "explanation": "The prediction contains an item, 2023-10-21, which is not in ground truth."}

Question: "who are current band members of maroon 5?"
Ground Truth: "adam Levine, jesse carmichael, james valentine, matt flynn, pj morton, sam farrar"
Prediction: "they are jesse carmichael, matt flynn, adam levine, pj morton, sam farrar, and james valentine."
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "which movies comprise the matrix franchise?"
Ground Truth: "the matrix, the matrix reloaded, the matrix revolutions, the matrix resurrections"
Prediction: "the matrix, the matrix reloaded, the matrix revolutions, the animatrix, and the matrix resurrections."
Output: {"score": 0, "explanation": "The prediction covers more items than what are given by the ground truth."}

Question: "how deep is the deepest lake of new york?"
Ground Truth: "618 ft"
Prediction: "the deepest lake in new york is seneca lake, with a depth of 618.23 feet."
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "what is the closing price of meta yesterday?"
Ground Truth: "$310.17"
Prediction: "310.2"
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "what is the current market cap of appl?"
Ground Truth: "2.81 trillion"
Prediction: "2.667 trillion"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "what is the current pe ratio of appl?"
Ground Truth: "28.3"
Prediction: "the current pe ratio of apple is 26.66"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "how much is tesla's stock price down from its all-time high?"
Ground Truth: "$221.83"
Prediction: "209.52"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "what is the length of amazon river?"
Ground Truth: "over 4000 miles"
Prediction: "the length of amazon river is 4,000 miles"
Output: {"score": 0, "explanation": "The prediction does not say Amazon River is longer than 4000 miles."}

Question: "how many copies x were sold?"
Ground Truth: "2 million."
Prediction: "it is over 2 million"
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what is the population of country x?"
Ground Truth: "3,576,873"
Prediction: "the population of country x is 3.3 million."
Output: {"score": 0, "explanation": "The prediction, 3.3 M, does not match the number, 3.6 M, in ground truth."}

Question: "what is the current market value of stock x?"
Ground Truth: "$2,237,578,268"
Prediction: "$2.16 billion."
Output: {"score": 0, "explanation": "The prediction, 2.16 B, does not match the number, 2.2 B, in ground truth."}

Question: "how old is michael jordan?"
Ground Truth: "61"
Prediction: "mj was born on February 17, 1963."
Output: {"score": 0, "explanation": "The prediction does not explicitly say the age."}

Question: "how long did amy teach as a teacher?"
Ground Truth: "24 years and 255 days"
Prediction: "25 years."
Output: {"score": 0, "explanation": "The prediction does not exactly match the ground truth."}

Question: "at what age did usher perform in the half time show of superbowl?"
Ground Truth: "45 years old"
Prediction: "usher has not performed at a super bowl halftime show."
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what year did olivia rodrigo win the grammys best new artist award?"
Ground Truth: "2022"
Prediction: "sure! here's an answer in three words or fewer: 2022 grammy awards."
Output: {"score": 1, "explanation": "The prediction answers the correct number of the year."}

Question: "how many times have zendaya and tom holland starred in a film together?"
Ground Truth: "3"
Prediction: "they have starred in 3 films together: spider-man: homecoming, spider-man: far from home, and spider-man: no way home."
Output: {"score": 1, "explanation": "The prediction gives the right number and it is reasonable."}

Question: "who is younger, timothee chalamet or tom holland?"
Ground Truth: "tom holland"
Prediction: "timothée chalamet is younger than tom holland."
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "who had more number one hits on the us billboard, a or b?"
Ground Truth: "a had more number one hits on the us billboard than b, with 20 number one hits compared to b's 15."
Prediction: "a"
Output: {"score": 1, "explanation": "The prediction is a concise and correct summary of the ground truth."}

Question: "what is xxx's birthdate?"
Ground Truth: "1996-01-01."
Prediction: "02/01/1996"
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what was the worldwide box office haul for movie x?"
Ground Truth: "101756123."
Prediction: "102 million"
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "how much has spotify's user base increased by since 2020 in na?"
Ground Truth: "spotify's user base increased by 34 million since 2020."
Prediction: "spotify's north american user base increased from 36 million in 2020 to 85 million by 2021"
Output: {"score": 0, "explanation": "The prediction is not answering the question as it only gives the increase from 2020 to 2021."}
"""


# tokenizer = LlamaTokenizerFast.from_pretrained("tokenizer")

def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES

def parse_response(response: str):
    """
    Return a tuple of (explanation, score) from the response, 
    where score is 0 if the prediction is wrong, 1 if the prediction is correct.

    Need to handle
    Corner case 1:
        {"explanation": ...}
        Wait, no! I made a mistake. The prediction does not exactly match the ground truth. ...
        {...}

    Corner case 2:
        {"score": 0, "explanation": "The prediction does not contain item, nick "goose" bradshaw, that is in the ground truth."}
        return a tuple of (explanation, score)
    """
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"
    try:
        score = -1
        # Pattern to match the score
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score != 0 and score != 1:
                raise Exception("bad score: " + response)
        else:
            return "Parse Err: Score not found", -1

        # Pattern to match the explanation
        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        if explanation_match:
            explanation = explanation_match.group(1)
            return explanation, score
        else:
            return text, score
    except Exception as e:
        print(f"Parsing Error with resp: {response}")
        print(f"Error: {e}")
        return response, -1


class LLM_Evaluator:
    """LLM as a Judge."""

    async def evaluate_response(self, query: str, ground_truth: str, prediction: str) -> str:
        """Asynchronous function to evaluate a single answer."""
        system_prompt = get_system_message()
        user_message = f"""
            Question: {query}
            Ground truth: {ground_truth}
            Prediction: {prediction}
        """

        return await generate_eval_response(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            # timeout=180,
            response_format={"type": "json_object"},
        )

    async def evaluate_responses_async(self, queries: List[str], ground_truths: List[str], predictions: List[str]) -> List[str]:
        # Run all requests asynchronously
        responses = await asyncio.gather(*[self.evaluate_response(query, ground_truth, prediction)
                                           for query, ground_truth, prediction in zip(queries, ground_truths, predictions)])

        return responses

    def evaluate_responses(self, queries: List[str], ground_truths: List[str], predictions: List[str]) -> List[str]:
        """
        Wrapper to run async batch_generate_answer in a synchronous context.
        """
        return asyncio.run(self.evaluate_responses_async(queries, ground_truths, predictions))


def evaluate_predictions(queries, ground_truths_list, predictions, evaluation_model_name, batch_size):
    """
    Evaluates the predictions generated by a model against ground truth answers.

    Args:
    queries (List[str]): List of queries.
    ground_truths_list (List[List[str]]): List of lists of ground truth answers. 
        Note each query can have multiple ground truth answers.
    predictions (list): List of predictions generated by the model.
    evaluation_model_name (str): Name of the evaluation model.

    Returns:
    dict: A dictionary containing evaluation results.
    """

    evaluator = LLM_Evaluator()
    n_miss, n_correct = 0, 0
    history = [None for _ in range(len(predictions))]
    for i in tqdm(range(0, len(predictions) + batch_size, batch_size), desc="Evaluating Predictions"):
        # Process a batch of queries
        end = min(i + batch_size, len(predictions))
        batch_queries = queries[i: end]
        batch_ground_truths = ground_truths_list[i: end]
        batch_predictions = predictions[i: end]

        # Handle "I don't know" cases (skip evaluation)
        batch_indices_to_skip = {idx for idx, pred in enumerate(
            batch_predictions) if "i don't know" in pred.lower()}
        for idx in batch_indices_to_skip:
            history[i + idx] = {"id": i + idx, "score": 0,
                                "explanation": "I don't know."}
            n_miss += 1

        # Remove "I don't know" cases from batch
        batch_ids = [idx for idx in range(
            len(batch_queries)) if idx not in batch_indices_to_skip]
        batch_queries = [batch_queries[idx] for idx in range(
            len(batch_queries)) if idx not in batch_indices_to_skip]
        batch_ground_truths = [batch_ground_truths[idx] for idx in range(
            len(batch_ground_truths)) if idx not in batch_indices_to_skip]
        batch_predictions = [batch_predictions[idx] for idx in range(
            len(batch_predictions)) if idx not in batch_indices_to_skip]

        if not batch_queries:  # Skip empty batches
            continue

        # Use Llama 3 to evaluate batch
        batch_responses = evaluator.evaluate_responses(
            batch_queries, batch_ground_truths, batch_predictions)

        # Parse responses and determine accuracy
        for idx, response in zip(batch_ids, batch_responses):
            _, accuracy = parse_response(response)
            try:
                reason = maybe_load_json(response, force_load=False, default_output={
                                        "explanation": ""}).get('explanation', "")
            except Exception as e:
                print(response)
                raise e
            history[i + idx] = {"idx": str(i + idx),
                                "score": accuracy, "explanation": reason}
            if accuracy == 1:
                n_correct += 1

    # Compute final scores
    n = len(predictions)
    results = {
        "score": ((2 * n_correct + n_miss) / n - 1) * 100.0,
        "accuracy": (n_correct / n) * 100.0,
        "hallucination": ((n - n_correct - n_miss) / n) * 100.0,
        "missing": (n_miss / n) * 100.0,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_hallucination": n - n_correct - n_miss,
        "total": n,
        "llm": MODEL_NAME,
        "emb_llm": EMB_MODEL_NAME,
        "eval_llm": EVAL_MODEL_NAME
    }
    logger.info(results)
    return results, history