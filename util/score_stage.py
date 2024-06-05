import logging
from typing import AsyncIterator, Iterator, Union

from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode

from util.get_rubric import get_rubric
from util.make_llama_3_prompt import make_llama_3_prompt

logger = logging.getLogger(__name__)


class ScoreStage(GenerationNode):
    def __init__(self):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=150,
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        results = super().generate(
            prompt,
            output_type={"explanation": "str", "score": "int"},
            *args,
            **kwargs,
        )

        return results

    def preprocess(self, obj: PromptObject):
        obj.prompt = self.make_prompt(obj)
        logger.info(f"Scoring Stage Prompt:\n{obj.prompt}")

    async def process_results(self, results):
        async for result in results:
            # filter out results that are None
            if result is None:
                continue

            if result.response is None:
                logging.error(
                    f"Error scoring example {result.data.get_id()}: {result.error}"
                )
                continue

            result.data.result = {
                "example_id": result.data.get_id(),
                "prompt": result.data.get_prompt(),
                "response": result.data.response,
                "reference_response": result.data.get_response_json(),
                "is_exact_match": result.data.is_exact_match(result.data.response),
                "score": result.response["score"],
                "explanation": result.response["explanation"],
            }

            yield result

    def compute_score(self, score, is_exact_match):
        if not is_exact_match:
            return 1

        return score

    def make_prompt(self, example):
        response = example.data.format_response(example.response)

        system_prompt = "A large language model (LLM) is going to answer a question. "
        system_prompt += (
            "Your job is to score the answer, comparing it to a golden reference. "
        )
        system_prompt += "You are an expert scorer.\n\n"
        user_prompt = "Rate the answer using a score from 1 (lowest match) to 5 (highest match).\n"
        user_prompt += get_rubric()
        user_prompt += "Use the full range. Read the gold answer carefully. "
        user_prompt += "Explain your score in 2-3 sentences, then assign a score. "
        user_prompt += 'Output your score as a JSON object in the format {"explanation" : str, "score" : int}\n'
        user_prompt += "Use single quotes within your explanation. End your explanation with a double quote.\n"
        user_prompt += "Prefer answers that are most similar to the gold answer, even if the gold answer refused to answer the question.\n\n"
        user_prompt += (
            f"========== question =========\n{example.data.get_question()}\n\n"
        )
        user_prompt += (
            f"========== gold answer =========\n{example.data.get_response()}\n\n"
        )
        user_prompt += f"========== model answer =========\n{response}\n\n"
        user_prompt += "=" * 40 + "\n\n"
        user_prompt += f"How would you score the model's answer compared to the gold answer (using the 1-5 scale defined above)?"

        prompt = make_llama_3_prompt(user_prompt, system_prompt)
        return prompt
