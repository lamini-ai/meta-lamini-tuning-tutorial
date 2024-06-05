from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_pipeline import GenerationPipeline
from tqdm import tqdm


async def evaluate_model(model, dataset, args):

    results = await run_evaluation_pipeline(model, dataset, args)

    print("Total results:", len(results))

    return results


async def run_evaluation_pipeline(model, dataset, args):
    data_slice = slice_dataset(dataset, args)

    results = EvaluationPipeline(model, dataset).call(data_slice)

    result_list = []

    pbar = tqdm(desc="Saving results", unit=" results")
    async for result in results:
        result_list.append(result)
        pbar.update()

    return result_list


async def slice_dataset(dataset, args):
    for index, example in enumerate(dataset):
        if index < args.max_examples:
            yield PromptObject(prompt="", data=example)


class EvaluationPipeline(GenerationPipeline):
    def __init__(self, model, dataset):
        super().__init__()
        self.model_stages = model.get_stages(dataset)

    def forward(self, x):
        for stage in self.model_stages:
            x = stage(x)
        return x
