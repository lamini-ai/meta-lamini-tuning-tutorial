# Tune Llama 3 for a 95% accurate SQL model with Lamini

This repo and notebook `meta-lamini.ipynb` demonstrate how to tune Llama 3 to generate valid SQL queries with 95% accuracy.

In this notebook we'll be using Lamini, and more specificylly, Lamini Memory Tuning. 

Lamini is an integrated platform for LLM inference and tuning for the enterprise. Lamini Memory Tuning is a new way to embed facts into LLMs that improves factual accuracy and reduces hallucinations to previously unachievable levels. Learn more about Lamini Memory Tuning: https://www.lamini.ai/blog/lamini-memory-tuning

Please head over to https://app.lamini.ai/account to get your free api key.

You can authenticate by writing the following to a file `~/.lamini/configure.yaml`

```
production:
    key: <YOUR-LAMINI-API-KEY>
```

This tuning tutorial uses the `nba_roster` sqlite database to tune a Llama 3 model.