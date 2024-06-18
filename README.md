# Tune Llama 3 for a 95% Accuarate SQL Agent with Lamini

This repo and notebook `meta-lamini.ipynb` demonstrate how to tune Llama 3 to generate valid SQL queries to answer your questions.

In this notebook we'll be using Lamini, an integrated platform for LLM inference and tuning.

Head over to https://app.lamini.ai/account to get your free api key.

You can authenticate by writing the following to a file `~/.lamini/configure.yaml`

```
production:
    key: <YOUR-LAMINI-API-KEY>
```

This tuning tutorial uses the `nba_roster` sqlite database to tune a Llama 3 model.
