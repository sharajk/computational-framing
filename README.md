# computational-framing

This repository contains codes for the analysis, appendix and figures obtained in the manuscript:\
Computational framing analysis revisited: On LLMs for studying news coverage.

Find the full paper here:

### Authors

Sharaj Kunjar\*, Alyssa Hasegawa Smith, Tyler R Mckenzie, Rushali Mohbe, Samuel V Scarpino, Brooke Foucault Welles

\*corresponding author: sharaj.kunjar\@gmail.com

### Packages used

**R:** stringr, readr, dplyr, irr, Metrics, kableExtra, snakecase

**Python:** pandas, numpy, newspaper4k, networkx, collections, json, pickle, matplotlib, sklearn, os, re, anthropic, time, typing, torch, transformers, ollama, pathlib,yaml, datasets, evaluate, nltk, glob, sentence_transformers

# Structure of the Repository

## Dataset

Contains datasets prior to any form of analysis.

### Raw

1.  mediacloud_raw.csv: the list of all urls pertaining to news articles that talk about monkeypox within our timeline, as obtained from mediacloud
2.  webscrape_articles.ipynb: code to use the above dataset and newspaper4k to webscrape the texts for the articles
3.  newspaper_raw.csv: file with urls and texts for articles obtained from webscraping

### Pre-processing

Codes to pre-process the articles before further analysis.

1.  data_selection.ipynb: select articles based on basic exclusion/inclusion criteria
2.  mpox_news_data.csv: the resultant data from data selection above
3.  boilerplate.py: contains a dictionary of boilerplate content for removal
4.  dedup_boiler.ipynb: code to deduplicate texts and remove boilerplates
5.  mpox_news_data_95_word_overlap_threshold_for_annotation.csv: file with deduplicated articles
6.  mpox_news_data_95_word_overlap_threshold_for_annotation_boilerplate_removed.csv: file with deduplicated articles where boilerplate is also removed.

## Relevance

Contains codes and files for relevance classification (the processing step), where articles are chosen based on relevance to the Mpox.

1.  unlabelled_relevance.csv: all the articles that need to be classified for relevance
2.  relevance_codebook.{Rmd/pdf/json}: codebook used for relevance classification by manual coders

### Human

Files for manual coding of articles for relevance. Annotator 1 is the first author, Annotator 2 is the second author.

1.  A1_r1.csv: annotator 1, round 1 labels
2.  A1_r2:csv: annotator 1, round 2 labels
3.  A2_r1.csv: annotator 2, round 1 labels
4.  A2_r2:csv: annotator 2, round 2 labels
5.  inter_rater_metrics.qmd: contains code to compute inter-rater reliability scores

### SLM

Files for relevance classification with discriminative (BoW or encoder-only) models: Naive Bayes classifier, BERT, DeBERTa.

1.  classifiers.py: contains code to call and setup different models
2.  relevance.ipynb: contains code to deploy the classifiers as implemented above
3.  {naive_bayes/bert/deberta}\_test.csv: contains relevance labels from classifiers for the test set
4.  {naive_bayes/bert/deberta}\_all.csv: contains relevance labels from classifiers for the entire dataset of pre-processed articles.
5.  Scores.md: contains classifier performance scores evaluated against the manually labelled gold standard

### LLM

Files for relevance classification with generative (decoder-only) large language models: Llama, GPT-OSS and Claude Sonnet.

1.  relevance\_{llama,gpt,claude}.ipynb: code to call and run the models for relevance classification
2.  Predicted: folder containing all the predictions from the LLMs
3.  metrics.ipynb: code to calculate performance metrics for the LLMs
4.  Scores.csv: file containing all the performance metrics for the LLMs computed above

## Codebook

Contains files and code for frames codebook development based on a set of relevant (post-processing) articles.

1.  relevant_articles.csv: the set of all relevant articles
2.  100_relevant_articles.csv: a subset of 100 relevant articles used for codebook development

### Human

Contains the manual codebook (framing_codebook.md) developed through applied thematic analysis.

### SLM

1.  topic.ipynb: code to perform topic modelling on the subset of 100 articles using BERTopic
2.  BERT_topics.csv: file containing the list of topics and keywords identified by BERTopic

### LLM

Contains files and code for generating a codebook my simulating applied thematic analysis with Llama.

1.  llama_codebook.ipynb: code to deploy and run the model
2.  llama_initial_codes.csv: contains a list of initial codes identified by the model
3.  llama_themes.csv: contains a list of themes after clustering the initial codes
4.  llama_frames.csv: contains a list of frames after refining and describing the themes
5.  codebook_generation.R: code to take llama_frames.csv and convert it into a markdown codebook
6.  llama_frames_codebook.md: output from the previous code file
7.  llama3.1/8b-instruct-q8_0_zshot_llm_codebook.csv: contains llama's annotations of frames on a test set of 100 articles using the codebook developed by llama

## Framing

Contains files and code for conducting frame detection after articles have been processed and codebook has been developed.

1.  framing_codebook.md: codebook to be used for frame detection.

### Human

Contains annotated files for 100 article tests, labelled for presence of 7 frames, annotator 1 is the first author, annotator 2 is the third author.

1.  A1_r1.csv: annotator 1, round 1 labels
2.  A1_r2:csv: annotator 1, round 2 labels
3.  A2_r1.csv: annotator 2, round 1 labels
4.  A2_r2:csv: annotator 2, round 2 labels

### SLM

Files for frame detection with discriminative (BoW or encoder-only) models: Naive Bayes classifier, BERT, DeBERTa.

1.  labelled_frames_train.csv: 400 articles that are annotated for the presence of 7 frames by the first author. To be used for training classifiers.
2.  labelled_frames_test.csv: 100 articles that are annotated for the presence of 7 frames by the first author. To be used for testing classifiers.
3.  config.yml: set global parameters
4.  classifiers.py: code to initialize and setup the model
5.  framing_training.py: code to run the models and detect frames
6.  {naive_bayes, bert, deberta}\_test.csv: labels for frame presence on the test set of 100 articles for different classifiers

### LLM

Files for frame detection with generative (decoder-only) large language models: Llama, GPT-OSS and Claude Sonnet.

1.  labelled_frames_train.csv: 400 articles that are annotated for the presence of 7 frames by the first author. To be used for training classifiers.
2.  labelled_frames_test.csv: 100 articles that are annotated for the presence of 7 frames by the first author. To be used for testing classifiers.
3.  unlabelled_frames_test.csv: the test set without labels
4.  framing_codebook.json: codebook to be used for frame detection in JSON format
5.  framing_llama.ipynb: code to setup and deploy llama models using Ollama.
6.  framing_gpt.ipynb: code to setup and deploy gpt-oss-20b using huggingface and high performance computing
7.  framing_claude.ipynb: code to setup and deploy claude sonnet 4 using the Anthropic API
8.  framing_error.ipynb: code to run other implementation strategies using llama on Ollama.
9.  metrics.ipynb: code to calculate performance metrics for the generative models.
10. Folders:
    -   Error: Folder contains additional files to run the implementation strategies listed in framing_error

    -   Fine_tune: contains additional files to fine tune llama models in framing_llama

    -   Predicted: contains all the annotated files

    -   Scores: contains files with the performance metrics for all the models

## Labelled

Contains all the labelled files post analysis.

1.  formatting.qmd: code to format and collate all the annotations.
2.  relevance_all.csv: file containing relevance labels for all the pre-processed articles annotated by BERT
3.  Relevance_test.RData: file containing relevance labels for the test set of 500 articles.
4.  Framing_test.RData: file containing labels for frame detection labels for the test set of 100 articles.

## Notebooks

Contains files to summarize all our findings and generate the appendix for the paper.
