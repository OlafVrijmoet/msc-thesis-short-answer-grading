# Datasets

The study uses five public ASAG datasets. None of them are redistributed in this
repository: ASAP-sas is bound by Kaggle competition rules, and the redistribution
terms of the others are not explicit enough to ship copies here. Each one is a
short download, and the pipeline rebuilds everything else.

All student answers in these datasets come from published research corpora. No new
student data was collected for this project.

| Dataset | Answers | Grading scale | Source |
|---|---|---|---|
| ASAP-sas | 17,207 | 0-2 or 0-3 | [Kaggle: Automated Student Assessment Prize, short answer scoring](https://www.kaggle.com/c/asap-sas) |
| SciEntsBank | 10,804 | correct / incorrect | [SemEval-2013 Task 7 (SRA corpus)](https://aclanthology.org/S13-2045/) |
| Beetle | 3,941 | correct / incorrect | [SemEval-2013 Task 7 (SRA corpus)](https://aclanthology.org/S13-2045/) |
| Texas-2011 | 2,273 | 0-5, two graders averaged | [Mohler, Bunescu & Mihalcea 2011](https://aclanthology.org/P11-1076/), mirror on [Hugging Face](https://huggingface.co/datasets/nkazi/MohlerASAG) |
| Neural Network Course | 646 | 0-2 | [DigiKlausur/ASAG-Dataset](https://github.com/DigiKlausur/ASAG-Dataset) (MPL-2.0) |

## Where to place the files

The raw ingestion step (`RAW_PHASE` in `main.py`) expects this layout under
`data/raw/data/`:

```
data/raw/data/
├── ASAP_sas/ASAP_sas.tsv          # train.tsv from the Kaggle competition, renamed
├── beetle/                        # Beetle XML from the SRA corpus
│   ├── train/
│   └── test/{test-unseen-answers,test-unseen-questions}/
├── sciEntsBank/                   # SciEntsBank XML from the SRA corpus
│   ├── train/
│   └── test/{test-unseen-answers,test-unseen-domains,test-unseen-questions}/
├── Texas.csv                      # Texas-2011 as one CSV (question, student answer, grades)
└── nn_course.csv                  # asag_dataset.csv from DigiKlausur, renamed
```

Running the phases in `main.py` from the repository root (toggle them in
`constants.py`) then fills `data/raw/data/` with per-dataset CSVs and generates the
standardized, split, and processed stages the models read from.

## Processing overview

1. **Raw** (`data/raw/`): XML and TSV sources converted to CSV.
2. **Standardized** (`data/standardized/`): one schema for every dataset
   (`student_answer`, `reference_answer`, `assigned_points`, `max_points`,
   `domain`, ...). Grades normalized to [0, 1] by the question's maximum points.
3. **Splits** (`data/splits/`): 70-20-10 train/test/validation, plus concatenated
   datasets and concatenated domains with leave-one-out variants.
4. **Model preparation**: spelling correction, tokenization for BERT, and word
   embeddings for the embedding-based models.

ASAP-sas has no reference answers; the standardization step synthesizes one per
question by taking the full-score student answer closest to the TF-IDF centroid of
all full-score answers for that question.
