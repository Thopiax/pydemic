# Coronavirus Mortality Rate

## Data Collection

### Diseases

  - SARS: no work needed - dataset from Kaggle
  - MERS: scrape and reconcile Excel files from Disease Outbreak News reports in WHO website.
  - Ebola: clean + aggregate dataset found

## Data Analysis

We want to develop a model to estimate the mortality curve of different infection outbreaks given the infection curve. Although we initially attempted to devise a regularised regression based on previous outbreak data, we noticed that we could simply use the current Coronavirus data to estimate the mortality curve.
