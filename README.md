# Career Clusters: Grouping Job Advertisements with Unsupervised Learning

## Project Description
This project explores a data-driven approach to clustering LinkedIn job postings based on key skills and description terms. The model is designed to recommend similar job postings to enhance users' search experience and expand the scope of relevant opportunities beyond their initial search query.

## Repository Contents
* [Setup Dependencies Jupyter Notebook](https://github.com/edgar-rosales95/ads-508-team-project/blob/main/Dependencies.ipynb)
* [Main Project Jupyter Notebook](https://github.com/edgar-rosales95/ads-508-team-project/blob/main/ADS_508_Final_Notebook.ipynb)
  
## Data Sources
The three datasets are retrieved from Kaggle and stored in a public AWS S3 bucket (s3://linkedin-postings) which will establish a connection with AWS Sagemaker.
* [postings.csv](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings?select=postings.csv)
* [salaries.csv](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
* [job_skills.csv](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings?select=jobs)

## Tools Used
* AWS SageMaker
* AWS S3
* AWS Athena
* AWS Data Wrangler
* Python (AWS SageMaker)
* SQL

## Methods
* Data Exploration
* Data Visualization
* Data Pre-processing
* Statistical Modeling
* Model Evaluation

## Getting Started
### Installation
1. In AWS, clone the repository by running the following commands:
```
git init
```
```
git clone https://github.com/edgar-rosales95/ads-508-team-project.git
```
2. Run the [Dependencies Notebook](https://github.com/edgar-rosales95/ads-508-team-project/blob/main/Dependencies.ipynb)
3. Run the [ADS_508_Final_Notebook](https://github.com/edgar-rosales95/ads-508-team-project/blob/main/ADS_508_Final_Notebook.ipynb)

## Authors
* [April Chia](https://github.com/aprilchia)
* [Christian Lee](https://github.com/mitosisgg)
* [Edgar Rosales](https://github.com/edgar-rosales95)
