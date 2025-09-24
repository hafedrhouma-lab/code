### Load model artifacts from AWS S3
Load from QA
```bash
aws s3 cp s3://talabat-qa-ace-airflow-eu-west-2/ace/ace_artifacts/ranking/twotower_v23 ./ --recursive
```

### Truncate loaded `parquet` files
Run in the repo's root:
```bash
python ./scripts/cut_parquet_files.py --dryrun "**/streamlit_*.parquet" ./tests/fixtures/s3/ace/ace_artifacts/ranking 5
python ./scripts/cut_parquet_files.py --dryrun "**/tt_user_embeddings_recall*.parquet" ./tests/fixtures/s3/ace/ace_artifacts/ranking 5
```
