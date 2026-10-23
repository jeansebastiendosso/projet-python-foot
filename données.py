import s3fs
import pandas as pd

# Preferred: create the shared AWS profile once, then reuse it here.
fs = s3fs.S3FileSystem(
    endpoint_url="https://minio.lab.sspcloud.fr",
    client_kwargs={"region_name": "us-east-1"},
)


obr=pd.read_parquet("s3://lfaivre/diffusion/aus1league_obraggregates_20242025.parquet", filesystem=fs) 
passing=pd.read_parquet("s3://lfaivre/diffusion/aus1league_passingaggregates_20242025.parquet", filesystem=fs) 
physical=pd.read_parquet("s3://lfaivre/diffusion/aus1league_physicalaggregates_20242025.parquet", filesystem=fs) 

