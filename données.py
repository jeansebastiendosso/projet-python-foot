import s3fs

# Preferred: create the shared AWS profile once, then reuse it here.
fs = s3fs.S3FileSystem(
    endpoint_url="https://minio.lab.sspcloud.fr",
    client_kwargs={"region_name": "us-east-1"},
)
df=pd.read_parquet("s3://lfaivre/diffusion/aus1league_obraggregates_20242025.csv", filesystem =fs) 
