import kagglehub

# Download the latest version.
kagglehub.dataset_download('bricevergnou/spotify-recommendation')

# Download a specific version.
kagglehub.dataset_download('bricevergnou/spotify-recommendation/versions/1')

# Download a single file.
kagglehub.dataset_download('bricevergnou/spotify-recommendation', path='data.csv')

# Download a dataset or file, even if previously downloaded to cache.
kagglehub.dataset_download('bricevergnou/spotify-recommendation', force_download=True)