
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = ""


df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "atharvjairath/empathetic-dialogues-facebook-ai",
  file_path,
 
)

print("First 5 records:", df.head())