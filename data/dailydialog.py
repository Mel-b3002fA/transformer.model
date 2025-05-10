import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = ""

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "thedevastator/dailydialog-multi-turn-dialog-with-intention-and",
  file_path,
)

print("First 5 records:", df.head())