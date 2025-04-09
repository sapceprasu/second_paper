from datasets import load_dataset

ds = load_dataset("Hellisotherpeople/DebateSum")


print(ds[0:5])