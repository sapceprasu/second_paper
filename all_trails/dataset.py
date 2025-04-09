
from datasets import load_dataset
import pdb


# Load the dataset
dataset = load_dataset("Yusuf5/OpenCaselist")
pdb.set_trace()


for data in dataset[0:20]:
    print(data)

# # Extract unique topics
# unique_topics = set()
# for entry in dataset['train']:
#     topic = entry.get('topic')
#     if topic:
#         unique_topics.add(topic)

# # Convert the set to a list
# topics_list = list(unique_topics)

# # Display the number of unique topics
# print(f"Number of unique topics: {len(topics_list)}")
