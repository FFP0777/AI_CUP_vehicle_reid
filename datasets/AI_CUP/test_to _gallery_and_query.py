import pandas as pd


file_path = 'C:\Users\LPCAS\Desktop\vehicle_reid\datasets\AI_CUP\bounding_box_test_labels.txt'
test_data = pd.read_csv(file_path, sep=" ", header=None, names=["filename", "id"])



query_data = test_data.groupby('id').head(1).reset_index(drop=True)
gallery_data = test_data[~test_data.index.isin(query_data.index)].reset_index(drop=True)


query_data.to_csv('query.csv', index=False)
gallery_data.to_csv('gallery.csv', index=False)

print("Query set:", query_data.shape)
print("Gallery set:", gallery_data.shape)
