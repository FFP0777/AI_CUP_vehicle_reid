import os
import pandas as pd


directory = r"C:\Users\LPCAS\Desktop\vehicle_reid\datasets\AI_CUP\bounding_box_test"
output_txt_file = r"C:\Users\LPCAS\Desktop\vehicle_reid\datasets\AI_CUP\bounding_box_test_labels.txt"
output_csv_file = r"C:\Users\LPCAS\Desktop\vehicle_reid\datasets\AI_CUP\bounding_box_test_labels.csv"


data = []
with open(output_txt_file, 'w') as file:
    for filename in os.listdir(directory):
        if filename.endswith(".bmp"):  
            
            file_id = int(filename.split('_')[0])
            file.write(f"{filename} {file_id}\n")
            data.append((filename, file_id))


print(f"Data collected: {data}")
print(f"Labels saved to {output_txt_file}")


df = pd.DataFrame(data, columns=['path', 'iD'])
df['path'] = 'AI_CUP/bounding_box_test/' + df['path']
print(df.head())  
df.to_csv(output_csv_file, index=False)

print(f"CSV file saved to {output_csv_file}")

