#%%
import os

#%%

### -------------------------###
# Change this to choose which set to convert
which_convert = 'test' # 'train'
### -------------------------###

test_lab_set = ['ground', 'pix', 'piy', 'halfpix', 'halfpiy', 'halfpixy', 'halfpiyx']
train_lab_set = ['ground', 'excited']
default_pth = 'Data/tomo_train_240608/' # High/low fidelity data should be saparate in different experiment
# e.g., `tomo_train_HF_240608`; `tomo_train_LF_240608`
convert_set = {'train': [train_lab_set, 'train/'], 'test': [test_lab_set, 'test/']}


def process_txt_file(file_path):
    merged_data = []

    with open(file_path, 'r') as txt_file:
        lines = txt_file.readlines()
        for i in range(0, len(lines), 4):
            i_values = lines[i].strip().split(',')[1:]
            q_values = lines[i+2].strip().split(',')[1:]

            i_line = [float(x) if x else 0.0 for x in i_values]
            q_line = [float(x) if x else 0.0 for x in q_values]
            merged_data.append(i_line + q_line)

        return merged_data

if __name__ == '__main__':
    label_set, add_pth = convert_set[which_convert]
    for lbs in label_set:
        input_folder = default_pth + add_pth + lbs
        output_csv_path = default_pth + add_pth + which_convert + '_'+ lbs + '.csv'

        all_data = []
        txt_files = [file for file in os.listdir(input_folder) if file.endswith('.txt')]

        for txt_file in txt_files:
            file_path = os.path.join(input_folder, txt_file)
            processed_data = process_txt_file(file_path)
            all_data.extend(processed_data)

        # Writing the merged data to a CSV file
        with open(output_csv_path, 'w') as output_csv:
            for data in all_data:
                output_csv.write(','.join(map(str, data)) + '\n')
# %%
# This code block is invalid now

### -------------------------###
# Change this to choose which set to convert
which_convert = 'LF' # 'LF'
### -------------------------###

train_lab_set = ['ground', 'excited']
default_pth = 'Data/crosstalk_240625/'
convert_set = {'HF': [train_lab_set, 'test/HF/'], 'LF': [train_lab_set, 'test/LF/']}


def process_txt_file(file_path):
    merged_data = []

    with open(file_path, 'r') as txt_file:
        lines = txt_file.readlines()
        for i in range(0, len(lines), 4):
            i_values = lines[i].strip().split(',')[1:]
            q_values = lines[i+2].strip().split(',')[1:]

            i_line = [float(x) if x else 0.0 for x in i_values]
            q_line = [float(x) if x else 0.0 for x in q_values]
            merged_data.append(i_line + q_line)

        return merged_data

if __name__ == '__main__':
    label_set, add_pth = convert_set[which_convert]
    for lbs in label_set:
        input_folder = default_pth + add_pth + lbs
        output_csv_path = default_pth + add_pth + which_convert + '_'+ lbs + '.csv'

        all_data = []
        txt_files = [file for file in os.listdir(input_folder) if file.endswith('.txt')]

        for txt_file in txt_files:
            file_path = os.path.join(input_folder, txt_file)
            processed_data = process_txt_file(file_path)
            all_data.extend(processed_data)

        # Writing the merged data to a CSV file
        with open(output_csv_path, 'w') as output_csv:
            for data in all_data:
                output_csv.write(','.join(map(str, data)) + '\n')