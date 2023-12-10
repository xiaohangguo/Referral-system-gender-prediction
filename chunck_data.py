import pandas as pd
import torch
from transformers import BertTokenizer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# 定义一个函数来处理每个数据块
def preprocess_data(row_dict, tokenizer, max_len):
    # 将字典行转换回DataFrame行并进行处理
    text = ' '.join(str(value) for value in row_dict.values())
    label = row_dict['gender']  # 假设'gender'是标签列

    # 使用BERT tokenizer
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    return encoding['input_ids'].flatten(), encoding['attention_mask'].flatten(), torch.tensor(label, dtype=torch.long)

# 初始化tokenizer
tokenizer = BertTokenizer.from_pretrained('/public/home/lvshuhang/model_space/workspace/bert-base-uncased')
max_len = 512

chunk_size = 10000  # 您可以根据需要调整块大小
file_path = 'click_ad_test.csv'
total_rows = sum(1 for row in open(file_path, 'r')) - 1
def preprocess_row(row_dict):
    return preprocess_data(row_dict, tokenizer, max_len)
pbar = tqdm(total=total_rows)

# 分块读取数据
chunks = pd.read_csv(file_path, chunksize=chunk_size)
chunk_index = 0

for chunk in chunks:
    with ProcessPoolExecutor() as executor:
        chunk_dicts = chunk.to_dict(orient='records')
        results = list(executor.map(preprocess_row, chunk_dicts))
        
        # 准备保存数据
        input_ids, attention_masks, labels = zip(*results)
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)
        
        # 将每个块的数据保存到单独的文件
        torch.save((input_ids, attention_masks, labels), f'test_data/preprocessed_data_chunk_{chunk_index}.pt')
        chunk_index += 1

    pbar.update(len(chunk))

pbar.close()
