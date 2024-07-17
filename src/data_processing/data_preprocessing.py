from data_processing.data_handler import DataHandler
from sklearn.model_selection import train_test_split
from time import perf_counter

data_path = 'H:/Datasets/Summarization/news/data.csv'

if __name__ == '__main__':
    data = DataHandler.load_data(data_path, nrows=500)

    data = data[['Content', 'Summary']]

    start = perf_counter()

    # data['Content'] = data['Content'].apply(DataHandler.preprocess_data, to_lower=True)
    # data['Summary']= data['Summary'].apply(DataHandler.preprocess_data, to_lower=True)

    data['Content'] = DataHandler.to_sequences(data['Content'].tolist())
    data['Summary'] = DataHandler.to_sequences(data['Summary'].tolist())

    end = perf_counter()

    print(f'Время выполнения нераспараллеленной программы: {end - start} c\n')

    start = perf_counter()

    # data['Content'] = DataHandler.parallel_preprocess_data(data['Content'], to_lower=True)
    # data['Summary']= DataHandler.parallel_preprocess_data(data['Summary'], to_lower=True)

    data['Content'] = DataHandler.parallel_to_sequences(data['Content'].tolist())
    data['Summary'] = DataHandler.parallel_to_sequences(data['Summary'].tolist())

    end = perf_counter()

    print(f'Время выполнения распараллеленной программы: {end - start} c')
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=50)

# X_train = DataHandler.to_sequences(train_data['Content'].tolist())
# y_train = DataHandler.to_sequences(train_data['Summary'].tolist())

# X_test = DataHandler.to_sequences(test_data['Content'].tolist())
# y_test = DataHandler.to_sequences(test_data['Summary'].tolist())

# # X_train = train_data['Content'].tolist()
# # y_train = train_data['Summary'].tolist()

# # X_test = test_data['Content'].tolist()
# # y_test = test_data['Summary'].tolist()

# train_data = {'X_train': X_train.tolist(), 'y_train': y_train.tolist()}
# test_data = {'X_test': X_test.tolist(), 'y_test': y_test.tolist()}

# DataHandler.data_to_csv(content=train_data, filename='train_data')
# DataHandler.data_to_csv(content=test_data, filename='test_data')

# DataHandler.tokenizer_to_json()