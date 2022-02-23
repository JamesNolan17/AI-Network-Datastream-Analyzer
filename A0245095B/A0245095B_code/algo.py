from __future__ import division
import datetime
import argparse
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.svm import SVC


parser = argparse.ArgumentParser()
parser.add_argument("file1", type=str)
parser.add_argument("file2", type=str)


file1 = parser.parse_args().file1
file2 = parser.parse_args().file2

# classifier = SVC(kernel="rbf", gamma='auto', random_state=0, decision_function_shape='ovo')
classifier = KNeighborsClassifier(n_neighbors= 10, weights="distance")
PROFILE_TRAIN = [1, 2, 3, 4, 5, 6, 7, 8]
PROFILE_TEST = [file1, file2]
FEATURE_TRAIN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
# FEATURE_TRAIN = [3, 4, 6, 7, 8, 11, 12, 13, 24, 25, 26]
# SVC 0.3142857142857143
# SVC 0.45714285714285713 KNN3 0.8285714285714286 KNN 6 0.8

DATASET_HEADER = ['pkt_num_in', #0
                  'pkt_num_out', #1
                  'pkt_num', #2
                  'ratio_pkt_num_in', #3
                  'ratio_pkt_num_out', #4
                  'pkt_length_min_in', #5 x
                  'pkt_length_quartile_in', #6
                  'pkt_length_mean_in', #7
                  'pkt_length_third_quartile_in', #8
                  'pkt_length_max_in', #9
                  'pkt_length_min_out', #10 x
                  'pkt_length_quartile_out', #11
                  'pkt_length_mean_out', #12
                  'pkt_length_third_quartile_out', #13
                  'pkt_length_max_out', #14
                  'pkt_length_min', #15 x
                  'pkt_length_quartile', #16
                  'pkt_length_mean', #17
                  'pkt_length_third_quartile', #18
                  'pkt_length_max', #19
                  'pkt_length', #20
                  'ratio_pkt_length_in', #21
                  'ratio_pkt_length_out', #22
                  'session_duration', #23
                  'time_per_pkt', #24
                  'ratio_info_pkt_num_out', #25
                  'ratio_info_pkt_num_in', #26
                  'ratio_info_pkt_num_in_out', #27
                  'url_id']
dataset_train = pd.DataFrame(columns=DATASET_HEADER)

def feature_abstraction_and_data_processing():
    global dataset_train
    for profile_index in PROFILE_TRAIN:
        for url_index in range(1, 36):
            folder_name = f"profile{profile_index}"
            dataset_gen_per_url(folder_name, url_index, 0)


def dataset_gen_per_url(folder_name, url_index, flag):
    global dataset_train
    dataset = pd.DataFrame(columns=DATASET_HEADER)
    if flag == 0:
        location = f"../traces/{folder_name}/{url_index}"
    else:
        location = f"{folder_name}/{url_index}-anon"
    raw_file = None
    try:
        raw_file = pd.read_table(location, delimiter=" ", names=["timestamp", "length", "direction"])
    except pd.errors.EmptyDataError:
        # print(f"{folder_name}/{url_index} is not a table / empty")
        pass
    except FileNotFoundError:
        # print(f"{folder_name}/{url_index} not found")
        pass
    if raw_file is not None:
        # Calculate the features
        pkt_in = []
        pkt_out = []
        info_pkt_in = []
        info_pkt_out = []
        time = []

        for index, row in raw_file.iterrows():
            time_item = row["timestamp"]
            h, m, _s = time_item.split(":")
            sec, ms = _s.split(".")
            datetime_object = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(sec), microseconds=int(ms))

            # Stat in
            if row["direction"] == "in":
                pkt_in.append(row["length"])
                if row["length"] == 0:
                    info_pkt_in.append(row["length"])
            # Stat out
            elif row["direction"] == "out":
                pkt_out.append(row["length"])
                if row["length"] == 0:
                    info_pkt_out.append(row["length"])
            else:
                continue
            time.append(datetime_object)
        # Sort the packets
        pkt_in.sort()
        pkt_out.sort()
        pkt = pkt_in + pkt_out
        pkt.sort()
        time.sort()
        if len(pkt_in) == 0 or len(pkt_out) == 0 or len(time) == 0:
            # print(f"{folder_name}/{url_index} got a 0 data field: {len(pkt_in)}{len(pkt_out)}{len(time)}")
            pass
        # Calculate stat
        # Feature Lists #
        # Overall statistic
            # | - 1.pkt number statistic

        pkt_num_in = len(pkt_in)
        pkt_num_out = len(pkt_out)
        pkt_num = len(pkt)
        if pkt_num > 0:
            ratio_pkt_num_in = pkt_num_in / pkt_num
            ratio_pkt_num_out = pkt_num_out / pkt_num
        else:
            ratio_pkt_num_in = 0
            ratio_pkt_num_out = 0
            # | - 2. pkt length statistic
        # In:
        if len(pkt_in) > 0:
            pkt_length_min_in = pkt_in[0]
            pkt_length_quartile_in = np.percentile(pkt_in, 25)
            pkt_length_mean_in = np.mean(pkt_in)
            pkt_length_third_quartile_in = np.percentile(pkt_in, 75)
            pkt_length_max_in = pkt_in[-1]
        else:
            pkt_length_min_in = 0
            pkt_length_quartile_in = 0
            pkt_length_mean_in = 0
            pkt_length_third_quartile_in = 0
            pkt_length_max_in = 0

        # Out:
        if len(pkt_out) > 0:
            pkt_length_min_out = pkt_out[0]
            pkt_length_quartile_out = np.percentile(pkt_out, 25)
            pkt_length_mean_out = np.mean(pkt_out)
            pkt_length_third_quartile_out = np.percentile(pkt_out, 75)
            pkt_length_max_out = pkt_out[-1]
        else:
            pkt_length_min_out = 0
            pkt_length_quartile_out = 0
            pkt_length_mean_out = 0
            pkt_length_third_quartile_out = 0
            pkt_length_max_out = 0

        # Total:
        if len(pkt) > 0:
            pkt_length_min = pkt[0]
            pkt_length_quartile = np.percentile(pkt, 25)
            pkt_length_mean = np.mean(pkt)
            pkt_length_third_quartile = np.percentile(pkt, 75)
            pkt_length_max = pkt[-1]
        else:
            pkt_length_min = 0
            pkt_length_quartile = 0
            pkt_length_mean = 0
            pkt_length_third_quartile = 0
            pkt_length_max = 0

        pkt_length = np.sum(pkt)
        if np.sum(pkt) > 0:
            ratio_pkt_length_in = np.sum(pkt_in) / np.sum(pkt)
            ratio_pkt_length_out = np.sum(pkt_out) / np.sum(pkt)
        else:
            ratio_pkt_length_in = 0
            ratio_pkt_length_out = 0

            # | - 3. time statistic
        if len(time) > 0:
            session_duration = time[-1] - time[0]
            session_duration = session_duration.total_seconds()
            time_per_pkt = session_duration / pkt_num
        else:
            session_duration = 0
            time_per_pkt = 0
            # print(session_duration)

        # Special statistic:
        if len(pkt_out) > 0:
            ratio_info_pkt_num_out = len(info_pkt_out) / len(pkt_out)  # Out packet with 0 bytes of payload
        else:
            ratio_info_pkt_num_out = 0
        if len(pkt_in) > 0:
            ratio_info_pkt_num_in = len(info_pkt_in) / len(pkt_in)  # In packet with 0 bytes of payload
        else:
            ratio_info_pkt_num_in = 0
        if len(info_pkt_out) > 0:
            ratio_info_pkt_num_in_out = len(info_pkt_in) / len(info_pkt_out)
        else:
            ratio_info_pkt_num_in_out = 999

        summary = {
            'pkt_num_in': pkt_num_in,
            'pkt_num_out': pkt_num_out,
            'pkt_num': pkt_num,
            'ratio_pkt_num_in': ratio_pkt_num_in,
            'ratio_pkt_num_out': ratio_pkt_num_out,
            'pkt_length_min_in': pkt_length_min_in,
            'pkt_length_quartile_in': pkt_length_quartile_in,
            'pkt_length_mean_in': pkt_length_mean_in,
            'pkt_length_third_quartile_in': pkt_length_third_quartile_in,
            'pkt_length_max_in': pkt_length_max_in,
            'pkt_length_min_out': pkt_length_min_out,
            'pkt_length_quartile_out': pkt_length_quartile_out,
            'pkt_length_mean_out': pkt_length_mean_out,
            'pkt_length_third_quartile_out': pkt_length_third_quartile_out,
            'pkt_length_max_out': pkt_length_max_out,
            'pkt_length_min': pkt_length_min,
            'pkt_length_quartile': pkt_length_quartile,
            'pkt_length_mean': pkt_length_mean,
            'pkt_length_third_quartile': pkt_length_third_quartile,
            'pkt_length_max': pkt_length_max,
            'pkt_length': pkt_length,
            'ratio_pkt_length_in': ratio_pkt_length_in,
            'ratio_pkt_length_out': ratio_pkt_length_out,
            'session_duration': session_duration,
            'time_per_pkt': time_per_pkt,
            'ratio_info_pkt_num_out': ratio_info_pkt_num_out,
            'ratio_info_pkt_num_in': ratio_info_pkt_num_in,
            'ratio_info_pkt_num_in_out': ratio_info_pkt_num_in_out,
            'url_id': url_index
        }

        if flag == 0:  # That means dataset_train
            dataset_train = dataset_train.append(summary, ignore_index=True)
        else:
            dataset = dataset.append(summary, ignore_index=True)
    return dataset


def ML_train():
    global dataset_train
    global classifier
    # print(dataset_train)
    x_train = dataset_train.iloc[:, FEATURE_TRAIN].values
    y_train = dataset_train.iloc[:, -1].values
    y_train = y_train.astype('int')

    # Feature Scaling
    x_train = preprocessing.normalize(x_train)

    # Fit classifier
    classifier.fit(x_train, y_train)


def predict(target_dataset):
    x_test = target_dataset.iloc[:, FEATURE_TRAIN].values
    x_test = preprocessing.normalize(x_test)
    y_pred = classifier.predict(x_test)
    prob = classifier.predict_proba(x_test)
    return y_pred[0], prob[0]


def test(filename):
    correct = 0
    correct_final = 0
    prob_matrix = []
    result = []
    for unknown_url_index in range(1, 36):
        dataset_train_url = dataset_gen_per_url(filename, unknown_url_index, 1)
        _result, prob = predict(dataset_train_url)
        prob_matrix.append(prob)
        if _result == unknown_url_index:
            correct += 1
    # Transpose
    prob_matrix = list(map(list, zip(*prob_matrix)))
    URL = 1
    for i in prob_matrix:
        printable_result = 1 + i.index(np.max(i))
        result.append(printable_result)
        if URL == printable_result:
            correct_final += 1
        URL += 1

    # print(f"Accuracy:{correct/35}")
    # print(f"Final Accuracy:{correct_final / 35}")
    return result

def main():
    # print(str(PROFILE_TEST))
    feature_abstraction_and_data_processing()
    ML_train()
    result1 = test(PROFILE_TEST[0])
    result2 = test(PROFILE_TEST[1])
    # print(result1)
    # print(result2)
    result_file = open("result.txt", "w")
    for i in range(0,35):
        result_file.write(f"{result1[i]} {result2[i]}\n")

if __name__ == '__main__':
    main()
