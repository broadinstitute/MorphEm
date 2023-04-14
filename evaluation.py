import pandas as pd
import numpy as np

import utils

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, classification_report


def evaluate(features_path, df_path, leave_out, leaveout_label, model_choice, use_gpu: bool):
    # Load features and metadata
    print('Loading features...')

    features = np.load(features_path)
    df = pd.read_csv(df_path)

    # Count number of tasks
    tasks = list(df['train_test_split'].unique())
    tasks.remove('Train')
    if leave_out is not None:
        leaveout_ind = tasks.index(leave_out)

    # Get index for training and each testing set
    train_indices = np.where(df['train_test_split'] == 'Train')[0]
    all_test_indices = [np.where(df[task])[0] for task in tasks]

    # Convert categorical labels to integers    
    target_value = list(df['Label'].unique())

    encoded_target = {target_value[i]: i for i in range(len(target_value))}
    df['encoded_label'] = df.Label.map(encoded_target)

    # Split data into training and testing for regular classification
    train_X = features[train_indices]
    test_Xs = [features[test_indices] for test_indices in all_test_indices]

    task_Ys = [df['encoded_label'].values] * len(tasks)
    train_Ys = [task_Ys[task_ind][train_indices] for task_ind in range(len(tasks))]
    test_Ys = [task_Ys[task_ind][test_indices] for task_ind, test_indices in enumerate(all_test_indices)]

    # Data splitting for leave one out task
    if leave_out is not None:
        df_takeout = df[df[leave_out]]
        groups = list(df_takeout[leaveout_label].unique())

        all_group_indices = [df_takeout[df_takeout[leaveout_label] == group].index.values for group in groups]
        all_other_indices = [df_takeout[df_takeout[leaveout_label] != group].index.values for group in groups]

        takeout_X = [features[group_indices] for group_indices in all_group_indices]
        rest_X = [features[np.concatenate((train_indices, other_indices))] for other_indices in all_other_indices]

        takeout_Y = [task_Ys[leaveout_ind][group_indices] for group_indices in all_group_indices]
        rest_Y = [task_Ys[leaveout_ind][np.concatenate((train_indices, other_indices))] for other_indices in
                  all_other_indices]

    print('Train classifiers...')
    accuracies = []
    f1scores_macro = []
    reports_str = []
    reports_dict = []

    for task_ind, task in enumerate(tasks):
        if task != leave_out:  # standard classification
            if model_choice == 'knn':
                model = utils.FaissKNeighbors(k=1, use_gpu=use_gpu)
            elif model_choice == 'sgd':
                model = SGDClassifier(alpha=0.001, max_iter=100)
            else:
                raise NotImplementedError(f'{model_choice} is not implemented. Try sgd or knn.')

            model.fit(train_X, train_Ys[task_ind])
            predictions = model.predict(test_Xs[task_ind])
            ground_truth = test_Ys[task_ind]

        else:  # leave-one-out
            predictions = []
            ground_truth = []
            for group_ind, group in enumerate(groups):
                model = utils.FaissKNeighbors(k=1, use_gpu=use_gpu)
                ## Use the rest_X and rest_Y to train the model, then predict on the takeout_X
                model.fit(rest_X[group_ind], rest_Y[group_ind])
                group_predictions = model.predict(takeout_X[group_ind])
                group_ground_truth = takeout_Y[group_ind]

                predictions.append(group_predictions)
                ground_truth.append(group_ground_truth)

            predictions = np.concatenate(predictions)
            ground_truth = np.concatenate(ground_truth)
        # Compute evaluation metrics
        int_labels = np.unique(ground_truth)
        str_labels = [target_value[idx] for idx in int_labels]

        accuracy = np.mean(predictions == ground_truth)
        report_str = classification_report(ground_truth, predictions, labels=int_labels, target_names=str_labels)
        report_dict = classification_report(ground_truth, predictions, labels=int_labels,
                                            target_names=str_labels, output_dict=True)
        f1score_macro = f1_score(ground_truth, predictions, labels=np.unique(ground_truth), average='macro')

        accuracies.append(accuracy)
        f1scores_macro.append(f1score_macro)
        reports_str.append(report_str)
        reports_dict.append(report_dict)

    return {
        "tasks": tasks,
        "accuracies": accuracies,
        "f1scores_macro": f1scores_macro,
        "reports_str": reports_str,
        "reports_dict": reports_dict,
        "encoded_target": encoded_target
    }
