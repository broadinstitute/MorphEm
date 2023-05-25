import pandas as pd
import os
import json
import evaluation

import argparse
import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    """
    Example:
        python benchmark.py --root_dir "../../datasets/morphem_70k_2.0" --dest_dir "../results/convnext" \
                            --feature_dir "../../datasets/morphem_70k_2.0/features" \
                            --feature_file "pretrained_convnext_channel_replicate.npy" \
                            --use_gpu \
                            --classifier "knn" \
                            --umap
    """

    parser = argparse.ArgumentParser('MorphEm-benchmarking', add_help=False)

    parser.add_argument('--root_dir', default="../../datasets/morphem_70k_2.0", type=str,
                        help='Path to data directory')
    parser.add_argument('--dest_dir', default="../../results", type=str,
                        help='Path to save results')
    parser.add_argument('--feature_dir', default='../../datasets/morphem_70k_2.0/features', type=str,
                        help='Directory of features')
    parser.add_argument('--feature_file', default='pretrained_resnet18_features.npy', type=str,
                        help='Filename of features')

    # Training parameters
    parser.add_argument('--use_gpu', default=False, action='store_true', 
                        help='Use GPU to run classifier')
    parser.add_argument('--classifier', default='knn', type=str, choices=['knn', 'sgd'], 
                        help='Classifier to use')
    parser.add_argument('--umap', default=False, action='store_true', 
                        help='Create umap for features')
    parser.add_argument('--knn_metric', default='l2', type=str, choices=['l2', 'cosine'], 
                        help='Metric of KNN')
    return parser


def save_results(results, dest_dir, dataset, classifier):
    # Helper function
    # Save results for each dataset as a json dictionary at dest_dir
    full_reports_dict = {}
    full_reports_dict['target_encoding'] = results["encoded_target"]
    for task_ind, task in enumerate(results["tasks"]):
        full_reports_dict[task] = results["reports_dict"][task_ind]

    if not os.path.exists(dest_dir + '/'):
        os.makedirs(dest_dir + '/')

    dict_path = f'{dest_dir}/{dataset}_{classifier}_results.json'
    with open(dict_path, 'w') as f:
        json.dump(full_reports_dict, f)
    print("wrote results to ", dict_path)

    return


def main(args):
    # read all input parameters
    root_dir = args.root_dir
    dest_dir = args.dest_dir
    classifier = args.classifier
    feature_dir = args.feature_dir
    feature_file = args.feature_file
    use_gpu = args.use_gpu
    umap = args.umap
    knn_metric = args.knn_metric

    # encode dataset, task, and classifier
    task_dict = pd.DataFrame({'dataset': ['Allen', 'HPA', 'CP'],
                              'classifier': [classifier] * 3,
                              'leave_out': [None, 'Task_three', 'Task_four'],
                              'leaveout_label': [None, 'cell_type', 'Plate'],
                              'umap_label': ['Structure', 'cell_type', 'source'] 
                              })
    
    full_result_df = pd.DataFrame(columns=['dataset', 'task', 'classifier', 'accuracy', 'f1_score_macro'])

    # Iterate over each dataset
    for idx, row in task_dict.iterrows():
        dataset = row.dataset
        classifier = row.classifier
        leave_out = row.leave_out
        leaveout_label = row.leaveout_label
        umap_label = row.umap_label
        
        features_path = f'{feature_dir}/{dataset}/{feature_file}'
        df_path = f'{root_dir}/{dataset}/enriched_meta.csv'
        
        # Create umap and run classification
        if umap:
            evaluation.create_umap(dataset, features_path, df_path, dest_dir, ['Label', umap_label])
            
        results = evaluation.evaluate(features_path, 
                                      df_path, 
                                      leave_out, 
                                      leaveout_label, 
                                      classifier, 
                                      use_gpu=use_gpu, 
                                      knn_metric=knn_metric)

        # Print the full results
        print('Results:')
        for task_ind, task in enumerate(results["tasks"]):
            print(f'Results for {dataset} {task} with {classifier} :')
            print(results["reports_str"][task_ind])

        # Save results as dictionary
        save_results(results, dest_dir, dataset, classifier)

        # Save results as csv
        result_temp = pd.DataFrame({'dataset': [dataset] * len(results["tasks"]),
                                    'task': results["tasks"],
                                    'classifier': [classifier] * len(results["tasks"]),
                                    'accuracy': results["accuracies"],
                                    'f1_score_macro': results["f1scores_macro"]})

        full_result_df = pd.concat([full_result_df, result_temp]).reset_index(drop=True)
        
    if classifier == 'knn':
        out_path = f'{dest_dir}/{classifier}_{knn_metric}_full_results.csv'
    else:
        out_path = f'{dest_dir}/{classifier}_full_results.csv'
        
    full_result_df.to_csv(out_path, index=False)
    print("wrote full results to ", out_path)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
