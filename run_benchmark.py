import pandas as pd
import os
import json
import evaluation

import argparse
import warnings
warnings.filterwarnings("ignore")

def get_args_parser():
    
    # command
    
    # python run_benchmark.py --root_dir "../datasets/morphem_70k_2.0" --dest_dir "../results" \
    #                         --feature_dir "../datasets/morphem_70k_2.0/features" \
    #                         --feature_file "pretrained_resnet18_features.npy" \
    #                         --classifier "knn" --umap True
    #                          

    
    parser = argparse.ArgumentParser('MorphEm-benchmarking', add_help=False)
    
    # Path parameters
        
    parser.add_argument('--root_dir', default="../datasets/morphem_70k_2.0", type=str, help='Path to data directory')
    parser.add_argument('--dest_dir', default="../results", type=str, help='Path to save results')
    parser.add_argument('--feature_dir', default='pretrained_resnet18_features.npy', type=str, help='Filename of features')
    parser.add_argument('--feature_file', default='pretrained_resnet18_features.npy', type=str, help='Filename of features')

    
    # Training parameters
    # parser.add_argument('--gpu', default=None, type=int, help='GPU to use')
    parser.add_argument('--classifier', default='knn', type=str, help='Classifier to use')
    parser.add_argument('--umap', default=False, type=bool, help='Output umap of features')
 
    return parser



def save_results(results, dest_dir, dataset, classifier):
    # Helper function
    # Save results for each dataset as a json dictionary at dest_dir
    full_reports_dict = {}
    full_reports_dict['target_encoding'] = results["encoded_target"]
    for task_ind, task in enumerate(results["tasks"]):
        full_reports_dict[task] = results["reports_dict"][task_ind]

    if not os.path.exists(dest_dir+ '/'):
        os.makedirs(dest_dir+ '/')

    dict_path = f'{dest_dir}/{dataset}_{classifier}_results.json'
    with open(dict_path, 'w') as f:
        json.dump(full_reports_dict, f)

    return
            
def main(args):
    
    # command
    
    # python run_benchmark.py --root_dir "./datasets/morphem_70k_2.0" --dest_dir "./results" \
    #                         --feature_dir "../datasets/morphem_70k_2.0/features" \
    #                         --feature_file "pretrained_resnet18_features.npy" \
    #                         --classifier "knn" --umap True
    
    
    # read all input parameters
    
    root_dir              = args.root_dir
    dest_dir              = args.dest_dir
    classifier            = args.classifier
    feature_dir           = args.feature_dir
    feature_file          = args.feature_file
    umap                  = args.umap
    # gpu                   = args.gpu
    
    # encode dataset, task, and classifier
    task_dict = pd.DataFrame({'dataset':['Allen', 'HPA', 'CP'], 
                              'classifier':[classifier for i in range(3)], \
                              'leave_out': [None, 'Task_three', 'Task_four'], \
                              'leaveout_label': [None, 'cell_type', 'Plate'], \
                              'umap_label': ['Structure', 'cell_type', 'source'] 
                             })
    print('Results:')
    full_result_df = pd.DataFrame(columns=['dataset', 'task', 'classifier', 'accuracy', 'f1_score_macro'])
    
    # Iterrate over each dataset
    for idx, row in task_dict.iterrows():
        dataset        = row.dataset
        classifier     = row.classifier
        leave_out      = row.leave_out
        leaveout_label = row.leaveout_label
        umap_label     = row.umap_label
        
        features_path  = f'{feature_dir}/{dataset}/{feature_file}'
        df_path        = f'{root_dir}/{dataset}/enriched_meta.csv'
        
        # Create umap and run classification
        if umap:
            evaluation.create_umap(dataset, features_path, df_path, dest_dir, ['Label', umap_label])
        results = evaluation.evaluate(features_path, df_path, leave_out, leaveout_label, classifier)

        # Print the full results
        for task_ind, task in enumerate(results["tasks"]):
            print(f'Results for {dataset} {task} with {classifier} :')
            print(results["reports_str"][task_ind])
        
        # Save results as dictionary
        save_results(results, dest_dir, dataset, classifier)
        
        # Save results as csv
        result_temp = pd.DataFrame({'dataset': [dataset for i in range(len(results["tasks"]))],\
                        'task': results["tasks"],'classifier': [classifier for i in range(len(results["tasks"]))],\
                        'accuracy': results["accuracies"],'f1_score_macro': results["f1scores_macro"]})
        full_result_df = pd.concat([full_result_df, result_temp]).reset_index(drop=True)
            
    full_result_df.to_csv(f'{dest_dir}/{classifier}_full_results.csv', index=False) 
    
    
if __name__ == '__main__':
    
    args = get_args_parser()
    args = args.parse_args()

    main(args)