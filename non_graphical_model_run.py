import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
from typing import Tuple
import xgboost as xgb
from sklearn.metrics import roc_auc_score,  average_precision_score
import warnings
import tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
## hyper parameters
save_path = "results.csv"
## helper functions
def make_data(cut_off)->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Make training testing and validation data for the model.
    
        Args:
            cut_off: int, the cut off for the number of drugs to include in the model
        Returns:
            train_df: pd.DataFrame, the training data
            val_df: pd.DataFrame, the validation data
            test_df: pd.DataFrame, the testing data
            n_unique_patients: int, the number of unique patients in the data
    """
    ## Load and merge the data
    patient_to_cancer_type_path = "data/indices/diagnosed_with_edges.tsv"
    patient_to_drug_path = "data/indices/treated_with_edges.tsv"
    patient_to_cancer_type = pd.read_csv(patient_to_cancer_type_path, sep="\t")
    patient_to_drug = pd.read_csv(patient_to_drug_path, sep="\t")
    merged_df  = patient_to_cancer_type.merge(patient_to_drug, on="patient_index")
    ## drop rows that have a drug index that dont occur at cutoff.
    drug_counts = merged_df.value_counts('drug_index')
    most_common_drugs = drug_counts[drug_counts>cut_off]
    merged_df = merged_df[merged_df['drug_index'].isin(most_common_drugs.index)]
    ## give a unique id to each patient
    n_unique_patients = len(merged_df['patient_index'].unique())
    # Split the data into train and test on unique patient 
    train, test = train_test_split(merged_df['patient_index'].unique(), test_size=0.1) 
    train, val = train_test_split(train, test_size=0.1/0.9) #0.9 X x = 0.1, x = 0.1/0.9

    ## make a unique id for each cancer type and drug
    
    # merged_df['cancer_type_idx'] = pd.factorize(merged_df['cancer_type_index'])[0]
    merged_df['drug_idx'] = pd.factorize(merged_df['drug_index'])[0]
    ## want to save the mapping of drug_idx to drug_index in case we need it later.
    drug_idx_to_drug = merged_df[['drug_idx', 'drug_index']].drop_duplicates()
    drug_idx_to_drug.to_csv("drug_idx_to_drug_index_mapping.csv", index=False)
    # merged_df.drop(columns=['cancer_type_index', 'drug_index'], inplace=True)
    merged_df.drop(columns=["drug_index"], inplace=True)
    ## one hot encode cancer type 
    cancer_type_one_hot = pd.get_dummies(merged_df['cancer_type_index'], prefix='cancer_type', dtype=int)
    # import seaborn as sb
    ## heat map of cancer_type_index and drug_idx

    merged_df = pd.concat([merged_df, cancer_type_one_hot], axis=1)
    
    merged_df.drop(columns=['cancer_type_index'], inplace=True)

    ## split the df based on the train, val, test
    train_df = merged_df[merged_df['patient_index'].isin(train)]
    val_df = merged_df[merged_df['patient_index'].isin(val)]
    test_df = merged_df[merged_df['patient_index'].isin(test)]

    ## drop the patient_idx
    train_df.drop(columns=['patient_index'], inplace=True)
    val_df.drop(columns=['patient_index'], inplace=True)
    test_df.drop(columns=['patient_index'], inplace=True)
    ## 

    num_drugs = len(np.unique(train_df['drug_idx']))
    return train_df, val_df, test_df, n_unique_patients

def get_values_and_labels(data:pd.DataFrame)->Tuple[np.ndarray, np.ndarray]:
    """Get X and y from the data frame, assumes that the labels are in the first column 

    Args:
        data: pd.DataFrame, the data frame to extract the values and labels from
    Returns:
        X: np.ndarray, the features
        y: np.ndarray, the labels
    """
    # import ipdb; ipdb.set_trace()
    X = data.values[:, 1:]
    y = data.values[:, 0]
    return X, y
def evaluate_model(model ,X_train, y_train, X_val, y_val, X_test, y_test, cut_off, n_unique_patients:int)->None:
    """Evaluate the model and write the results to a file

    Args:
        model: Sklearn model to evaluate
        X_train: np.ndarray, the training features
        y_train: np.ndarray, the training labels
        X_val: np.ndarray, the validation features
        y_val: np.ndarray, the validation labels
        X_test: np.ndarray, the testing features
        y_test: np.ndarray, the testing labels
        cut_off: int, the cut off for the number of drugs to include in the model
        n_unique_patients: int, the number of unique patients in the data
    Returns:
        None
    """
    n_unique_drugs = len(np.unique(y_train))
    train_preds = model.predict_proba(X_train)
    val_preds = model.predict_proba(X_val)
    test_preds = model.predict_proba(X_test)
    test_preds = model.predict_proba(X_test)
    test_preds = model.predict_proba(X_test)
    ## count zeros in diference 
    # np.non_zero(y_test - test_preds)
    # len(y_test)-np.count_nonzero(y_test - test_preds)
    #test_value_count = test_preds.value_counts()
    ## get vlaue counts for test preds 
    # np.unique(test_preds, return_counts=True)

    import ipdb; ipdb.set_trace()
    ## diference y test and test_preds
    train_auc_ovr = roc_auc_score(y_train, train_preds, multi_class='ovr')
    val_auc_ovr = roc_auc_score(y_val, val_preds,  multi_class='ovr')
    test_auc_ovr = roc_auc_score(y_test, test_preds,  multi_class='ovr')
    train_auc_ovo = roc_auc_score(y_train, train_preds, multi_class='ovo')
    val_auc_ovo = roc_auc_score(y_val, val_preds,  multi_class='ovo')
    test_auc_ovo = roc_auc_score(y_test, test_preds,  multi_class='ovo')
    train_ap = average_precision_score(y_train, train_preds)
    val_ap = average_precision_score(y_val, val_preds)
    test_ap = average_precision_score(y_test, test_preds)
    with open(save_path, "a") as f:
        f.write(f"{cut_off},{n_unique_drugs},{n_unique_patients},{train_auc_ovr},{train_auc_ovo},{train_ap},{val_auc_ovr},{val_auc_ovo},{val_ap},{test_auc_ovr},{test_auc_ovo},{test_ap}\n")
    f.close()
def make_file_header()->None:
    """write header to output file.

    Args:
        None
    Returns:
        None
    """
    with open(save_path, "w") as f:
        f.write("cut_off,n_unique_drugs,n_unique_patients, train_auc_ovr,train_auc_ovo,train_ap,val_auc_ovr,val_auc_ovo,val_ap,test_auc_ovr,test_auc_ovo,test_ap\n")
    f.close()
def handel_failure(cut_off:int)->None:
    """write zeros to the output file to indicate that the model failed.

        Args:
            cut_off: int, the cut off for the number of drugs to include in the model
        Return: 
            None"""
    with open(save_path, "a") as f:
        f.write(f"{cut_off},0,0,0,0,0,0,0,0,0,0,0\n")
    f.close()
def sample_patient_number_plots()->None:
    """Make plots of the number of unique drugs and patients at different cut offs.
    
    Args:
        None
    Returns:
        None
    """
    ## Load and merge the data
    patient_to_cancer_type_path = "data/hetero_graph_patient_drug_cancer_onco-gene/patient_cancer_type.tsv"
    patient_to_drug_path = "data/hetero_graph_patient_drug_cancer_onco-gene/patient_drug.tsv"
    patient_to_cancer_type = pd.read_csv(patient_to_cancer_type_path, sep="\t")
    patient_to_drug = pd.read_csv(patient_to_drug_path, sep="\t")
    merged_df  = patient_to_cancer_type.merge(patient_to_drug, on="patient")
    drug_counts = merged_df.value_counts('drug')
    num_drugs_at_cutoff = []
    num_patients_at_cutoff = []
    ## drop UNKNOWN
    drug_counts = drug_counts[drug_counts.index != "UNKNOWN"]
    total_drugs = len(drug_counts)
    total_patients = len(merged_df['patient'].unique())
    total_known_patients = len(merged_df[merged_df['drug'] != "UNKNOWN"]['patient'].unique())
    for cut_off in range(0, 50, 1):
        most_common_drugs = drug_counts[drug_counts>cut_off]
        num_drugs_at_cutoff.append(len(most_common_drugs))
        num_patients_at_cutoff.append(len(merged_df[merged_df['drug'].isin(most_common_drugs.index)]['patient'].unique()))
    fig, ax = plt.subplots(2,1)
    ax[0].plot(range(0, 50, 1), num_drugs_at_cutoff)
    ax[0].set_xlabel("Cut off")
    ax[0].set_ylabel("Number of unique drugs")
    ax[0].set_title("Number of unique drugs at cut-off" )
    ax[0].axvline(x=1, color='r', label="x=1", linestyle='--')
    ax[0].axhline(y=num_drugs_at_cutoff[1], color='r', label="y=1", linestyle='--')
    ax[0].set_xticks([0, 1, 50])
    ax[0].set_yticks([total_drugs, num_drugs_at_cutoff[1], num_drugs_at_cutoff[-1]])
    ax[1].plot(range(0, 50, 1), num_patients_at_cutoff)
    ax[1].set_xlabel("Cut off")
    ax[1].set_ylabel("Number of unique patients")
    ax[1].set_title("Number of unique patients at cut-off ({0} total unique patients)".format(total_patients) )
    ax[1].axvline(x=1, color='r', label="x=1", linestyle='--')
    ax[1].axhline(y=num_patients_at_cutoff[1], color='r', label="y=1", linestyle='--')
    ax[1].set_xticks([0, 1, 50])    
    ax[1].set_yticks([total_known_patients, num_patients_at_cutoff[1], num_patients_at_cutoff[-1]])
    plt.tight_layout()
    print(total_known_patients/total_patients)
    plt.savefig("./figs/unique_drugs_and_patients_vs_cut_off.png")



def main():
    make_file_header()
    for cutoff in tqdm.tqdm(range(20, 25, 1)):
        # n_classes
        train_df, val_df, test_df, n_unique_patients = make_data(cutoff)
        X_train, y_train = get_values_and_labels(train_df)
        # import ipdb; ipdb.set_trace()
        np.sum(X_train, axis=1)
        X_val, y_val = get_values_and_labels(val_df)
        X_test, y_test = get_values_and_labels(test_df)
        n_classes = len(np.unique(y_train))
        xgb_model = xgb.XGBClassifier(objective='multi:softprob')
        
        # import ipdb; ipdb.set_trace()
        try:
            xgb_model.fit(X_train, y_train) ## check 
            evaluate_model(xgb_model,X_train, y_train, X_val, y_val, X_test, y_test, cutoff, n_unique_patients)
        except:
            handel_failure(cutoff)
            continue
    # sample_patient_number_plots()
if __name__ == "__main__":
    main()