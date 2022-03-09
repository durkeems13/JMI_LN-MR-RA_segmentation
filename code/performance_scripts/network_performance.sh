#!/bin/sh
export PKLS_READ=../analysis/LN_testset/processing/combined_inference_pkls
export GT_PKLS=../data/manual_pkls/Human_FFPE_ss_5fold_testset
export CSV_NAME=LuN_folds/LuN_F2

for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    python3 get_prediction_features_joint.py --pkls_read "$PKLS_READ" --gt_pkls "$GT_PKLS" --csv_name "$CSV_NAME" --threshold $i
    python3 seg_metrics_paraffin_ss_joint.py --csv_dir "$CSV_NAME" --threshold $i
done
