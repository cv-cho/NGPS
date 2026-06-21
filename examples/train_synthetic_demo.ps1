python scripts/make_synthetic_demo_data.py --output-root data/synthetic --train-cases 4 --val-cases 2 --depth 24 --size 96
python scripts/train_ngps.py --data-root data/synthetic --config configs/synthetic_demo.yaml --output-dir runs/synthetic_ngps --device cpu
python scripts/infer_volume.py --checkpoint runs/synthetic_ngps/model_final.pth --input data/synthetic/val/noisy/val_case_000.npy --output outputs/synthetic_val_case_000_ngps.npy --device cpu
python scripts/evaluate_volume.py --pred outputs/synthetic_val_case_000_ngps.npy --clean data/synthetic/val/clean/val_case_000.npy --output-csv outputs/synthetic_val_case_000_metrics.csv
