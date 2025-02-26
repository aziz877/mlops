DATASET_TRAIN=churn-bigml-80.csv
DATASET_TEST=churn-bigml-20.csv
MODEL=modelRF.joblib

all: prepare train evaluate
notify_success:
	@powershell.exe -ExecutionPolicy Bypass -File ~/notify.ps1
prepare:
	python notify.py "Preparation Completed" "python main.py prepare $(DATASET_TRAIN) $(DATASET_TEST)"
	make notify_success

train:
	python notify.py "Training Completed" "python main.py train $(DATASET_TRAIN) $(DATASET_TEST) --model_filename $(MODEL)"
	make notify_success

evaluate:
	python notify.py "Evaluation Completed" "python main.py evaluate $(DATASET_TRAIN) $(DATASET_TEST) --model_filename $(MODEL)"
	make notify_success

full:
	python notify.py "Pipeline Completed" "python main.py full $(DATASET_TRAIN) $(DATASET_TEST) --model_filename $(MODEL)"
	make notify_success


