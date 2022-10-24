.PHONY: install
install:
	pip install -r requirements.txt
	pip install -r yolov5/requirements.txt
	pip install -U rich



.PHONY: prepare_data
prepare_data:
	python src/data/prepare.py


.PHONY: train
train:
	python yolov5/train.py --img 480 --batch 64 --epochs 500 --data configs/data.yml --weights weights/yolov5s.pt --cfg configs/yolov5s.yml --hyp configs/hyps.yml


.PHONY: convert
convert:
    python yolov5/export.py --weights weights/yolov5s_barcode_detector.pt --img 480 --dynamic --nms --include torchscript onnx


.PHONY: lint
lint:
	PYTHONPATH=. flake8 src/


.PHONY: run_unit_tests
run_unit_tests:
	PYTHONPATH=. pytest tests/unit/


.PHONY: run_integration_tests
run_integration_tests:
	PYTHONPATH=. pytest tests/integration/


.PHONY: run_tests
run_tests:
	make run_unit_tests
	make run_integration_tests


.PHONY: generate_coverage_report
generate_coverage_report:
	PYTHONPATH=. pytest --cov=src --cov-report html  tests/

