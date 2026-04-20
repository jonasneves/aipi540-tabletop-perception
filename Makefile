.PHONY: dataset naive classical dl eval sync serve deploy all

PY := python3

dataset:
	$(PY) scripts/make_dataset.py --out data/processed

naive:
	$(PY) scripts/naive.py --train data/processed/train --test data/processed/test --out results/naive.json

classical:
	$(PY) scripts/classical.py --train data/processed/train --test data/processed/test --out results/classical.json --save-model models/classical.pkl

dl:
	$(PY) scripts/train_dl.py --train data/processed/train --test data/processed/test --out-dir models --results-dir results --epochs 12 --skip-detector

eval: naive classical dl

sync:
	mkdir -p public/models public/results
	cp models/classifier.onnx public/models/ 2>/dev/null || true
	cp results/*.json public/results/ 2>/dev/null || true

serve: sync
	cd public && $(PY) -m http.server 8088

deploy: sync
	git add public models results
	git commit -m "update models + app artifacts" || echo "nothing to commit"
	git push

all: dataset eval sync
