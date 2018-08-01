all: data main

data:
	bash data/download.sh

main:
	python main.py --gpu=0 --resume=0 --checkpoints=./checkpoints/ --config=./experiments/inpaint.yaml

test:
	python test.py --gpu=0 --resume=0 --checkpoints=./checkpoints/ --config=./experiments/inpaint.yaml
	