# Harmonizing Visual Text Comprehension and Generation


## Environment
```
pip install requirements.txt
```
prepare the weights following MM-Interleaved

## Inference
```
torchrun --nproc_per_node 1 --nnodes 1 --master_port 2333 inference.py  --config_file=TextHarmony/mm_interleaved/configs/release/example_inference.yaml
```

## Evaluation

### image comprehension
```
torchrun --nproc_per_node 1 --nnodes 1 --master_port 2333 evaluate.py --config_file=TextHarmony/mm_interleaved/configs/release/896-moe-eval.yaml
```

### image generation
```
torchrun --nproc_per_node 1 --nnodes 1 --master_port 2333 inference.py  --config_file=TextHarmony/mm_interleaved/configs/release/896-moe-inference.yaml

python TextHarmony/image_eval/eval_dgocr.py
```
