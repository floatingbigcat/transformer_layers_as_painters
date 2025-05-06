# Transformer Layers as Painters 🧑‍🎨
<p align="center">
  📚 <a href="https://arxiv.org/abs/2407.09298">[Paper]</a>
</p>

## Requirements
1. Please run following commands to set up basic python environment
```
git clone git@github.com:floatingbigcat/transformer-as-painter.git
cd transformer-as-painter

# We use python 3.10

python -m venv painter_env
source painter_env/bin/activate
pip install -r requirements.txt
```

2. Our evaluation of GPT style model is based on [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness), we fix the module commit to to make sure our experiments is reproducible, and make minmal modification on `llama` and `mistral` to enable our methods on GPT style model.
Please run the following commands to install lm_eval on the current python environment.
```
cd gpt

cd lm-eval
git submodule update --init

cp -f ../__main__.py lm_eval/
cp -f ../evaluator.py lm_eval/
cp -f ../modify_model.py lm_eval/
cp -f ../routing_llama.py lm_eval/
cp -f ../routing_mistral.py lm_eval/
cp -f ../routing_neox.py lm_eval/

pip install -e .
```

## Example Usage
Basically, you can run all our methods by simiply change the argument in the `example.sh` under gpt/ or bert/

### GPT 
```
cd gpt
bash example.sh
```

### Bert 
```
cd bert
bash example.sh
```

## Cosine Similiary Plot

please check `./cos_sim_plotter.ipynb` about how we obtain the cosine similiary heat map of hidden states over layers

