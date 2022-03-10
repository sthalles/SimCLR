python run.py -dataset-name cifar10 --level_number 4 --loss_at_all_level --epochs 1000 
python run.py -dataset-name cifar10 --level_number 4 --loss_at_all_level --epochs 1000  --regularization --per_level
python run.py -dataset-name cifar10 --level_number 4 --loss_at_all_level --epochs 1000  --regularization --per_node
python run.py -dataset-name cifar10 --level_number 4 --loss_at_all_level --epochs 1000  --regularization --regularization_at_all_level --per_level
python run.py -dataset-name cifar10 --level_number 4 --loss_at_all_level --epochs 1000  --regularization --regularization_at_all_level --per_node

