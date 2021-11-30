
for NUMQUERIES in 100 200 500 1000 2000 5000 10000 20000 50000 ; do
	srun -p p100 --gres=gpu:1 --mem=16G python steal.py \
		-data ~/datasets/cifar10 -dataset-name cifar10 \
		--log-every-n-steps 100 --epochs 100 \
		--num_queries=$NUMQUERIES \
		--n-views 1 \
		--arch resnet18 \
		--folder_name "resnet18_100-epochs_cifar10" \
		--logdir "stolen_resnet18_100epochs_cifar10_noaug_queries$NUMQUERIES" &
done
