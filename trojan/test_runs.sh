for DATASET in pdf mnist cifar10 driving
do
  python experiment.py rsc $DATASET --test_run --no_output
done
