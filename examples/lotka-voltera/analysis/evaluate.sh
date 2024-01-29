## Run the command "python main.py --seed 000" with different values of the seed unifromly spread in [0, 1000]

# for seed in $(seq 10 12)
for seed in $(seq 1000 3000)
do
    python main.py --seed $seed
done


