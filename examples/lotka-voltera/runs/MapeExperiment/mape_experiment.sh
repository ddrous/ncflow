## Loop over the number of ncf sample sizes
for beta in $(seq 0.27 0.0316 1.22)
do
for delta in $(seq 0.27 0.0316 1.22)
do
## pretend the result of echo to base_ncf_sample_size.py
echo -e "beta = $beta\ndelta = $delta\n$(cat base_dataset_mape.py)" > dataset_mape.py
# python main_mape.py > /dev/null
mape_score=$(python main_mape.py | tail -n 1)
## Print the return value of the main_mape.py
echo "beta = $beta, delta = $delta, mape = $mape_score"
## Append the result to a csv file
echo "$beta,$delta,$mape_score" >> mape_scores.csv
done
done
