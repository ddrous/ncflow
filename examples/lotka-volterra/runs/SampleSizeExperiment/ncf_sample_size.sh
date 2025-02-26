
## Loop over the number of ncf sample sizes
for i in $(seq 1 9)
do
    ## pretend the result of echo to base_ncf_sample_size.py
    echo -e "ncf_sample_size = $i\n$(cat base_ncf_sample_size.py)" > main_$i.py
    python main_$i.py
done
