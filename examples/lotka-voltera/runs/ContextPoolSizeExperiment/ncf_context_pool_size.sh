
## Loop over the number of ncf sample sizes
for i in $(seq 1 9)
do
    ## pretend the result of echo to base_ncf_sample_size.py
    echo -e "context_pool_size = $i\n$(cat main_T2_base.py)" > main_T2_$i.py
    python main_T2_$i.py
done
