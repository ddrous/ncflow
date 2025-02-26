
## Loop over the number of ncf sample sizes
for i in $(seq 1 1 10)
do
    ## pretend the result of echo to log_ctx_size.py
    echo -e "log_ctx_size = $i\n$(cat main_T2_base.py)" > main_T2_$i.py
    python main_T2_$i.py
done
