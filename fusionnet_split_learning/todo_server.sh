client_start=True

server_seed="42"

if [ "$client_start" = True ] ; then
    server_wait_time="100"
else
    server_wait_time="60"
fi

for architecture_choice in {1..4}
do
    for cd_method in "no_compression" "zstd"
    do
        for i in {1..3}
        do
            seed=$((server_seed + i))
            if [ "$client_start" = True ] ; then
                python server_u_shaped.py --seed ${seed} --connection_start_from_client --client_in_sambanova --save_root result
            else
                python server_u_shaped.py --seed ${seed} --save_root result
            fi
            sleep ${server_wait_time}
        done
    done
done

