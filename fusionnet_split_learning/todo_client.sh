client_start=False

client_seed="42"
epoch="3"

if [ "$client_start" = True ] ; then
    client_wait_time="60"
else
    client_wait_time="100"
fi

for architecture_choice in {1..4}
do
    for cd_method in "no_compression" "zstd"
    do
        for i in {1..3}
        do
            seed=$((client_seed + i))
            if [ "$client_start" = True ] ; then
                python client_u_shaped.py --cd_method ${cd_method} --epoch ${epoch} --seed ${seed} --architecture_choice ${architecture_choice} --connection_start_from_client --client_in_sambanova --save_root result
            else
                python client_u_shaped.py --cd_method ${cd_method} --epoch ${epoch} --seed ${seed} --architecture_choice ${architecture_choice} --save_root result
            fi
            sleep ${client_wait_time}
        done
    done
done






