client_start=True


server_seed="42"

if [ "$client_start" = True ] ; then
    server_wait_time="100"
else
    server_wait_time="60"
fi


for i in {1..3}
do
    seed=$((server_seed + i))
    if [ "$client_start" = True ] ; then
        python server_resnet50_u_shaped.py --seed ${seed} --connection_start_from_client --client_in_sambanova
    else
        python server_resnet50_u_shaped.py --seed ${seed}
    fi
    sleep ${server_wait_time}
done


