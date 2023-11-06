client_start=True


server_seed="42"

if [ "$client_start" = True ] ; then
    server_wait_time="60"
else
    server_wait_time="30"
fi



for i in {1..3}
do
    seed=$((server_seed + i))
    sleep ${server_wait_time}
    if [ "$client_start" = True ] ; then
        python server_resnet50_u_shaped.py --seed ${seed} --connection_start_from_client
    else
        python server_resnet50_u_shaped.py --seed ${seed}
    fi
done


