client_start=True



client_seed="42"
epoch="2"

if [ "$client_start" = True ] ; then
    client_wait_time="30"
else
    client_wait_time="60"
fi


for i in {1..3}
do
    seed=$((client_seed + i))
    sleep ${client_wait_time}
    if [ "$client_start" = True ] ; then
        python client_resnet50_u_shaped.py --epoch ${epoch} --seed ${seed} --connection_start_from_client
    else
        python client_resnet50_u_shaped.py --epoch ${epoch} --seed ${seed}
    fi
done






