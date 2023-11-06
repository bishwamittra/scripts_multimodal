client_start=True



client_seed="42"
epoch="2"

if [ "$client_start" = True ] ; then
    client_wait_time="60"
else
    client_wait_time="100"
fi


for i in {1..3}
do
    seed=$((client_seed + i))
    sleep ${client_wait_time}
    if [ "$client_start" = True ] ; then
        python client_resnet50_u_shaped.py --epoch ${epoch} --seed ${seed} --connection_start_from_client --client_in_sambanova
    else
        python client_resnet50_u_shaped.py --epoch ${epoch} --seed ${seed}
    fi
done






