client_seed="42"
epoch="100"
device="cuda"

for architecture_choice in 0 5
do 
    for i in {1..3}
    do
        seed=$((client_seed + i))
        python main.py --epoch ${epoch} --seed ${seed} --architecture_choice ${architecture_choice} --device ${device} --cuda_id 1 --save_root result_non_split
    done

done







