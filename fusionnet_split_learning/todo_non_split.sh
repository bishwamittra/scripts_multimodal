client_seed="42"
epoch="100"

for i in {1..3}
do
    seed=$((client_seed + i))
    python main.py --epoch ${epoch} --seed ${seed} --device cuda --cuda_id 1 --save_root result
done






