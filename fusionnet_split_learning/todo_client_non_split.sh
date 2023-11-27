client_seed="42"
epoch="2"

for i in {1..3}
do
    seed=$((client_seed + i))
    python main.py --epoch ${epoch} --seed ${seed} --device cpu
done






