python compile_u_shaped.py run \
-b 256 \
--model rescale18 \
--in-height 8 \
--in-width 8 \
--channels 64 \
--num-classes 10 \
--drop-conv 0.0 \
--drop-fc 0.0 \
--mac-v2 \
--pef rescale18_split_u_shaped/rescale18_split_u_shaped.pef \
--orig-in-height 32 \
--orig-in-width 32 \
--compute-input-grad



# python compile.py run \
# -b 256 \
# --model rescale18 \
# --in-height 8 \
# --in-width 8 \
# --channels 64 \
# --num-classes 10 \
# --drop-conv 0.0 \
# --drop-fc 0.0 \
# --mac-v2 \
# --pef rescale18_split_test/rescale18_split_test.pef \
# --orig-in-height 32 \
# --orig-in-width 32 \
# --compute-input-grad
