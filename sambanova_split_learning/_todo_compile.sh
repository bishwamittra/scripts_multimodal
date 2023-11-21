python compile_u_shaped.py compile \
-b 256 \
--model rescale18 \
--in-height 8 \
--in-width 8 \
--channels 64 \
--num-classes 10 \
--compiler-configs-file ./configs/compiler_configs/rescale_compiler_configs_v2.json \
--drop-conv 0.0 \
--drop-fc 0.0 \
--mac-v2 \
--pef-name pef_rescale18_split_u_shaped \
--output-folder ./ \
--orig-in-height 32 \
--orig-in-width 32 \
--arch sn30 \
--compute-input-grad



# -o0 \