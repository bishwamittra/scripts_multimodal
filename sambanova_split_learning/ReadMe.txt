ReadMe:

1) The suggested changes which are to be done before Compiling "compile" on Running "run" the Compile.py file.

    1.1) For compilation/generating .pef check if the directory structure/path is correct on line no. 13 of compile.py file,
         line no.3 of rescale.py file and line no.2 of the rescale_estimator.py file, otherwise make the required changes.
        
        "compile" command:

        python compile.py compile \
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
        --arch sn30 \
        --pef-name rescale18_split \
        --output-folder=./ \
        --orig-in-height 32 \
        --orig-in-width 32

    1.2) For generating "trace_graph" using "run" command on compile.py file, make sure to pass the correct path for the generated .pef file.

        "run" command:

        python compile.py run \
        -b 256 \
        -p ./rescale18_split/rescale18_split.pef \
        -v \
        --in-height 8 \
        --model rescale18 \
        --drop-conv 0.0 \
        --drop-fc 0.0 \
        --mac-v2 \
        --in-width 8 \
        --channels 64 \
        --num-classes 10 \
        --orig-in-height 32 \
        --orig-in-width 32 \
        --device CPU # for printing the summary of the model using torchinfo


2) The changes which were made in the client code to resolve the trace_graph issue.

    For the issue that is reported, the reason is that there was a difference in the model definition and optimizer that was used during compilation vs. running it. 
    Here's the main changes made to run the trace_graph successfully in server_split.py and now the consolidated compile.py file:

        2.1) Model definition:
        #self.mean_pool = nn.AvgPool2d((input_shapes[0] // 8, input_shapes[1] // 8))
        self.mean_pool = nn.AvgPool2d((input_shapes[0] // 32, input_shapes[1] // 32))

        2.2) Instantiating the model:
        def rescale18(num_classes = 10, drop_conv = 0.0, drop_fc = 0.0, **kwargs):
        return ReScale([2, 2, 2, 2],
        num_classes = num_classes,
        drop_conv = drop_conv,
        input_shapes = (32, 32), # KT Orig values: (8, 8) Values passed during compilation: (32, 32)

        2.3) Optimizer:
        AdamW used in compilation, but SGD used in server_split.py
