import pnrscript


def partition_0_0(prism):
    node = prism.graph.template_graph["rescale__layer1__0__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__0__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__0__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__1__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__1__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__1__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")


def partition_0_1(prism):
    node = prism.graph.template_graph["rescale__layer1__2__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__2__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__2__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__0__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__0__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__0__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__1__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__1__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__1__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")


def partition_0_2(prism):
    node = prism.graph.template_graph["rescale__layer2__2__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__2__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__2__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__3__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__3__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__3__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__0__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__0__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__0__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")


def partition_0_3(prism):
    node = prism.graph.template_graph["rescale__layer3__1__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__1__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__1__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__2__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__2__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__2__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__3__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__3__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__3__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")


def partition_0_4(prism):
    node = prism.graph.template_graph["rescale__layer3__4__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__4__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__4__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__5__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__5__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__5__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer4__0__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer4__0__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer4__0__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")


def partition_0_5(prism):
    node = prism.graph.template_graph["rescale__layer4__1__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__1__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer4__1__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer4__2__conv1__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer4__2__conv2__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer4__2__conv3__conv2d"]
    prism.board.set_template_perf_attr(node, "kUser4")


def partition_0_8(prism):
    node = prism.graph.template_graph["rescale__layer4__2__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer4__2__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__2__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__2__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer4__2__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer4__2__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__2__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")


def partition_0_9(prism):
    node = prism.graph.template_graph["rescale__layer4__1__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__1__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__1__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__1__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__1__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__1__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__1__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")


def partition_0_10(prism):
    node = prism.graph.template_graph["rescale__layer4__0__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__0__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__0__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__0__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__0__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer4__0__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer4__0__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")


def partition_0_11(prism):
    node = prism.graph.template_graph["rescale__layer3__5__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__5__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__5__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__5__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__5__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__5__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__5__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__5__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__5__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")


def partition_0_12(prism):
    node = prism.graph.template_graph["rescale__layer3__4__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__4__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__4__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__4__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__4__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__4__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__4__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__4__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__4__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")


def partition_0_13(prism):
    node = prism.graph.template_graph["rescale__layer3__3__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__3__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__3__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__3__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__3__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__3__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__3__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__3__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__3__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")


def partition_0_14(prism):
    node = prism.graph.template_graph["rescale__layer3__2__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__2__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__2__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__2__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__2__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__2__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__2__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__2__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__2__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")


def partition_0_15(prism):
    node = prism.graph.template_graph["rescale__layer3__1__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__1__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__1__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__1__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__1__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__1__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__1__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__1__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__1__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")


def partition_0_16(prism):
    node = prism.graph.template_graph["rescale__layer2__3__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__3__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__3__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer3__0__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__0__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__0__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__0__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__0__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__0__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__0__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__0__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer3__0__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__3__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__3__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__3__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__3__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__3__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__3__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")


def partition_0_17(prism):
    node = prism.graph.template_graph["rescale__layer2__1__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__1__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__1__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__2__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__2__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__2__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__2__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__2__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__2__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__2__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__2__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__2__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__1__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__1__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__1__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__1__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__1__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__1__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")


def partition_0_18(prism):
    node = prism.graph.template_graph["rescale__layer1__2__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__2__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__2__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__0__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__0__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__0__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__0__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__0__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__0__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__0__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer2__0__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer2__0__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__2__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__2__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__2__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__2__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__2__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__2__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")


def partition_0_19(prism):
    node = prism.graph.template_graph["rescale__layer1__0__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__0__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__0__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__1__conv1__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__1__conv2__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__1__conv3__conv2d_recompute_"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__1__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__1__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__1__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__1__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__1__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__1__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__0__conv3__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__0__conv3__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__0__conv2__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__0__conv2__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser4")
    node = prism.graph.template_graph["rescale__layer1__0__conv1__conv2d_bwd_loss"]
    prism.board.set_template_perf_attr(node, "kUser3")
    node = prism.graph.template_graph["rescale__layer1__0__conv1__conv2d_bwd_grad"]
    prism.board.set_template_perf_attr(node, "kUser3")
