import sys
# sys.path.append("/nvmedata/scratch/shrirajp/Astar_updated_code/classification_1.16.5-38/classification/")

import csv
import os
import time
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from classification.estimators.utils import load_data_pipeline, all_gather
from rescale import rescale18, rescale50
# from rescale_u_shaped import rescale18, rescale50
from classification.schedulers.warmup_scheduler import WarmupScheduler
from classification.schema import common_schema, schema
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import models
from tqdm import tqdm

import sambaflow.samba as samba
import sambaflow.samba.utils as sn_utils
from samba_tools import interface
from samba_tools.checkpointer import Checkpointer, load_checkpoint_from_file
from samba_tools.constants import EstimatorMode
from samba_tools.distributed import allow_interrupt
from samba_tools.logging import Logger
from sambaflow.samba import SambaTensor
from sambaflow.samba.sambaloader import SambaLoader

# from .utils import all_gather


def _ckpt_dict_filter_fn(model, ckpt_dict):
    '''
    Function to edit a checkpoint before loading, and to determine whether to load checkpoint strictly.
    Must accept 2 arguments: `model` and `ckpt_dict`.

    Inputs:
        model : torch.nn.Module
        ckpt_dict : checkpoint dictionary
    Returns:
        ckpt_dict : modified checkpoint dictionary
        strict : bool specifying if checkpoint should be loaded strictly
    '''
    strict = True

    num_classes = model.fc.weight.shape[0]

    # If training a model with a different number of classes, remove last linear
    if num_classes != ckpt_dict['state_dict']['fc.weight'].shape[0]:
        strict = False
        ckpt_dict['state_dict'].pop('fc.weight')
        ckpt_dict['state_dict'].pop('fc.bias')

    return ckpt_dict, strict


interrupter = allow_interrupt()


class RescaleEstimator(interface.Estimator):
    def __init__(self, task_cfg: common_schema.TaskBase, meta_cfg: common_schema.MetaBase, model_cfg: schema.Model,
                 run_cfg: schema.Run, optim_cfg: schema.Optim, data_cfg: schema.Data, data_pipeline: dict,
                 regression_test_cfg: schema.RegressionTest, logging_cfg: schema.Logging, analysis_cfg: schema.Analysis,
                 mode: str) -> None:

        self.mode = mode

        EstimatorMode.assert_value_is_legal(mode)
        if meta_cfg.pef is not None:
            self.pef = str(meta_cfg.pef)
        else:
            self.pef = None

        self.pinned_memory = None

        # Required model arguments for mode=compile
        self.num_classes = model_cfg.num_classes
        self.drop_conv = model_cfg.drop_conv
        self.drop_fc = model_cfg.drop_fc
        self.hot_layers = model_cfg.hot_layers
        self.model_type = model_cfg.model
        self.device = model_cfg.device
        self.in_height = model_cfg.in_height
        self.in_width = model_cfg.in_width
        self.channels = model_cfg.channels
        self.num_flexible_classes = model_cfg.num_flexible_classes
        self.weighted_cross_entropy = model_cfg.weighted_cross_entropy
        self.class_weights = model_cfg.class_weights
        self.compute_input_grad = model_cfg.compute_input_grad
        self.batch_size = model_cfg.batch_size
        self.mapping = model_cfg.mapping
        # KT TEST SPLIT LEARNING
        self.orig_in_height = model_cfg.orig_in_height
        self.orig_in_width = model_cfg.orig_in_width

        self.inference = model_cfg.inference

        self.optimizer_type = optim_cfg.optimizer
        self.learning_rate = optim_cfg.learning_rate
        self.weight_decay = optim_cfg.weight_decay
        self.momentum = optim_cfg.momentum
        self.schedulers = None

        self.acc_test = False
        self.run_benchmark = regression_test_cfg.run_benchmark
        self.benchmark_steps = regression_test_cfg.benchmark_steps if self.run_benchmark else 0
        self.benchmark_warmup_steps = regression_test_cfg.benchmark_warmup_steps if self.run_benchmark else 0

        self.ckpt_file = None
        self.resume = False
        self.log_dir = None

        mp.set_start_method('forkserver')

        self._assert_model_config_ok()

        # Additional arguments for mode=train or predict (running from rescale_hook.py)
        if mode in [EstimatorMode.TRAIN, EstimatorMode.EVAL, EstimatorMode.PREDICT]:
            self.pinned_memory = run_cfg.pinned_memory

            if self.pinned_memory:
                print("Enabling pinned memory")
                samba.session.enable_pinned_memory()

            self.task_name = task_cfg.task_name

            self.epochs = run_cfg.epochs
            self.resume = run_cfg.resume
            self.seed = run_cfg.seed
            self.num_workers = run_cfg.num_workers
            self.use_ddp = run_cfg.use_ddp
            self.num_samples_per_iter = self.batch_size * sn_utils.get_world_size()
            self.data_init_iters = run_cfg.data_init_iters
            self.use_distributed_val = run_cfg.use_distributed_val
            self.use_sambaloader = run_cfg.use_sambaloader
            self.data_parallel = model_cfg.data_parallel
            self.reduce_on_rdu = model_cfg.reduce_on_rdu

            self.multi_step = optim_cfg.multi_step
            self.gamma = optim_cfg.gamma
            self.scheduler_type = optim_cfg.scheduler
            self.warmup_epochs = optim_cfg.warmup_epochs

            self.acc_test = regression_test_cfg.acc_test

            self.generate_cams = analysis_cfg.generate_cams

            if data_cfg.data_dir is not None:
                self.data_dir = str(data_cfg.data_dir)
            else:
                self.data_dir = None

            self.data_pipeline = data_pipeline

            self.print_freq = logging_cfg.print_freq
            self.log_dir = logging_cfg.log_dir
            self.ckpt_dir = logging_cfg.ckpt_dir or logging_cfg.log_dir
            self.ckpt_file = logging_cfg.ckpt_file
            self.log_type = logging_cfg.log_type
            self.log_to_stdout = logging_cfg.log_to_stdout
            self.run_tag = logging_cfg.run_tag
            self.setup_subdirs = logging_cfg.setup_subdirs

            # Some defaults
            if self.device == 'GPU':
                self.rank = int(os.environ.get('LOCAL_RANK', 0))
            else:
                self.rank = int(os.environ.get('PMI_RANK', 0))
                assert self.use_ddp is False, "--use-ddp can only be used with GPU"
            self._global_step = 0
            self.best_val = 0.0
            if self.log_dir is not None:
                self.log_dir = Path(self.log_dir)

            # Log to TB by default
            self.logger = Logger(
                loggers_to_use=[self.log_type, 'tensorboard'],
                root_dir=self.log_dir,
                rank=self.rank,
                log_to_stdout=self.log_to_stdout,
                project=self.run_tag,
                setup_subdirs=self.setup_subdirs,
            )

            if self.ckpt_dir:
                self.checkpointer = Checkpointer(
                    ckpt_dir=self.ckpt_dir,
                    keep_last_n=logging_cfg.n_keep,
                    read_only=self.acc_test  # do not overwrite any checkpoints when running an accuracy test
                )
            else:
                self.checkpointer = None

            self._assert_run_config_ok()

        # Initialize model, optimizer and criterion
        self.model = self.init_model(self.model_type)
        self.optimizers = self.init_optim(self.model, self.model_type)
        if self.weighted_cross_entropy:
            if self.device == 'GPU':
                self.criterion = nn.CrossEntropyLoss(
                    torch.tensor(self.class_weights, dtype=torch.float, device=self.rank))
            else:
                self.criterion = nn.CrossEntropyLoss(torch.tensor(self.class_weights, dtype=torch.float))
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.dropout_hyperparam_dict = None
        self.input_names = None
        """
        Convert input data to bf16 while reading data. This will be true
        by default when the device is RDU to allow setting input tensors by
        session.run
        """
        self.use_bf16 = self.device == "RDU"

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        else:
            print("log dir not specified. Will not be logging")

    @classmethod
    def from_argparse(cls, args, mode):

        if mode == EstimatorMode.COMPILE:
            transforms = None
        else:
            transforms = load_data_pipeline(args)

        return cls(task_cfg=args,
                   meta_cfg=args,
                   model_cfg=args,
                   run_cfg=args,
                   optim_cfg=args,
                   data_cfg=args,
                   data_pipeline=transforms,
                   regression_test_cfg=args,
                   logging_cfg=args,
                   analysis_cfg=args,
                   mode=mode)

    @classmethod
    def from_pydantic(cls, cfg, mode):

        if cfg.data is not None:
            transforms = cfg.data.transforms
        else:
            transforms = None

        return cls(task_cfg=cfg.task,
                   meta_cfg=cfg.meta,
                   model_cfg=cfg.model,
                   run_cfg=cfg.run,
                   optim_cfg=cfg.optim,
                   data_cfg=cfg.data,
                   data_pipeline=transforms,
                   regression_test_cfg=cfg.regression_test,
                   logging_cfg=cfg.logging,
                   analysis_cfg=cfg.analysis,
                   mode=mode)

    def compile(self):
        raise NotImplementedError

    def _assert_model_config_ok(self):
        if self.num_flexible_classes != -1 and not self.num_classes >= self.num_flexible_classes:
            raise ValueError("Number of classes should be greater than or equal to number of flexible classes.")

    def _assert_run_config_ok(self):
        if self.mode in [EstimatorMode.PREDICT, EstimatorMode.EVAL
                         ] and self.ckpt_file is None and self.ckpt_dir is None:
            raise ValueError("Need to specify either ckpt-dir or ckpt-file when mode is predict")
        if self.mode == EstimatorMode.TRAIN and self.resume and self.ckpt_file is None and self.ckpt_dir is None:
            raise ValueError("Need to specify either ckpt-dir or ckpt-file when resuming")

    def __del__(self):
        if self.mode in ['train', 'predict']:
            if self.run_benchmark:
                samba.session.end_samba_profile(time.strftime("samba_profile_%Y-%m-%dT%H.%M.%S.txt"))

            if self.use_ddp and dist.is_initialized():
                self.cleanup_distributed_training()

    def setup_distributed_training(self):
        """
        Ensure distributed training is set up.
        Requires environment variables LOCAL_RANK and WORLD_SIZE.
        MASTER_ADDR and MASTER_PORT are optionally set.
        """
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12321'
        dist.init_process_group('nccl', init_method='env://')

    def cleanup_distributed_training(self):
        dist.destroy_process_group()

    def init_model(self, model_type: str) -> torch.nn.Module:
        """
        Create model with pretrained weights and also freeze layers.
        """

        if model_type.startswith("rescale"):
            if model_type == "rescale18":
                model_class = rescale18
            elif model_type == "rescale50":
                model_class = rescale50
            else:
                raise Exception(f"Unsupported rescale:{model_type} type")
            model = model_class(num_classes=self.num_classes,
                                drop_conv=self.drop_conv,
                                drop_fc=self.drop_fc,
                                # KT TEST SPLIT LEARNING
                                #input_shapes=(self.in_height, self.in_width),
                                input_shapes=(self.orig_in_height, self.orig_in_width),
                                num_flexible_classes=self.num_flexible_classes)

            if self.inference:
                model.eval()

            if self.hot_layers is not None:
                hot_layers = [f"layer{layer}" for layer in self.hot_layers]
                print(f"The following layers are hot {hot_layers}")

                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue

                    requires_grad = False

                    # These nodes are always hot
                    if name in ['addbias2._bias', 'addbias2.dummy_conv.bias', 'fc.weight', 'fc.bias']:
                        requires_grad = True
                        continue

                    for layer in hot_layers:
                        if layer in name:
                            requires_grad = True
                            break

                    param.requires_grad = requires_grad

        elif model_type == "resnet18":
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)

            for name, param in model.named_parameters():
                if "layer4" in name or name in ['fc.weight', 'fc.bias']:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise Exception("{model_type} not supported")

        if self.device == "GPU":
            # setup
            if self.use_ddp:
                self.setup_distributed_training()
                torch.cuda.set_device(self.rank)

                model = model.to(self.rank)

                # initialize distributed data parallel (DDP)
                model = DDP(model, device_ids=[self.rank], output_device=self.rank)
            else:
                model.cuda()

        elif self.device == "RDU":
            samba.from_torch_model_(model)

        return model

    def init_optim(self, model: torch.nn.Module, model_type: str) -> torch.optim:
        """
        Initialize optimizer based on the model type.
        """
        if model_type.startswith("rescale"):
            params_w_decay = []
            params_wo_decay = []
            for name, p in model.named_parameters():
                if p.requires_grad:
                    if 'addbias' in name or '_scale' in name:
                        params_wo_decay.append(p)
                    else:
                        params_w_decay.append(p)
            if self.optimizer_type == 'adamw':
                optim = [
                    samba.optim.AdamW(params_wo_decay, lr=self.learning_rate, betas=(0.9, 0.997), weight_decay=0),
                    samba.optim.AdamW(
                        params_w_decay, lr=self.learning_rate, betas=(0.9, 0.997), weight_decay=self.weight_decay)
                ] if not self.inference else None
            elif self.optimizer_type == 'sgd':
                optim = [
                    samba.optim.SGD(params_wo_decay, lr=self.learning_rate, weight_decay=0, momentum=self.momentum),
                    samba.optim.SGD(
                        params_w_decay, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
                ] if not self.inference else None

        elif model_type == "resnet18" and self.device != "RDU":
            if self.optimizer_type == 'adamw':
                optim = samba.optim.AdamW(
                    model.parameters(), lr=self.learning_rate, betas=(0.9, 0.997),
                    weight_decay=self.weight_decay) if not self.inference else None
            elif self.optimizer_type == 'sgd':
                optim = samba.optim.SGD(
                    model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                    momentum=self.momentum) if not self.inference else None

        else:
            raise Exception("InvalidConfig: Optimizer could not be initialized")

        return optim

    def init_schedulers(self, loader: DataLoader) -> Optional[WarmupScheduler]:
        if self.scheduler_type == 'step':
            if self.multi_step is None:
                self.multi_step = [int((i + 1) * (self.epochs // 3.3)) for i in range(3)]  # [30, 60, 90] for 100 epochs

            return [
                WarmupScheduler(base_lr=self.learning_rate,
                                iter_per_epoch=len(loader),
                                max_epoch=self.epochs,
                                multi_step=self.multi_step,
                                gamma=self.gamma,
                                warmup_epoch=self.warmup_epochs)
            ]
        return None

    def setup(self, loader: DataLoader = None, trace_graph=True) -> None:
        """Initialize the network graph

        Two steps for initialization
        1. Load from a checkpoint if data_dir contains pre-trained model
        2. Trace graph is `trace_graph` is True

        trace_graph should be provided when the model is run on RDU and a PEF has been generated. trace_graph should not
        be called during compilation of the network 
        """
        skip_initialization = False
        if self.mode in [EstimatorMode.PREDICT, EstimatorMode.EVAL]:
            # Load best checkpoint or ckpt_file
            self.load_checkpoint(True, ckpt_file=self.ckpt_file)
        elif self.mode == EstimatorMode.TRAIN:
            self.schedulers = self.init_schedulers(loader=loader)
            if self.resume or self.ckpt_file:
                skip_initialization = True
                self.load_checkpoint(ckpt_file=self.ckpt_file)
                self.logger.log_line(f"Continuing training from step {self._global_step}")

        # Do not trace while compiling.
        if trace_graph and self.device == 'RDU':
            if self.mode == EstimatorMode.TRAIN and not skip_initialization:
                self.logger.log_line("data dependent initialization of bias")
                for count, sample in enumerate(loader):
                    if count >= self.data_init_iters:
                        break
                    self.model.init_pass(sample[0].float(), count)

            inputs = (samba.randn(self.batch_size,
                                  self.channels,
                                  self.in_height,
                                  self.in_width,
                                  name='input',
                                  batch_dim=0,
                                  requires_grad=self.compute_input_grad).bfloat16(), )

            self.input_names = [ipt.sn_name for ipt in inputs]

            sn_utils.trace_graph(self.model,
                                 inputs,
                                 self.optimizers,
                                 init_output_grads=not self.inference,
                                 pef=self.pef,
                                 mapping=self.mapping,
                                 dev_mode=True)

            samba.session.start_samba_profile()

    def compute_log_metrics(self,
                            mode: str,
                            prediction: torch.Tensor,
                            label: torch.Tensor,
                            loss: torch.Tensor,
                            prediction_is_probabilities: bool = True) -> Dict[str, Any]:

        if self.device == "RDU":
            prediction = samba.to_torch(prediction)
            label = samba.to_torch(label)
        log_metrics = {}
        log_metrics["loss"] = loss.item()
        if prediction_is_probabilities:
            prediction = torch.argmax(prediction, dim=-1)

        log_metrics["accuracy"] = self.get_accuracy(prediction, label)

        # truepos, falsepos, trueneg, falseneg = self.evaluate_metrics(prediction, label)
        # log_metrics["precision"] = self.get_precision(truepos, falsepos, trueneg, falseneg)
        # log_metrics["recall"] = self.get_recall(truepos, falsepos, trueneg, falseneg)
        # log_metrics["truepos"] = truepos
        # log_metrics["falsepos"] = falsepos
        # log_metrics["trueneg"] = trueneg
        # log_metrics["falseneg"] = falseneg

        return log_metrics

    def aggregate_test_metrics(self, accuracy_metrics: Dict[str, Any]) -> Dict[str, Any]:
        loss_sum = 0
        acc_sum = 0
        truepos_sum = 0
        falsepos_sum = 0
        trueneg_sum = 0
        falseneg_sum = 0

        for acc_metric in accuracy_metrics:
            # Aggregate across different steps
            loss_sum += acc_metric.get("loss", 0)
            acc_sum += acc_metric.get("accuracy", 0)

            truepos_sum += acc_metric.get("truepos", 0)
            falsepos_sum += acc_metric.get("falsepos", 0)
            trueneg_sum += acc_metric.get("trueneg", 0)
            falseneg_sum += acc_metric.get("falseneg", 0)

        val_metrics = {
            "loss": loss_sum / len(accuracy_metrics),
            "accuracy": acc_sum / len(accuracy_metrics),
            # "truepos": truepos_sum,
            # "falsepos": falsepos_sum,
            # "trueneg": trueneg_sum,
            # "falseneg": falseneg_sum,
        }

        return val_metrics

    def get_precision(self, truepos: int, falsepos: int, trueneg: int, falseneg: int) -> float:
        denomenator = truepos + falsepos
        if denomenator == 0:
            return 0
        return float(truepos) / float(denomenator)

    def get_recall(self, truepos: int, falsepos: int, trueneg: int, falseneg: int) -> float:
        denomenator = truepos + falseneg
        if denomenator == 0:
            return 0
        return float(truepos) / float(denomenator)

    def get_accuracy(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        return ((preds == labels).sum() / np.prod(preds.shape)).item()

    def setup_cams(self):
        self.cam_counter = 0
        self.extracted_features = None
        if self.device == 'RDU':
            if self.model_type == 'rescale18':
                self.cam_layer_name = 'rescale__layer4__1__relu_10_used_by_rescale__addbias2__add'
        else:
            if self.model_type == 'rescale18':
                pre_pool_name = 'addbias2'
            else:
                print(f'CAM generation not supported for model: {self.model_type}')
                self.generate_cams = False
                return

            # Get output after final conv
            def get_extracted_features(module, input, output):
                self.extracted_features = output.data.cpu().numpy()

            self.model._modules.get(pre_pool_name).register_forward_hook(get_extracted_features)

    def generateCAM(self, input_images: np.array, feature_map: np.array, predictions: np.array,
                    labels: np.array = None) -> List[str]:
        """
        Generate a Class Activation Map after Global Average Pooling
        by highlighting activations on convolution feature map for
        each class.

        Saves images under 'self.log_dir/cams/imgX_predY[_labZ].jpg'

        Args:
            input_images: np.array - a set of images with shape (N, C, H, W)
            feature_map: np.array - output of final convolution before
                                pooling layer. (N, features, h, w)
            predictions: np.array - model output (softmax) with shape (N, num_classes)
        optional:
            labels: np.array - image labels if provided for filename (N)

        Returns:
        output_image_paths: List[str] - list of output paths for cams
        """
        # Get output dir:
        if self.get_log_filepath() is None:
            return
        output_dir = os.path.join(self.log_dir, 'cams')
        os.makedirs(output_dir, exist_ok=True)

        # Get weights used for generating class predictions
        model_params = list(self.model.parameters())
        weights = np.squeeze(model_params[-2].data.numpy())

        output_size = (self.in_width, self.in_height)
        bs, nfeatures, height, width = feature_map.shape
        pred_classes = np.argmax(predictions, axis=-1)

        colormap = plt.get_cmap('jet')

        output_image_paths = []
        for i, image in enumerate(input_images):
            pred_class = pred_classes[i]
            class_map = weights[pred_class].dot(feature_map[i].reshape((nfeatures, height * width)))
            class_map = class_map.reshape(height, width)

            original = np.moveaxis(np.uint8(255 * (image - np.min(image)) / (np.max(image) - np.min(image))), 0, 2)

            # Turn map into single-channel image
            class_map = np.uint8(255 * (class_map - np.min(class_map)) / (np.max(class_map) - np.min(class_map)))
            class_map = PIL.Image.fromarray(class_map)
            class_map = class_map.resize(output_size, resample=PIL.Image.BILINEAR)

            cam = colormap(np.array(class_map))
            cam = np.uint8(255 * cam[:, :, :3]) * 0.5 + original * 0.5
            cam = np.uint8(255 * (cam - np.min(cam)) / (np.max(cam) - np.min(cam)))

            # TODO: (davidku) Replace cam_counter with image metadata
            output_path = os.path.join(output_dir, 'img' + str(self.cam_counter) + '_pred' + str(pred_class))
            if labels is not None:
                output_path += '_lab' + str(labels[i])
            output_image_paths.append(output_path + '.jpg')

            # Save the images
            PIL.Image.fromarray(cam).save(output_image_paths[-1])
            PIL.Image.fromarray(original).save(output_path + '_in.jpg')
            self.cam_counter += 1
        return output_image_paths

    def get_fieldnames(self):
        return ["image_path", "prediction"]

    @torch.no_grad()
    @samba.session.profiler.event("prediction_step")
    def prediction_step(self, images):
        output = self.forward_pass(images, mode=self.mode)
        return output

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        self.setup(loader=loader)
        self.model.eval()

        if self.use_sambaloader:
            function_hook = lambda data: [data[0]]
            loader = SambaLoader(loader, self.input_names, function_hook=function_hook, return_original_batch=True)

        # TODO: can we pass in a loader where this logic has already been done?
        if self.run_benchmark:
            num_predict_steps = self.benchmark_steps
            loader_iter = iter(cycle(loader))
        else:
            num_predict_steps = len(loader)
            loader_iter = iter(loader)

        prediction_dir = Path(os.path.join(self.log_dir, 'predictions'))
        prediction_dir.mkdir(exist_ok=True)
        csv_filename = prediction_dir / f'output_predict.csv'
        fieldnames = self.get_fieldnames()

        for _ in range(num_predict_steps):

            with samba.session.profiler.event("predict_batch"):
                images, target, metadata = self.fetch_data(loader_iter)
                output = self.prediction_step(images)

            # TODO: This needs to move to a diferent argument
            if not self.run_benchmark:
                prediction = torch.argmax(output, dim=-1)
                self.save_predictions_info(csv_filename, fieldnames, prediction_dir, metadata, prediction)

        profile_keys = ['data_step', 'session_forward', 'prediction_step', 'predict_batch']
        self.logger.log_line(f"reporting statistics over {num_predict_steps} steps:")
        stats = samba.session.profiler.print_statistics(profile_keys,
                                                        self.benchmark_warmup_steps,
                                                        num_samples_per_iter=self.num_samples_per_iter)
        self.logger.log_line(stats)

    def save_predictions_info(self, csv_filename: str, fieldnames: List[str], directory: Path, metadata,
                              prediction: List[int]) -> None:
        with open(csv_filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for idx in range(self.batch_size):
                image_path = metadata['image_path'][idx]
                pred = prediction[idx]
                row = {'image_path': image_path, 'prediction': pred}
                writer.writerow(row)

    def evaluate(self, test_loader: DataLoader) -> dict:
        assert self.mode == EstimatorMode.EVAL, f"current mode is {self.mode}"

        self.setup(loader=test_loader)
        val_metrics = self._evaluate(test_loader, epoch=None)

        return val_metrics

    @torch.no_grad()
    @samba.session.profiler.event("evaluation_step")
    def evaluation_step(self, images):
        """One evaluation step

        Should this include post-processing?
        """
        eval_mode = "validation"  # TODO: Change to estimator mode
        output = self.forward_pass(images, eval_mode)
        return output

    @torch.no_grad()
    def _evaluate(self, data_loader: DataLoader, epoch: int = None) -> dict:
        eval_mode = "validation"  # TODO: Change to estimator mode
        self.model.eval()
        val_metrics = {}
        if self.use_sambaloader:
            # You need data: [data[0]] to ensure the filtered batch is list-like
            function_hook = lambda data: [data[0]]
            data_loader = SambaLoader(data_loader,
                                      self.input_names,
                                      function_hook=function_hook,
                                      return_original_batch=True)
        if self.run_benchmark:
            num_predict_steps = self.benchmark_steps
            loader_iter = iter(cycle(data_loader))
        else:
            num_predict_steps = len(data_loader)
            loader_iter = iter(data_loader)
        if dist.is_initialized():
            dist.barrier()

        accuracy_metrics = []
        for _ in range(num_predict_steps):
            with samba.session.profiler.event("evaluate_batch"):
                images, target, metadata = self.fetch_data(loader_iter)
                output = self.evaluation_step(images)

            loss = self.criterion(output, target)
            # introduce all gather here
            if self.use_distributed_val and dist.is_initialized():
                target = torch.cat(all_gather(target, device=self.device), dim=0)
                output = torch.cat(all_gather(output, device=self.device), dim=0)
                dist.all_reduce(loss)
                loss = (loss / dist.get_world_size())

            if not self.run_benchmark:
                metrics = self.compute_log_metrics(eval_mode, output, target, loss)
                metrics["prediction"] = output
                metrics["target"] = target
                accuracy_metrics.append(metrics)

        if not self.run_benchmark:

            val_metrics = self.aggregate_test_metrics(accuracy_metrics)
            val_metrics["is_best_ckpt"] = False
            if epoch is not None:
                val_metrics["epoch"] = epoch

            self.logger.add_metrics(val_metrics, step=self._global_step, prefix=eval_mode)

        profile_keys = ['data_step', 'session_forward', 'evaluation_step', 'evaluate_batch']
        self.logger.log_line(f"reporting statistics over {num_predict_steps} steps:")
        stats = samba.session.profiler.print_statistics(profile_keys,
                                                        self.benchmark_warmup_steps,
                                                        num_samples_per_iter=self.num_samples_per_iter)
        self.logger.log_line(stats)
        return val_metrics

    def get_hyperparams(self, epoch: int) -> float:
        """
        Helper function for LR scheduler. Use static WD and Momentum.
        """
        if self.schedulers is not None:
            return {"lr": self.schedulers[0].get_last_lr()[0]}
        return {"lr": self.learning_rate}

    def get_drop_hyperparam_dict(self, mode: str, direction: str = 'forward') -> float:
        """
        Helper function for getting hyperparams for dropout nodes.
        """
        if direction not in ['forward', 'backward']:
            raise ValueError(f"direction must be 'forward' or 'backward', got {direction}")

        if mode != EstimatorMode.TRAIN:
            return {"p": 0}

        # Create the hyperparam dict only once.
        if self.dropout_hyperparam_dict is None:
            dropout_hyperparam = {}
            dropout_argins = samba.session.get_dropout_rate_argin_names()

            for argin_name in dropout_argins:
                if "dropout2d" in argin_name:
                    dropout_hyperparam[argin_name] = self.drop_conv if mode == EstimatorMode.TRAIN else 0
                else:
                    dropout_hyperparam[argin_name] = self.drop_fc if mode == EstimatorMode.TRAIN else 0

            self.logger.log_line(f"Created dropout hyperparam dict: {dropout_hyperparam}")
            self.dropout_hyperparam_dict = dropout_hyperparam
        return self.dropout_hyperparam_dict

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        The main function which controls training or prediction.
        """
        # Initialize training/prediction.
        self.setup(loader=train_loader)

        self.logger.log_line(f"Running training")

        # We need to initialize train and validation metrics because some of the mock tests do not run a train_step
        # This happens when the requested number of epochs is < than the initialized resume epochs
        # TODO: is to instantiate mock args in a way that will account for this
        train_metrics = {}
        val_metrics = {}

        train_sampler = train_loader.sampler
        if self.use_sambaloader:
            # You need data: [data[0]] to ensure the filtered batch is list-like
            function_hook = lambda data: [data[0]]
            train_loader = SambaLoader(train_loader,
                                       self.input_names,
                                       function_hook=function_hook,
                                       return_original_batch=True)

        start_epoch = self._global_step // len(train_loader)
        for epoch in tqdm(range(start_epoch, self.epochs)):
            train_metrics = self.train_epoch(train_loader, epoch, train_sampler)
            val_metrics = self._evaluate(val_loader, epoch)
            self.save_checkpoint(epoch)

            # Save the best validation checkpoint.
            if not self.run_benchmark and val_metrics["accuracy"] > self.best_val:
                self.best_val = val_metrics["accuracy"]
                self.save_checkpoint(epoch, is_best_ckpt=True, best_metric=self.best_val)
                val_metrics["is_best_ckpt"] = True

        self.logger.log_line("Benchmark Complete.")

        profile_keys = ["train_epoch"]
        self.logger.log_line(f"reporting statistics over {self.epochs} epochs:")
        stats = samba.session.profiler.print_statistics(profile_keys, measurements=['latency'])
        self.logger.log_line(stats)
        return train_metrics, val_metrics

    def get_log_filepath(self) -> str:
        """
        Returns the path to log file.
        """
        if self.log_dir is None:
            return None

        return os.path.join(self.log_dir, f"logs_{self.run_tag}_rank_{self.rank}.txt")

    def preprocess_data(self, sample: Tuple[torch.Tensor], mode="train") -> torch.Tensor:
        """
        Helper function to preprocess the data from the dataloader before training.
        Incase application needs a different preprocessing override the function.

        If the dataloader is a Torch dataloader, it collects B samples into a batch and collates them into a larger sample:
        sample = (images,target,metadata)

        If the dataloader is a Sambaloader, it returns a tuple of (samba_sample,sample), where sample is
        from the Torch dataloader, and samba_sample is the same tuple, just with RDU tensors.

        Thus, the expanded Sambaloader tuple is ((rdu_images,rdu_target,rdu_metadata),(images,target,metadata)).
        """
        if self.device == "RDU":
            if self.use_sambaloader:
                images, target, metadata = sample[0][0], sample[1][1], sample[1][2]
            else:
                images, target, metadata = sample
                images = samba.from_torch_tensor(images, name="input", batch_dim=0)
            images.requires_grad = not self.inference
        elif self.device == "GPU":
            images, target, metadata = sample
            images = images.cuda()
            if target is not None:
                target = target.cuda()
        else:
            images, target, metadata = sample

        return images, target, metadata

    def forward_pass(self, input_tensors: Tuple[SambaTensor], mode: str) -> torch.Tensor:
        if self.device == "RDU":
            output = self._forward_rdu(input_tensors, mode)
        else:
            output = self._forward_gpu(input_tensors)

        return output

    def _forward_rdu(self, input_tensors: Tuple[SambaTensor], mode: str) -> torch.Tensor:
        output_tensors = self.model.output_tensors
        hyperparam_dict = self.get_drop_hyperparam_dict(mode=mode, direction='forward')

        with samba.session.profiler.event("session_forward"):
            output = samba.session.run(input_tensors=(input_tensors, ),
                                       output_tensors=output_tensors,
                                       section_types=["fwd"],
                                       hyperparam_dict=hyperparam_dict,
                                       data_parallel=self.data_parallel,
                                       reduce_on_rdu=self.reduce_on_rdu)[0]

        # Convert back to float to compute loss on host.
        output = samba.to_torch(output.float())
        output.requires_grad = not self.inference
        return output

    def _forward_gpu(self, input_tensors: Tuple[SambaTensor]) -> torch.Tensor:
        return self.model(input_tensors)

    def backward_pass(self, input: torch.Tensor, loss: torch.Tensor, output: torch.Tensor, mode: str, epoch: int):
        """
        Computes back gradient and optimizer step.
        """
        if self.device == "RDU":
            self._backward_rdu(input, loss, output, mode, epoch)
        else:
            self._backward_gpu(loss)

        if self.schedulers is not None:
            for sched in self.schedulers:
                sched.step()

    def _backward_rdu(self, input: torch.Tensor, loss: torch.Tensor, output: torch.Tensor, mode: str, epoch: int):
        # Compute the gradients.
        loss.backward()

        # Compute samba backward + optimizer step
        hyperparam_dict = self.get_hyperparams(epoch)
        hyperparam_dict.update(self.get_drop_hyperparam_dict(mode=mode, direction='backward'))

        with samba.session.profiler.event("session_backward"):
            self.model.output_tensors[0].sn_grad = output.grad
            samba.session.run(input_tensors=(input, ),
                              output_tensors=self.model.output_tensors,
                              section_types=['bckwd', 'opt'],
                              hyperparam_dict=hyperparam_dict,
                              data_parallel=self.data_parallel,
                              reduce_on_rdu=self.reduce_on_rdu)

    def _backward_gpu(self, loss):
        loss.backward()

        # Run Optimizer Step.
        for opt in self.optimizers:
            opt.step()

        # Zero the gradients.
        for opt in self.optimizers:
            opt.zero_grad()

    @samba.session.profiler.event("data_step")
    def fetch_data(self, loader):
        sample = next(loader)
        return self.preprocess_data(sample)

    @samba.session.profiler.event("train_step")
    def training_step(self, images, target, epoch):
        with interrupter.context(self.save_checkpoint, epoch=epoch):
            output = self.forward_pass(images, mode=EstimatorMode.TRAIN)
            loss = self.criterion(output, target)
            # Get LR before scheduler step in backward_pass
            self.backward_pass(images, loss, output, "train", epoch)
        return output, loss

    @samba.session.profiler.event("train_epoch")
    def train_epoch(self, train_loader: DataLoader, epoch: int, sampler: DistributedSampler):
        """
        Train for one epoch.
        """
        self.model.train()

        if dist.is_initialized():
            sampler.set_epoch(epoch)
            dist.barrier()

        if self.run_benchmark:
            num_train_steps = self.benchmark_steps
            train_iter = iter(cycle(train_loader))
        else:
            num_train_steps = len(train_loader)
            train_iter = iter(train_loader)

        self.logger.log_line(f'Training Epoch: {epoch}')
        if self.schedulers is not None:
            self.logger.log_line('\n'.join([str(sched) for sched in self.schedulers]))
        for i in range(num_train_steps):
            start_time = time.time()

            with samba.session.profiler.event("train_batch"):
                images, target, _ = self.fetch_data(train_iter)
                output, loss = self.training_step(images, target, epoch=epoch)

            # measure accuracy and record loss
            log_metrics = {'lr': self.get_hyperparams(epoch)['lr']}
            log_metrics.update(self.compute_log_metrics("train", output, target, loss))
            log_metrics["epoch"] = epoch

            end_time = time.time()
            log_metrics["time_per_step"] = end_time - start_time

            if i % self.print_freq == 0:
                self.logger.add_metrics(log_metrics, step=self._global_step, prefix="train")

            self._global_step += 1

        profile_keys = ["data_step", "session_forward", "session_backward", "train_step", "train_batch"]
        self.logger.log_line(f"reporting statistics over {num_train_steps} steps:")
        stats = samba.session.profiler.print_statistics(profile_keys,
                                                        self.benchmark_warmup_steps,
                                                        num_samples_per_iter=self.num_samples_per_iter)
        self.logger.log_line(stats)
        return log_metrics

    def load_checkpoint(self, is_best_ckpt: bool = False, ckpt_file: Optional[str] = None):
        '''
        Load model state from an existing checkpoint.
        '''

        strict = False if self.num_flexible_classes != -1 else True  # allow flexible classes to read in ImageNet pre-trained weights

        # TODO: (davidku) Change checkpoint to include optimizer state dict
        optimizer_load_failure_ok = True

        if ckpt_file:
            with self.logger.capture_stdout():
                result = load_checkpoint_from_file(ckpt_file,
                                                   self.model,
                                                   optimizers=self.optimizers if self.resume else None,
                                                   strict=strict,
                                                   optimizer_load_failure_ok=optimizer_load_failure_ok,
                                                   state_dict_filter_fn=_ckpt_dict_filter_fn)
        elif self.checkpointer is None:
            self.logger.log_line('ckpt_dir not provided, not loading from a checkpoint')
            return
        else:
            with self.logger.capture_stdout():
                result = self.checkpointer.load_checkpoint(self.model,
                                                           optimizers=self.optimizers if self.resume else None,
                                                           is_best_ckpt=is_best_ckpt,
                                                           strict=strict,
                                                           optimizer_load_failure_ok=optimizer_load_failure_ok,
                                                           state_dict_filter_fn=_ckpt_dict_filter_fn)

        _, _, _, global_step, _, best_metric = result

        if self.resume or self.mode == EstimatorMode.PREDICT:
            self._global_step = global_step
            self.best_val = best_metric

        # Update LR if it's changed from ckpt
        if self.optimizers:
            for opt in self.optimizers:
                for param_group in opt.param_groups:
                    param_group['lr'] = self.learning_rate

        if self.schedulers is not None:
            for sched in self.schedulers:
                sched.base_lr = self.learning_rate
                sched.current_iter = self._global_step + 1

        # If using GPUs, move the model to GPU:
        if self.device == "GPU":
            self.model.cuda()

        # Model weights will be synced to RDU via graph tracing
        return

    def save_checkpoint(self, epoch: int, is_best_ckpt: bool = False, best_metric: float = 0.):
        '''
        Save the model to file.
        '''
        if self.rank != 0:
            # Save only for rank0.
            return

        if self.checkpointer is None:
            self.logger.log_line('ckpt_dir not provided, not loading from a checkpoint')
            return

        if self.device == "RDU":
            if self.optimizer_type == 'adamw':
                samba.session.to_cpu(self.model, self.optimizers)
            else:
                samba.session.to_cpu(self.model)

        with self.logger.capture_stdout():
            self.checkpointer.save_checkpoint(self.model,
                                              self.optimizers,
                                              step=self._global_step,
                                              epoch=epoch,
                                              is_best_ckpt=is_best_ckpt,
                                              best_metric=best_metric)
