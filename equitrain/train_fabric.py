import pytorch_lightning as pl
from torch_geometric.utils import degree
from accelerate import DistributedDataParallelKwargs, Accelerator
import torch
import math
import logging
import numpy as np

from typing  import Optional
from pathlib import Path

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from timm.optim.adabelief import AdaBelief
from timm.scheduler import create_scheduler

from equitrain.dataloaders   import get_dataloaders
from equitrain.model         import get_model

def log_metrics(args, logger, prefix, postfix, loss_metrics):

    info_str  = prefix
    info_str += 'loss: {loss:.5f}'.format(loss=loss_metrics['total'].avg)

    if args.energy_weight > 0.0:
        info_str += ', loss_e: {loss_e:.5f}'.format(
            loss_e=loss_metrics['energy'].avg,
        )
    if args.force_weight > 0.0:
        info_str += ', loss_f: {loss_f:.5f}'.format(
            loss_f=loss_metrics['forces'].avg,
        )
    if args.stress_weight > 0.0:
        info_str += ', loss_s: {loss_f:.5f}'.format(
            loss_f=loss_metrics['stress'].avg,
        )

    if postfix is not None:
        info_str += postfix

    logger.info(info_str)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (name.endswith(".bias") or name.endswith(".affine_weight")  
            or name.endswith(".affine_bias") or name.endswith('.mean_shift')
            or 'bias.' in name 
            or name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def compute_weighted_loss(args, energy_loss, force_loss, stress_loss):
    result = 0.0
    # handle initial values correctly when weights are zero, i.e. 0.0*Inf -> NaN
    if energy_loss is not None and (not math.isinf(energy_loss) or args.energy_weight > 0.0):
        result += args.energy_weight * energy_loss
    if force_loss is not None and (not math.isinf(force_loss) or args.force_weight > 0.0):
        result += args.force_weight * force_loss
    if stress_loss is not None and (not math.isinf(stress_loss) or args.stress_weight > 0.0):
        result += args.stress_weight * stress_loss

    return result

def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        optimizer_name=cfg.opt,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs

def create_optimizer(args, model, filter_bias_and_bn=True):
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )


def create_optimizer_v2(
        model: torch.nn.Module,
        optimizer_name: str = 'sgd',
        lr: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        filter_bias_and_bn: bool = True,
        **kwargs):
    """ Create an optimizer.

    TODO currently the model is passed in and all parameters are selected for optimization.
    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller

    Args:
        model (nn.Module): model containing parameters to optimize
        optimizer_name: name of optimizer to create
        lr: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through

    Returns:
        Optimizer
    """
    opt_lower = optimizer_name.lower()
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()
    #if 'fused' in opt_lower:
    #    assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=lr, weight_decay=weight_decay, **kwargs)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = torch.optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = torch.optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = torch.optim.Adam(parameters, **opt_args) 
    elif opt_lower == 'adabelief':
        optimizer = AdaBelief(parameters, rectify=False, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = torch.optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = torch.optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=momentum, **opt_args)
    else:
        assert False and "Invalid optimizer"

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer

class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op

class FileLogger:
    def __init__(self, is_master=False, is_rank0=False, output_dir=None, logger_name='training', version='1'):
        # only call by master 
        # checked outside the class
        self.output_dir = output_dir
        self.save_dir = output_dir
        self.name = logger_name
        self.version = version

        if is_rank0:
            self.logger_name = logger_name
            self.logger = self.get_logger(output_dir, log_to_file=is_master)
        else:
            self.logger_name = None
            self.logger = NoOp()
        
        
    def get_logger(self, output_dir, log_to_file):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        if output_dir and log_to_file:
            
            time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
            debuglog = logging.FileHandler(output_dir+'/debug.log')
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(time_formatter)
            logger.addHandler(debuglog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
        
        logger.propagate = False

        return logger
    
    def console(self, *args):
        self.logger.debug(*args)

    def event(self, *args):
        self.logger.warn(*args)

    def verbose(self, *args):
        self.logger.info(*args)

    def info(self, *args):
        self.logger.info(*args)
    
    def finalize(self, status):
        self.logger.info(f"Training finalized with status: {status}")
    
    def log_graph(self, model):
        pass
    
    def save(self):
        pass
    
    def log_metrics(self, metrics, step=None):
        for key, value in metrics.items():
            if step is not None:
                self.logger.info(f"{key}: {value} at step {step}")
            else:
                self.logger.info(f"{key}: {value}")

    def after_save_checkpoint(self, checkpoint):
        self.logger.info(f"Checkpoint saved at {checkpoint}")



class EquiTrainModule(pl.LightningModule):
    def __init__(self, args, model, criterion, logger=None):
        super(EquiTrainModule, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.custom_logger = logger
        self.loss_metrics = {
            'total': AverageMeter(),
            'energy': AverageMeter(),
            'forces': AverageMeter(),
            'stress': AverageMeter(),
        }

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        e_true = batch.y
        f_true = batch['force']
        s_true = batch['stress']

        e_pred, f_pred, s_pred = self(batch)

        loss_e, loss_f, loss_s = None, None, None
        if self.args.energy_weight > 0.0:
            loss_e = self.criterion(e_pred, e_true)
        if self.args.force_weight > 0.0:
            loss_f = self.criterion(f_pred, f_true)
        if self.args.stress_weight > 0.0:
            loss_s = self.criterion(s_pred, s_true)

        loss = compute_weighted_loss(self.args, loss_e, loss_f, loss_s)
        if torch.isnan(loss):
            self.logger.info(f'NaN value detected. Skipping batch...')
            return None

        # Log the metrics
        self.loss_metrics['total'].update(loss.item(), n=e_pred.shape[0])
        if self.args.energy_weight > 0.0:
            self.loss_metrics['energy'].update(loss_e.item(), n=e_pred.shape[0])
        if self.args.force_weight > 0.0:
            self.loss_metrics['forces'].update(loss_f.item(), n=f_pred.shape[0])
        if self.args.stress_weight > 0.0:
            self.loss_metrics['stress'].update(loss_s.item(), n=s_pred.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        pred_e, pred_f, pred_s = self(batch)

        loss_e, loss_f, loss_s = None, None, None
        if pred_e is not None:
            loss_e = self.criterion(pred_e, batch.y)
        if pred_f is not None:
            loss_f = self.criterion(pred_f, batch['force'])
        if pred_s is not None:
            loss_s = self.criterion(pred_s, batch['stress'])

        loss = compute_weighted_loss(self.args, loss_e, loss_f, loss_s)

        self.loss_metrics['total'].update(loss.item(), n=pred_e.shape[0])
        if pred_e is not None:
            self.loss_metrics['energy'].update(loss_e.item(), n=pred_e.shape[0])
        if pred_f is not None:
            self.loss_metrics['forces'].update(loss_f.item(), n=pred_f.shape[0])
        if pred_s is not None:
            self.loss_metrics['stress'].update(loss_s.item(), n=pred_s.shape[0])

        # Log validation loss (this will be used by ModelCheckpoint)
        self.log('val_loss', loss, prog_bar=True, batch_size=pred_e.shape[0])

        if loss_e is not None:
            self.log('loss_e', loss_e)
        if loss_f is not None:
            self.log('loss_f', loss_f)
        if loss_s is not None:
            self.log('loss_s', loss_s)

        return loss
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.args, self.model)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.args.patience_epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'val_loss',  # The metric to monitor for PlateauLR
                'interval': 'epoch',
                'frequency': 1,
                'reduce_on_plateau': True  # Tell Lightning it's a PlateauLR type scheduler
            }
        }


    def on_train_epoch_end(self):
        # Use the custom logger to log metrics at the end of the training epoch
        log_metrics(self.args, self.custom_logger, f"Epoch [{self.current_epoch}] Train -- ", None, self.loss_metrics)
        self.reset_loss_metrics()

    def on_validation_epoch_end(self):
        # Use the custom logger to log metrics at the end of the validation epoch
        log_metrics(self.args, self.custom_logger, f"Epoch [{self.current_epoch}] Val -- ", None, self.loss_metrics)
        self.reset_loss_metrics()

    def reset_loss_metrics(self):
        for meter in self.loss_metrics.values():
            meter.reset()

    #TEsting commits


def _train(args):
    logger = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    logger.info(args)

    device = torch.device("cuda:1")
    torch.set_float32_matmul_precision('medium')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Distributed Data Parallel options
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) if args.energy_weight == 0.0 else DistributedDataParallelKwargs()

    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    ''' Data Loader '''
    train_loader, val_loader, test_loader, r_max = get_dataloaders(args, logger=logger)
    
    ''' Network '''
    model = get_model(r_max, args, compute_force=args.force_weight > 0.0, compute_stress=args.stress_weight > 0.0, logger=logger)
    
    # Create Criterion
    criterion = torch.nn.L1Loss()

    ''' Lightning Module '''
    lightning_model = EquiTrainModule(args, model, criterion, logger)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',  # This is the key you are logging
        save_top_k=1,  # Save the best checkpoint
        mode='min',  # We're minimizing the loss
        filename='best-checkpoint'
    )

    ''' PyTorch Lightning Trainer '''
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator='gpu',
        precision=16 if args.mixed_precision else 32,
        log_every_n_steps=100,
        logger=logger,  # Integrate your custom logger
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_callback]
    )

    ''' Start Training '''
    trainer.fit(lightning_model, train_loader, val_loader)

    # Test on the test dataset
    if test_loader is not None:
        trainer.test(lightning_model, test_loader)


def train_fabric(args):

    if args.train_file is None:
        raise ValueError("--train-file is a required argument")
    if args.valid_file is None:
        raise ValueError("--valid-file is a required argument")
    if args.statistics_file is None:
        raise ValueError("--statistics-file is a required argument")
    if args.output_dir is None:
        raise ValueError("--output-dir is a required argument")

    if args.energy_weight == 0.0 and args.force_weight == 0.0 and args.stress_weight == 0.0:
        raise ValueError("at least one non-zero loss weight is required")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _train(args)