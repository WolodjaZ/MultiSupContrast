import os
import argparse

import time
import math
import wandb
import torch
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast

from losses import AsymmetricLoss
from datasets import CocoDetection, MultiLabelCelebA, VOCDataset
from networks.utils import create_model_base, add_classification_head
from utils import init_distributed_mode, fix_random_seeds, mAP, \
    initialize_exp, AverageMeter, adjust_learning_rate, warmup_learning_rate


def parse_option():
    parser = argparse.ArgumentParser(description='PyTorch Multi supervised contrastive evaluation')
    
    #############################
    # data and model parameters #
    #############################
    parser.add_argument('--data', type=str, default='/home/MSCOCO_2014/')
    parser.add_argument('--data-name', type=str, default='COCO')
    parser.add_argument('--model-name', default='tresnet_l')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--num-classes', type=int, default=80)
    parser.add_argument('--image-size', default=448, type=int,
                        metavar='N', help='input image size (default: 448)')
        
    ###############################
    ####### optim parameters ######
    ###############################
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    
    ###############################
    ####### dist parameters #######
    ###############################
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int, help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")
    
    ###############################
    ####### other parameters ######
    ###############################
    parser.add_argument('--method', type=str, default='CrossEntropy',
                        choices=['CrossEntropy', 'Asymetric'], help='choose method')
    parser.add_argument('--freeze', default=True, type=bool,
                        metavar='N', help='freeze backbone')
    parser.add_argument('--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--sync_bn', type=bool, default=False,
                        help='synchronic batch only with distributed gpu')
    parser.add_argument('--feat-dim', type=int, default=128,
                        help='feature dimension for contrastive learning')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--method_used', type=str, default='MultiSupCon',
                        choices=['MultiSupCon', 'SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                    help="Save the model periodically")
    parser.add_argument('--vis_3d', default=True, type=bool,
                        metavar='N', help='Visualize in 3d')
    parser.add_argument('--run', default=0, type=int,
                        metavar='N', help='run number')
    parser.add_argument("--dump_path", type=str, default="./experiment_eval",
                    help="experiment dump path for checkpoints and log")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    
    return parser


def main():
    # Prepering environment
    args = parse_option().parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger = initialize_exp(args, "epoch", "loss")
    
    # Build data
    if "COCO" in args.data_name:
        # COCO Data loading
        instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
        data_path_val = f'{args.data}/val2014'
        data_path_train = f'{args.data}/train2014'
        
        if args.data_name == "COCO":
            train_dataset = CocoDetection(
                data_path_train,
                instances_path_train,
                transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor()
                ])
            )
        elif args.data_name == "COCOCrop":
            train_dataset = CocoDetection(
                data_path_train,
                instances_path_train,
                transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor()
                ]),
                boxcrop=args.image_size
            )
        val_dataset = CocoDetection(
            data_path_val,
            instances_path_val,
            transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor()
            ])
        )
    elif "VOC" in args.data_name:
        if args.data_name == "VOC":
            train_dataset = VOCDataset(
                args.data,
                transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor()
                                ]),
                val=False
            )
        elif args.data_name == "VOCrop":
            train_dataset = VOCDataset(
                args.data,
                transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor()
                                ]),
                val=False,
                boxcrop=args.image_size
            )
        val_dataset = VOCDataset(
            args.data,
            transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor()
            ]),
            val=True
        )
    elif "CELEBA" in args.data_name:
        if args.data_name == "CELEBA":
            train_dataset = MultiLabelCelebA(
                args.data,
                split="train",
                transform=transforms.Compose([
                                transforms.RandomResizedCrop(size=args.image_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor(),
                                ])
            )
        val_dataset = MultiLabelCelebA(
            args.data,
            split="valid",
            transform=transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor()
            ]),
        )
    train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(500))
    val_dataset = torch.utils.data.Subset(val_dataset, torch.arange(500))
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(sampler is None),
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=128,
        num_workers=8
    )
    logger.info("Building data done with train {} images loaded and val {} images loaded.".format(len(train_dataset), len(val_dataset)))
    # Build model
    model = create_model_base(args)
    if args.model_path:
        # Loading model
        checkpoint = torch.load(
            wandb.restore(args.model_path),
            map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count()))
        model.load_state_dict(checkpoint['model_state_dict'])
        # freeze
        if args.freeze:
            for param in model.parameters():
                param.requires_grad = False
    # Adding classification head
    model = add_classification_head(model, args.num_classes)
    # Synchronize batch norm layers
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Copy model to GPU
    model = model.cuda()
    cudnn.benchmark = True
    logger.info(f'Building model {args.model_name} done')
    
    # Build optimizer and criterion
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
        
    # Warm-up for large-batch training,
    if args.batch_size > 256:
        args.warm = True
    if args.warm:
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    
    if args.method == "CrossEntropy":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.method == "Asymetric":
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    logger.info("Building optimizer and criterion done.")
    
    # wrap model
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )
    
    # Check for the checkpoints
    log_path = os.path.join(args.dump_path, f"run_{args.run}")
    if os.path.isdir(log_path):
        resume = True
    else:
        resume = False
        os.makedirs(log_path, exist_ok=True)
    
    # Log wandb
    if args.rank == 0:
        wandb.login(key="c77809672cac9c98eb589447ff82854fba590ff7")
        if resume:
            wandb.init(
                project="test-project", 
                entity="pwr-multisupcontr",
                name=f"validating_linear_multi_sup_con_{args.run}",
                resume=True 
            )
        else:
            wandb.init(
                project="test-project", 
                entity="pwr-multisupcontr",
                name=f"validating_multi_sup_con_{args.run}",
                config={
                    "data": args.data,
                    "image-size": args.image_size,
                    "batch-size": args.batch_size,
                    "epochs": args.epochs_con,
                    "learning_rate": args.learning_rate,
                    "lr_decay_epochs": args.lr_decay_epochs,
                    "lr_decay_rate": args.lr_decay_rate,
                    "weight_decay": args.weight_decay,
                    "momentum": args.momentum,
                    "method": args.method,
                    "method_used": args.method_used,
                    "cosine": args.cosine,
                    "warm": args.warm,
                    "seed": args.seed,
                    "freeze": args.freeze,
                    "sync_bn:": args.sync_bn
                }
            )
        wandb.watch(model, log="all")
    
    # Load checkpoint
    if resume:
        # Get last restore
        checkpoint_last = os.path.join(log_path, "last_checkpoint.pt")
        checkpoint_best = os.path.join(log_path, "best_checkpoint.pt")
        checkpoint = torch.load(
            wandb.restore(checkpoint_last),
            map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count()))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best = torch.load(
            wandb.restore(checkpoint_best),
            map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count()))
    else:
        start_epoch = 1
        best = {}
    
    ###############################
    ########### TRAINING ##########
    ###############################
    best = {}
    for epoch in range(start_epoch, args.epochs+1):
        
        # train the network for one epoch
        logger.info(f"============ Starting epoch {epoch} ... ============")
        
        # set sampler
        train_loader.sampler.set_epoch(epoch)
        
        # Adjuest learning rate
        adjust_learning_rate(args, optimizer, epoch)
        # train the network
        scores = train(
            train_loader,
            model,
            optimizer,
            criterion,
            epoch,
            logger,
            args
        )
        
        # save checkpoints
        if args.rank == 0:
            # Validate
            val_map = validate(val_loader, model)
            logger.info(f"Validate: Epoch [{epoch}], Mean Average Precision: {val_map:.3f}")
            # Log to wandb metrics
            wandb.log({
                "loss": scores[1],
                "map": scores[2],
                "learning_rate": optimizer.optim.param_groups[0]["lr"],
                "val_map": val_map,
            }, step=scores[0])
            # Update best loss
            if "map" not in best.keys():
                best = { 
                    'epoch': scores[0],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': scores[1],
                    'map': val_map,
                }
                # Save best loss
                checkpoint_path = os.path.join(log_path, f"best_checkpoint.pth.tar")
                torch.save(best, checkpoint_path)
                wandb.save(checkpoint_path)
            else:
                if val_map > best["map"]:
                    best = { 
                        'epoch': scores[0],
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': scores[1],
                        'map': val_map,
                    }
                    # Save best loss
                    checkpoint_path = os.path.join(log_path, f"best_checkpoint.pt")
                    torch.save(best, checkpoint_path)
                    wandb.save(checkpoint_path)
            # Save our checkpoint loc
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs:
                checkpoint_path = os.path.join(log_path, f"{epoch}_checkpoint.pt")
                checkpoint_last = os.path.join(log_path, "last_checkpoint.pt")
                torch.save({ 
                    'epoch': scores[0],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': scores[1],
                    'map': val_map,
                    }, checkpoint_path)
                wandb.save(checkpoint_path)
                torch.save({ 
                    'epoch': scores[0],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': scores[1],
                    'map': val_map,
                    }, checkpoint_last)
                wandb.save(checkpoint_last)

    if args.rank == 0:
        # End wandb
        wandb.finish()

    ###############################
    ########### Finished ##########
    ###############################
    logger.info("============ Finished ============")
    logger.info(f"Best Mean Average Precision:: {best['map']} with loss {best['loss']} on epoch {best['epoch']}")


def train(train_loader, model, optimizer, criterion, epoch, logger, args):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mAPs = AverageMeter()
    
    end = time.time()
    scaler = GradScaler()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        labels = labels.max(dim=1)[0]
        bsz = labels.shape[0]
        
        # Warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        
        # Compute loss
        with autocast():  # mixed precision
            output = model(images).float()
        loss = criterion(output, labels)
        # update metric
        losses.update(loss.item(), bsz)
        mAP_score = mAP(labels.cpu().detach().numpy(), output.cpu().detach().numpy())
        mAPs.update(mAP_score, bsz)
        
        # Optimizer
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if args.rank == 0 and idx % 50 == 0:
            logger.info((
                f'Train: Epoch [{epoch}], Step [{idx}/{len(train_loader)}], '
                f'Loss: {loss.item():.3f}, Mean Average Precision: {mAP_score:.3f}, '
                f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})'
            ))
    return (epoch, losses.avg, mAPs.avg)

def validate(val_loader, model):
    model.eval()
    
    labels_all = []
    outputs = []
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            labels = labels.max(dim=1)[0]
            if torch.cuda.is_available():
                images = images.cuda()
            # Gather results
            with autocast():
                outputs.append(model(images).cpu().detach())
            labels_all.append(labels.cpu().detach())
    
    #Calc MAP
    mAP_score = mAP(torch.cat(labels_all).numpy(), torch.cat(outputs).numpy())
    return mAP_score


if __name__ == '__main__':
    main()
