import os
import argparse

import time
import umap
import math
import wandb
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast

from losses import MultiSupConLoss
from networks.utils import create_model_base
from utils import init_distributed_mode, fix_random_seeds, initialize_exp, \
    AverageMeter, adjust_learning_rate, warmup_learning_rate
from datasets import TwoCropTransform, CocoDetection, MultiLabelCelebA, \
    VOCDataset, MultiLabelNUS


def parse_option():
    parser = argparse.ArgumentParser(description='PyTorch Multi supervised contrastive training')
    
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
    # MultiSupCon specific params #
    ###############################
    parser.add_argument('--method', type=str, default='MultiSupCon',
                        choices=['MultiSupCon', 'SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--feat-dim', type=int, default=128,
                        help='feature dimension for contrastive learning')
    parser.add_argument('--c_treshold', type=float, default=0.3,
                        help='Jaccard sim split parameter')
    
    ###############################
    ####### optim parameters ######
    ###############################
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
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
    parser.add_argument('--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--sync_bn', type=bool, default=False,
                        help='synchronic batch only with distributed gpu')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                    help="Save the model periodically")
    parser.add_argument('--vis_3d', default=True, type=bool,
                        metavar='N', help='Visualize in 3d')
    parser.add_argument('--run', default=0, type=int,
                        metavar='N', help='run number')
    parser.add_argument("--dump_path", type=str, default="./experiment",
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
        instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
        data_path_train = f'{args.data}/train2014'
        
        if args.data_name == "COCO":
            train_dataset = CocoDetection(
                data_path_train,
                instances_path_train,
                TwoCropTransform(transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor()
                ]))
            )
        elif args.data_name == "COCOCrop":
            train_dataset = CocoDetection(
                data_path_train,
                instances_path_train,
                TwoCropTransform(transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor()
                ])),
                boxcrop=args.image_size
            )
    elif "VOC" in args.data_name:
        if args.data_name == "VOC":
            train_dataset = VOCDataset(
                args.data,
                TwoCropTransform(transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor()
                                ])),
                val=False
            )
        elif args.data_name == "VOCrop":
            train_dataset = VOCDataset(
                args.data,
                TwoCropTransform(transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor()
                                ])),
                val=False,
                boxcrop=args.image_size
            )
    elif "NUS" in args.data_name:
        if args.data_name == "NUS":
            train_dataset = MultiLabelNUS(
                args.data,
                split="train",
                transform=transforms.Compose([
                                transforms.RandomResizedCrop(size=args.image_size, scale=(0.6, 1.)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor(),
                                ]),
            )
    elif "CELEBA" in args.data_name:
        if args.data_name == "CELEBA":
            train_dataset = MultiLabelCelebA(
                args.data,
                split="train",
                transform=TwoCropTransform(transforms.Compose([
                                transforms.RandomResizedCrop(size=args.image_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor(),
                                ]))
            )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(sampler is None),
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))
    
    # Build model
    model = create_model_base(args)
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
    
    criterion = MultiSupConLoss(temperature=args.temp, c_treshold=args.c_treshold)
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(),
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
        wandb.login()
        if resume:
            wandb.init(
                project="test-project", 
                entity="pwr-multisupcontr",
                name=f"training_multi_sup_con_{args.run}",
                resume=True 
            )
        else:
            wandb.init(
                project="test-project", 
                entity="pwr-multisupcontr",
                name=f"training_multi_sup_con_{args.run}",
                config={
                    "data": args.data,
                    "image-size": args.image_size,
                    "batch-size": args.batch_size,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "lr_decay_epochs": args.lr_decay_epochs,
                    "lr_decay_rate": args.lr_decay_rate,
                    "weight_decay": args.weight_decay,
                    "momentum": args.momentum,
                    "method": args.method,
                    "temp": args.temp,
                    "cosine": args.cosine,
                    "warm": args.warm,
                    "feat-dim": args.feat_dim,
                    "c_treshold": args.c_treshold,
                    "seed": args.seed,
                    "sync_bn": args.sync_bn,
                    "numb_of_gpu_used": args.gpu_to_work_on
                }
            )
        wandb.watch(model, log="all")
        
        # Log dataset plots
        logger.info("Logging correlation and bar plot of labels.")
        labels_all = []
        for idx, (_, labels) in enumerate(train_loader):
            labels_all.append(labels.max(dim=1)[0].cpu().detach())
        labels_all = torch.cat(labels_all).numpy()
        df_labels = pd.DataFrame(labels_all)
        fig_corelation = px.imshow(df_labels.corr())
        fig_bar = px.bar(df_labels.sum(axis=0))
        fig_bar.update_layout(
            title="Number of occurence in dataset for each class",
            xaxis_title="Class index",
            yaxis_title="Occurance in dataset",
            showlegend=False
        )
        wandb.log({
            "Barplot labels": fig_bar,
            "Correlation of labels": fig_corelation
        })
        logger.info("Finished logging correlation and bar plot of labels.")
        
    # Load checkpoint
    if resume:
        # Get last restore
        checkpoint_last = os.path.join(log_path, "last_checkpoint.pth.tar")
        checkpoint_best = os.path.join(log_path, "best_checkpoint.pth.tar")
        logger.info(f"Loading checkpooint {checkpoint_last}")
        checkpoint = torch.load(checkpoint_last,
               map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count()))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        best = torch.load(checkpoint_best,
            map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count()))
    else:
        start_epoch = 1
        best = {}
    
    ###############################
    ########### TRAINING ##########
    ###############################
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
            fig_validate = validate(train_loader, model, args.vis_3d)
            # Log to wandb metrics
            wandb.log({
                "loss": scores[1],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "umap_embeddings": fig_validate,
            }, step=epoch)
            # Update best loss
            if "loss" not in best.keys():
                best = { 
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': scores[1],
                }
                # Save best loss
                checkpoint_path = os.path.join(log_path, f"best_checkpoint.pth.tar")
                torch.save(best, checkpoint_path)
                wandb.save(checkpoint_path)
            else:
                if scores[1] < best["loss"]:
                    best = { 
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': scores[1],
                    }
                    # Save best loss
                    checkpoint_path = os.path.join(log_path, f"best_checkpoint.pth.tar")
                    torch.save(best, checkpoint_path)
                    wandb.save(checkpoint_path)
            # Save our checkpoint loc
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs:
                checkpoint_path = os.path.join(log_path, f"{epoch}_checkpoint.pth.tar")
                checkpoint_last = os.path.join(log_path, "last_checkpoint.pth.tar")
                torch.save({ 
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': scores[1],
                    }, checkpoint_path)
                torch.save({ 
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': scores[1],
                    }, checkpoint_last)
                wandb.save(checkpoint_last)

    if args.rank == 0:
        # End wandb
        wandb.run.summary["best_loss"] = best['loss']
        wandb.run.summary["best_epoch"] = best['epoch']
        wandb.finish()

    ###############################
    ########### Finished ##########
    ###############################
    logger.info("============ Finished ============")
    logger.info(f"Best loss: {best['loss']} on epoch {best['epoch']}")


def train(train_loader, model, optimizer, criterion, epoch, logger, args):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    scaler = GradScaler()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        labels = labels.max(dim=1)[0]
        bsz = labels.shape[0]
        
        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        
        # compute loss
        with autocast():  # mixed precision
            features = model(images).float()
        
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        f1 = torch.nn.functional.normalize(f1, dim=1)
        f2 = torch.nn.functional.normalize(f2, dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if args.method == 'MultiSupCon':
            loss = criterion(features, labels)
        elif args.method == 'SupCon':
            loss = criterion(features, labels, multi=False)
        elif args.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                            format(args.method))
        
        # update metric
        losses.update(loss.item(), bsz)
        
        # Optimizer
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if args.rank == 0 and idx % 50 == 0:
            logger.info((
                f'Train: Epoch [{epoch}], Step [{idx}/{len(train_loader)}], Loss: {loss.item():.3f}, '
                f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})'
            ))
    
    return (epoch, losses.avg)

def validate(train_loader, model, vis_3d=True):
    model.eval()
    outputs = []
    labels = []
    
    # Initialize umap
    if vis_3d:
        reducer = umap.UMAP(n_components=3)
    else:
        reducer = umap.UMAP(n_components=2)
    
    # Start validating
    with torch.no_grad():
        for idx, (images, label) in enumerate(train_loader):
            images = images[0]
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)

            # compute loss
            with autocast():
                outputs.append(
                    torch.nn.functional.normalize(model(images), dim=1).cpu().detach())
            labels.append(label.max(dim=1)[0].cpu().detach())
    
    # Do dimension reduction
    embedding = reducer.fit_transform(torch.cat(outputs).numpy())
    labels = torch.cat(labels)
    
    # Create text for embeding
    results = []
    for label in labels:
        result = torch.nonzero(label).reshape(-1)
        if result.shape[0] > 1:
            results.append(str(np.array2string(result.numpy(), separator=',')))
        else:
            results.append(str(result.numpy()[0]))
    
    # Create dataframe
    if vis_3d:
        df = pd.DataFrame({
            'x': embedding[:,0],
            'y': embedding[:,1],
            'z': embedding[:,2],
            'label': results,
            }
        )
        # Create scatter plot
        fig = px.scatter_3d(df, x='x', y='y', z='z', hover_data=
                dict({'x': False, 'y': False, 'z': False, 'label': True}))
        # Tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    else:
        df = pd.DataFrame({
            'x': embedding[:,0],
            'y': embedding[:,1],
            'label': result,
            }
        )
        # Create scatter plot
        fig = px.scatter(df, x='x', y='y', hover_data=
                dict({'x': False, 'y': False, 'label': True}))
        
    return fig


if __name__ == '__main__':
    main()
