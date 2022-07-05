import os
import argparse

import time
import torch
import wandb
from sklearn.metrics import f1_score
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler, Adam
from torch.cuda.amp import GradScaler, autocast

from losses import AsymmetricLoss
from datasets import CocoDetection, MultiLabelCelebA, VOCDataset, MultiLabelNUS
from networks.utils import create_model_base, add_ml_decoder_head
from utils import init_distributed_mode, fix_random_seeds, mAP, \
    initialize_exp, AverageMeter, add_weight_decay, ModelEma


def parse_option():
    parser = argparse.ArgumentParser(description='PyTorch Multi supervised contrastive evaluation')
    
    #############################
    # data and model parameters #
    #############################
    parser.add_argument('--data', type=str, default='/home/MSCOCO_2014/')
    parser.add_argument('--data-name', type=str, default='COCO')
    parser.add_argument('--model-name', default='tresnet_l')
    parser.add_argument('--model-path', default=None)
        #'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth', None])
    parser.add_argument('--num-classes', type=int, default=80)
    parser.add_argument('--image-size', default=448, type=int,
                        metavar='N', help='input image size (default: 448)')
    
    ###############################
    #### ML_Decoder parameters ####
    ###############################
    parser.add_argument('--use-ml-decoder', default=1, type=int)
    parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
    parser.add_argument('--decoder-embedding', default=768, type=int)
    parser.add_argument('--zsl', default=0, type=int)
    
    ###############################
    ####### optim parameters ######
    ###############################
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    
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
    parser.add_argument('--batch-size', default=56, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of training epochs')
    parser.add_argument('--sync_bn', type=bool, default=False,
                        help='synchronic batch only with distributed gpu')
    parser.add_argument('--feat-dim', type=int, default=128,
                        help='feature dimension for contrastive learning')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--method_used', type=str, default='MultiSupCon',
                        choices=['MultiSupCon', 'SupCon', 'SimCLR', "None"], help='choose method')
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                    help="Save the model periodically")
    parser.add_argument('--run', default=0, type=int,
                        metavar='N', help='run number')
    parser.add_argument("--dump_path", type=str, default="./experiment_eval_decoder",
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
            val_dataset = MultiLabelNUS(
                args.data,
                split="val",
                transform=transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor()
                ]),
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
        val_dataset,
        shuffle=False,
        batch_size=128,
        num_workers=8
    )
    logger.info("Building data done with train {} images loaded and val {} images loaded.".format(len(train_dataset), len(val_dataset)))
    model = create_model_base(args)
    if args.model_path:
        # Loading model
        checkpoint = torch.load(args.model_path,
            map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count()))
        model.load_state_dict(checkpoint['model_state_dict'])
        # freeze
        if args.freeze:
            for param in model.parameters():
                param.requires_grad = False
    # Adding classification head
    model = add_ml_decoder_head(model, args.num_classes)
    # Synchronize batch norm layers
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Copy model to GPU
    model = model.cuda()
    cudnn.benchmark = True
    logger.info(f'Building model {args.model_name} done')
    
    # Build optimizer, criterion and scheduler
    if args.method == "CrossEntropy":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.method == "Asymetric":
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, args.weight_decay)
    optimizer = Adam(params=parameters, lr=args.learning_rate, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, 
                                        max_lr=args.learning_rate, 
                                        steps_per_epoch=steps_per_epoch, 
                                        epochs=args.epochs,
                                        pct_start=0.2)
    logger.info("Building optimizer, criterion and learing scheduler done.")
    
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
                name=f"validating_linear_multi_sup_con_{args.run}",
                resume=True 
            )
        else:
            wandb.init(
                project="test-project", 
                entity="pwr-multisupcontr",
                name=f"ml_decoder_{args.method_used}_{args.run}",
                config={
                    "data": args.data,
                    "image-size": args.image_size,
                    "batch-size": args.batch_size,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "method": args.method,
                    "seed": args.seed,
                    "freeze": args.freeze,
                    "use-ml-decoder": args.use_ml_decoder,
                    "num-of-groups": args.num_of_groups,
                    "decoder-embedding": args.decoder_embedding,
                    "zsl": args.zsl,
                    "sync_bn:": args.sync_bn,
                    "numb_of_gpu_used": args.gpu_to_work_on
                }
            )
        wandb.watch(model, log="all")
    
    # Load checkpoint
    if resume:
        # Get last restore
        checkpoint_last = os.path.join(log_path, "last_checkpoint.pth.tar")
        checkpoint_best = os.path.join(log_path, "best_checkpoint.pth.tar")
        checkpoint = torch.load(checkpoint_last,
            map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count()))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        best = torch.load(checkpoint_best,
            map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count()))
    else:
        start_epoch = 0
        best = {}
    
    ###############################
    ########### TRAINING ##########
    ###############################
    best = {}
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82
    val_map = None
    val_map_ema = None
    for epoch in range(start_epoch, args.epochs):
        
        # train the network for one epoch
        logger.info(f"============ Starting epoch {epoch} ... ============")
        
        # set sampler
        train_loader.sampler.set_epoch(epoch)
        
        # train the network
        scores = train(
            train_loader,
            model,
            optimizer,
            criterion,
            scheduler,
            ema,
            epoch,
            logger,
            args
        )

        # save checkpoints
        if args.rank == 0:
            # Validate
            val_map, val_map_ema, mif1, maf1, sf1 = validate(val_loader, model, ema)
            logger.info(f"Validate: Epoch [{epoch}], Mean Average Precision: {val_map[0]:.3f}")
            # Log to wandb metrics
            wandb.log({
                "loss": scores[1],
                "map": scores[2],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "val_map": val_map[0],
                "val_map_ema": val_map_ema[0],
                "micro_f1_score": mif1,
                "macro_f1_score": maf1,
                "samples_f1_score": sf1
            }, step=epoch)
            # Update best loss
            if "map" not in best.keys():
                best = { 
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': scores[1],
                    'map': max(val_map[0], val_map_ema[0]),
                }
                # Save best loss
                checkpoint_path = os.path.join(log_path, f"best_checkpoint.pth.tar")
                torch.save(best, checkpoint_path)
                wandb.save(checkpoint_path)
            else:
                if max(val_map[0], val_map_ema[0]) > best["map"]:
                    best = { 
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': scores[1],
                        'map': max(val_map[0], val_map_ema[0]),
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
                    'map': max(val_map[0], val_map_ema[0]),
                    }, checkpoint_path)
                torch.save({ 
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': scores[1],
                    'map': max(val_map[0], val_map_ema[0]),
                    }, checkpoint_last)
                wandb.save(checkpoint_last)

    if args.rank == 0:
        #Log final metrics
        wandb.run.summary["best_Mean Average Precision"] = best['map']
        wandb.run.summary["best_loss"] = best['loss']
        wandb.run.summary["best_epoch"] = best['epoch']
        if val_map:
            val_ap = [(i, val_map[1][i]) for i in range(len(val_map[1]))]
            val_ap_ema = [(i, val_map_ema[1][i]) for i in range(len(val_map_ema[1]))]
            wandb.log({
                "val_ap": wandb.Table(data=val_ap, columns=["class_id", "Average_precision"]),
                "val_ap_ema": wandb.Table(data=val_ap_ema, columns=["class_id", "Average_precision"])
            })
        # End wandb
        wandb.finish()

    ###############################
    ########### Finished ##########
    ###############################
    logger.info("============ Finished ============")
    logger.info(f"Best Mean Average Precision:: {best['map']} with loss {best['loss']} on epoch {best['epoch']}")


def train(train_loader, model, optimizer, criterion, scheduler, ema, epoch, logger, args):
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
        
        # Compute loss
        with autocast():  # mixed precision
            output = model(images).float()  # sigmoid will be done in loss !
        loss = criterion(output, labels.float())
        # update metric
        losses.update(loss.item(), bsz)
        mAP_score, _ = mAP(labels.cpu().detach().numpy(), output.cpu().detach().numpy())
        mAPs.update(mAP_score, bsz)
        
        # Optimizer
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        ema.update(model)
        
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

def validate(val_loader, model, ema):
    model.eval()
    
    Sig = torch.nn.Sigmoid()
    labels_all = []
    outputs = []
    outputs_ema = []
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            labels = labels.max(dim=1)[0]
            if torch.cuda.is_available():
                images = images.cuda()
            with autocast():
                output = Sig(model(images.cuda())).cpu()
                output_ema = Sig(ema.module(images.cuda())).cpu()
            # Gather results
            outputs.append(output.cpu().detach())
            outputs_ema.append(output_ema.cpu().detach())
            labels_all.append(labels.cpu().detach())
    
    #Calc metrics
    labels_all = torch.cat(labels_all).numpy()
    outputs = torch.cat(outputs).numpy()
    outputs_ema = torch.cat(outputs_ema).numpy()
    mAP_score, ap_score = mAP(labels_all, outputs)
    mAP_score_ema, ap_score_ema = mAP(labels_all, outputs_ema)
    mif1 = max(f1_score(labels_all, outputs, average="micro"),f1_score(labels_all, outputs_ema, average="micro")) 
    maf1 = max(f1_score(labels_all, outputs, average="macro"),f1_score(labels_all, outputs_ema, average="macro")) 
    sf1 = max(f1_score(labels_all, outputs, average="samples"),f1_score(labels_all, outputs_ema, average="samples")) 
    return (mAP_score, ap_score), (mAP_score_ema, ap_score_ema), mif1, maf1, sf1


if __name__ == '__main__':
    main()
