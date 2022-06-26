import os
import torch
import torch.nn as nn
from urllib import request
from networks.ml_decoder import MLDecoder
from networks.tresnet import TResnetM, TResnetL, TResnetXL
from networks.resnet_big import Resnet18, Resnet34, Resnet50, Resnet101


def create_model_base(args, load_head=False):
    """Create a base model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name == 'tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name == 'tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name == 'tresnet_xl':
        model = TResnetXL(model_params)
    elif args.model_name == 'resnet18':
        model = Resnet18(model_params)
    elif args.model_name == 'resnet34':
        model = Resnet34(model_params)
    elif args.model_name == 'resnet50':
        model = Resnet50(model_params)
    elif args.model_name == 'resnet101':
        model = Resnet101(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    # loading pretrain model
    model_path = args.model_path
    if args.model_name == 'tresnet_l' and os.path.exists("./tresnet_l.pth"):
        model_path = "./tresnet_l.pth"
    if model_path:  # make sure to load pretrained model
        if not os.path.exists(model_path):
            print("downloading pretrain model...")
            request.urlretrieve(args.model_path, "./tresnet_l.pth")
            model_path = "./tresnet_l.pth"
            print('done')
        state = torch.load(model_path, map_location='cpu')
        if not load_head:
            filtered_dict = {k: v for k, v in state['model'].items() if
                             (k in model.state_dict() and 'head.fc' not in k)}
            model.load_state_dict(filtered_dict, strict=False)
        else:
            model.load_state_dict(state['model'], strict=True)
    
    #adding contrastive head
    model = add_mulitsupcon_head(model, feat_dim=args.feat_dim)
    return model

def add_mulitsupcon_head(model, head='mlp', feat_dim=128):
    num_features = model.num_features
    if hasattr(model, 'head'):
        del model.head
        if head == 'linear':
            model.head = nn.Linear(num_features, feat_dim)
        elif head == 'mlp':
            model.head = nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.ReLU(inplace=True),
                nn.Linear(num_features, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
    else:
        print("model is not suited for ml-supcon")
        exit(-1)

    return model

def add_classification_head(model, num_classes=-1):
    if num_classes == -1:
        num_classes = model.num_classes
    num_features = model.num_features
    if hasattr(model, 'head'):
        del model.head
        model.head = nn.Linear(num_features, num_classes)
    else:
        print("model is not suited for ml-supcon")
        exit(-1)

    return model

def add_ml_decoder_head(model, num_classes=-1, num_of_groups=-1, decoder_embedding=768, zsl=0):
    if num_classes == -1:
        num_classes = model.num_classes
    num_features = model.num_features
    if hasattr(model, 'head'):  # tresnet
        if hasattr(model, 'global_pool'):
            model.global_pool = nn.Identity()
        del model.head
        model.head = MLDecoder(num_classes=num_classes, initial_num_features=num_features, num_of_groups=num_of_groups,
                               decoder_embedding=decoder_embedding, zsl=zsl)
    else:
        print("model is not suited for ml-decoder")
        exit(-1)

    return model
