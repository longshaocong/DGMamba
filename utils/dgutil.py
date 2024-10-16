import torch



def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0    
    '''t represent the index of the dataloader in eval_loader, e.g., eval_loader = [0, 1, 2, 0, 1, 2, 3], 
    where 0-4 is the proxy of the domian, the 4-th domain is the target domain. the first three 0 1 2 is the source domain, 
    while the latter 0 1 2 is the valid-set, and the last 3 is the target domain'''
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict

def DG_accuracy(model, loader):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = model(x)

            if p.size(1) == 1:
                correct += (p.gt(0).ep(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    model.train()
    return correct / total

def img_param_init(args):
    dataset = args.dataset
    if dataset == 'PACS':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'VLCS':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset == 'office-home':
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
    elif dataset == 'terra_incognita':
        domains = ['location_100', 'location_38', 'location_43', 'location_46']
    elif dataset == 'domain_net':
        domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'], 
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'], 
        'office-home': ['Art', 'Clipart', 'Product', 'Real_World'], 
        'terra_incognita': ['location_100', 'location_38', 'location_43', 'location_46'], 
        'domain_net': ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    }
    args.input_shape = (3, 224, 224)
    if args.dataset == 'PACS':
        args.num_classes = 7
    elif dataset == 'VLCS':
        args.num_classes = 5
    elif args.dataset == 'office-home':
            args.num_classes = 65
    elif args.dataset == 'terra_incognita':
        args.num_classes = 10
    elif args.dataset == 'domain_net':
        args.num_classes = 345
    return args