import utils
import argparser
import os
from utils.logger import Logger

from apex.parallel import DistributedDataParallel
from apex import amp
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import random
import torch
from torch.utils import data
from torch import distributed

from dataset import VOCSegmentationIncremental, AdeSegmentationIncremental, COCOSegmentationIncremental
from dataset import transform
from metrics import StreamSegMetrics

from segmentation_module import make_model

from train import Trainer
import tasks


def save_ckpt(path, model, trainer, optimizer, scheduler, epoch, best_score):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "trainer_state": trainer.state_dict()
    }
    torch.save(state, path)


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = transform.Compose([
        transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if opts.crop_val:
        val_transform = transform.Compose([
            transform.Resize(size=opts.crop_size),
            transform.CenterCrop(size=opts.crop_size),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        # no crop, batch size = 1
        val_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    # labels, labels_old, path_base, num_class = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task,
                                                                     opts.step)
    labels_cum = labels_old + labels

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    elif opts.dataset == 'ade':
        dataset = AdeSegmentationIncremental
    elif opts.dataset == 'coco':
        dataset = COCOSegmentationIncremental
    else:
        raise NotImplementedError

    if opts.overlap:
        path_base += "-ov"

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    train_dst = dataset(root=opts.data_root, train=True, transform=train_transform,
                        labels=list(labels), labels_old=list(labels_old),
                        idxs_path=path_base + f"/train-{opts.step}.npy",
                        masking=not opts.no_mask, overlap=opts.overlap, step=opts.step, few_shot=opts.few_shot,
                        num_shot=opts.num_shot, batch_size=opts.batch_size, task=opts.task, folding=opts.folding)

    if not opts.no_cross_val:  # if opts.cross_val:
        train_len = int(0.8 * len(train_dst))
        val_len = len(train_dst)-train_len
        train_dst, val_dst = torch.utils.data.random_split(train_dst, [train_len, val_len])
    else:  # don't use cross_val
        val_dst = dataset(root=opts.data_root, train=False, transform=val_transform,
                          labels=list(labels), labels_old=list(labels_old),
                          idxs_path=path_base + f"/val-{opts.step}.npy",
                          masking=not opts.no_mask, overlap=True, batch_size=opts.batch_size, task=opts.task, folding=opts.folding)

    image_set = 'train' if opts.val_on_trainset else 'val'
    ###################################################################################################
    test_dst = dataset(root=opts.data_root, train=opts.val_on_trainset, transform=val_transform,
                       labels=list(labels_cum),
                       idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy",
                       batch_size=opts.batch_size, task=opts.task, folding=opts.folding)
    # labels_test = list(range(21))
    # test_dst = dataset(root=opts.data_root, train=opts.val_on_trainset, transform=val_transform,
    #                    labels=list(labels_test),
    #                    idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy", batch_size=opts.batch_size)
    ###################################################################################################

    return train_dst, val_dst, test_dst, len(labels_cum)
    # return train_dst, val_dst, test_dst, num_class


def main(opts):
    # if not opts.all_step or opts.step == 0 or (opts.all_step and opts.step == 1):
    #     distributed.init_process_group(backend='nccl', init_method='env://')

    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(0)

    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    if rank == 0:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step)
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)

    logger.print(f"Device: {device}")

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # xxx Set up dataloader
    train_dst, val_dst, test_dst, n_classes = get_dataset(opts)
    ####################################################################################
    # final_file_name =[]
    # if opts.few_shot and opts.step > 0:
    #     seed = opts.random_seed
    #     np.random.seed(seed)
    #     random.seed(seed)
    #     torch.manual_seed(seed)
    #     for _ in range(opts.num_shot):
    #         idx = random.choice(train_dst)
    #         while True:
    #             print('file name: ')
    #             print(final_file_name)
    #             print('idx: ')
    #             print(idx)
    #             if idx not in final_file_name:
    #                 final_file_name.append(idx)
    #                 break
    #             else:
    #                 idx = random.choice(train_dst)
    # else:
    #     final_file_name = train_dst
    #
    # train_dst = final_file_name

    # while len(train_dst) < opts.batch_size:
    #     if opts.num_shot == 5:
    #         train_dst = train_dst * 20
    #     elif opts.num_shot == 1:
    #         train_dst = train_dst * 100
    #     else:
    #         train_dst = train_dst * 5
####################################################################################



    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size,
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)



    ##################################################################################
    # print('\n printing max and min index 5 train image: ')
    # print(train_dst[0][0].size())
    # print(train_dst[1][0].size())
    # print(torch.max(train_dst[5][0]))
    # print(torch.min(train_dst[5][0]))
    # print(train_dst[0][0].is_cuda)
    # print('\n printing max and min index 5 train label: ')
    # print(train_dst[0][1].size())
    # print(train_dst[1][1].size())
    # print(torch.max(train_dst[5][1]))
    # print(torch.min(train_dst[5][1]))
    # print(train_dst[0][1].is_cuda)
    ##################################################################################
    ##################################################################################
    # print('\n printing max and min index 0 and 1 test image: ')
    # print(test_dst[0][0].size())
    # print(test_dst[1][0].size())
    # print(torch.max(test_dst[17][0]))
    # print(torch.min(test_dst[17][0]))
    # print(test_dst[0][0].is_cuda)
    # print('\n printing max and min index 0 and 1 test label: ')
    # print(test_dst[0][1].size())
    # print(test_dst[1][1].size())
    # print(torch.max(test_dst[17][1]))
    # print(torch.min(test_dst[17][1]))
    # print(test_dst[0][1].is_cuda)
    ##################################################################################







    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                f" Test set: {len(test_dst)}, n_classes {n_classes}")
    logger.info(f"Total batch size is {opts.batch_size * world_size}")

    # xxx Set up model
    logger.info(f"Backbone: {opts.backbone}")

    step_checkpoint = None
    model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
    logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")

    if opts.step == 0:  # if step 0, we don't need to instance the model_old
        model_old = None
    else:  # instance model_old
        model_old = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1))

    if opts.fix_bn:
        model.fix_bn()

    logger.debug(model)

    # xxx Set up optimizer
    params = []
    if not opts.freeze:
        params.append({"params": filter(lambda p: p.requires_grad, model.body.parameters()),
                       'weight_decay': opts.weight_decay})

    params.append({"params": filter(lambda p: p.requires_grad, model.head.parameters()),
                   'weight_decay': opts.weight_decay})

    params.append({"params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                   'weight_decay': opts.weight_decay})

    optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, max_iters=opts.epochs * len(train_loader), power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    else:
        raise NotImplementedError
    logger.debug("Optimizer:\n%s" % optimizer)

    if model_old is not None:
        [model, model_old], optimizer = amp.initialize([model.to(device), model_old.to(device)], optimizer,
                                                       opt_level=opts.opt_level)
        model_old = DistributedDataParallel(model_old)
    else:
        model, optimizer = amp.initialize(model.to(device), optimizer, opt_level=opts.opt_level)

    # Put the model on GPU
    model = DistributedDataParallel(model, delay_allreduce=True)

    # xxx Load old model from old weights if step > 0!
    if opts.step > 0:
        # get model path
        if opts.step_ckpt is not None:
            path = opts.step_ckpt
        else:
            path = f"checkpoints/step/{task_name}_{opts.name}_{opts.step - 1}.pth"

        # generate model from path
        if os.path.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False because of incr. classifiers
            if opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                model.module.init_new_classifier(device)
            # Load state dict from the model state dict, that contains the old model parameters
            model_old.load_state_dict(step_checkpoint['model_state'], strict=True)  # Load also here old parameters
            logger.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        elif opts.debug:
            logger.info(f"[!] WARNING: Unable to find of step {opts.step - 1}! Do you really want to do from scratch?")
        else:
            raise FileNotFoundError(path)
        # put the old model into distributed memory and freeze it
        for par in model_old.parameters():
            par.requires_grad = False
        model_old.eval()

    # xxx Set up Trainer
    trainer_state = None
    # if not first step, then instance trainer from step_checkpoint
    if opts.step > 0 and step_checkpoint is not None:
        if 'trainer_state' in step_checkpoint:
            trainer_state = step_checkpoint['trainer_state']

    # instance trainer (model must have already the previous step weights)
    trainer = Trainer(model, model_old, device=device, opts=opts, trainer_state=trainer_state,
                      classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))

    # xxx Handle checkpoint for current model (model old will always be as previous step or None)
    best_score = 0.0
    cur_epoch = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        logger.info("[!] Model restored from %s" % opts.ckpt)
        # if we want to resume training, resume trainer from checkpoint
        if 'trainer_state' in checkpoint:
            trainer.load_state_dict(checkpoint['trainer_state'])
        del checkpoint
    else:
        if opts.step == 0:
            logger.info("[!] Train from scratch")

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    if rank == 0 and opts.sample_num > 0:
        sample_ids = np.random.choice(len(val_loader), opts.sample_num, replace=False)  # sample idxs for visualization
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    TRAIN = not opts.test
    val_metrics = StreamSegMetrics(n_classes, opts.few_shot)
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0,100, (1,1)))
    # train/val here
    while cur_epoch < opts.epochs and TRAIN:
        # =====  Train  =====
        model.train()

        epoch_loss = trainer.train(cur_epoch=cur_epoch, optim=optimizer,
                                   train_loader=train_loader, scheduler=scheduler, logger=logger)

        logger.info(f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0]+epoch_loss[1]},"
                    f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}")

        # =====  Log metrics on Tensorboard =====
        logger.add_scalar("E-Loss", epoch_loss[0]+epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-reg", epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-cls", epoch_loss[0], cur_epoch)

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0:
            logger.info("validate on val set...")
            model.eval()
            val_loss, val_score, ret_samples = trainer.validate(loader=val_loader, metrics=val_metrics,
                                                                ret_samples_ids=sample_ids, logger=logger)

            logger.print("Done validation")
            logger.info(f"End of Validation {cur_epoch}/{opts.epochs}, Validation Loss={val_loss[0]+val_loss[1]},"
                        f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")

            logger.info(val_metrics.to_str(val_score))

            # =====  Save Best Model  =====
            if rank == 0:  # save best model at the last iteration
                score = val_score['Mean IoU']
                # best model to build incremental steps
                save_ckpt(f"checkpoints/step/{task_name}_{opts.name}_{opts.step}.pth",
                          model, trainer, optimizer, scheduler, cur_epoch, score)
                logger.info("[!] Checkpoint saved.")

            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("V-Loss", val_loss[0]+val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-reg", val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-cls", val_loss[0], cur_epoch)
            logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
            logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
            logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
            logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch)
            # logger.add_figure("Val_Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)

            # keep the metric to print them at the end of training
            results["V-IoU"] = val_score['Class IoU']
            results["V-Acc"] = val_score['Class Acc']

            for k, (img, target, lbl) in enumerate(ret_samples):
                img = (denorm(img) * 255).astype(np.uint8)
                target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
                lbl = label2color(lbl).transpose(2, 0, 1).astype(np.uint8)

                concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                logger.add_image(f'Sample_{k}', concat_img, cur_epoch)

        cur_epoch += 1

    # =====  Save Best Model at the end of training =====
    if rank == 0 and TRAIN:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(f"checkpoints/step/{task_name}_{opts.name}_{opts.step}.pth",
                  model, trainer, optimizer, scheduler, cur_epoch, best_score)
        logger.info("[!] Checkpoint saved.")

    torch.distributed.barrier()

    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(test_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)

    # load best model
    if TRAIN:
        model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
        # Put the model on GPU
        model = DistributedDataParallel(model.cuda(device))
        ckpt = f"checkpoints/step/{task_name}_{opts.name}_{opts.step}.pth"
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        logger.info(f"*** Model restored from {ckpt}")
        del checkpoint
        trainer = Trainer(model, None, device=device, opts=opts)

    model.eval()

    val_loss, val_score, _ = trainer.validate(loader=test_loader, metrics=val_metrics, logger=logger)
    logger.print("Done test")
    logger.info(f"*** End of Test, Total Loss={val_loss[0]+val_loss[1]},"
                f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")
    logger.info(val_metrics.to_str(val_score))
    logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    logger.add_results(results)

    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'], opts.step)
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'], opts.step)
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'], opts.step)

    logger.close()

    class_iou = list(val_score['Class IoU'].values())
    class_acc = list(val_score['Class Acc'].values())

    if opts.dataset == 'voc':
        task_dict = tasks.tasks_voc[opts.task]
    elif opts.dataset == 'coco':
        task_dict = tasks.tasks_coco[opts.task]
    first_cls = len(task_dict[0])

    print(f"...from 0 to {first_cls - 1} : best/test_before_mIoU : %.6f" % np.mean(
        class_iou[:first_cls]))
    print(f"...from {first_cls} to {len(class_iou) - 1} best/test_after_mIoU : %.6f" % np.mean(
        class_iou[first_cls:]))
    print(f"...from 0 to {first_cls - 1} : best/test_before_acc : %.6f" % np.mean(
        class_acc[:first_cls]))
    print(f"...from {first_cls} to {len(class_iou) - 1} best/test_after_acc : %.6f" % np.mean(
        class_acc[first_cls:]))

    return np.mean(class_iou[:first_cls]), np.mean(class_iou[first_cls:])


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    os.makedirs("checkpoints/step", exist_ok=True)

    distributed.init_process_group(backend='nccl', init_method='env://')
    if opts.step == 0 or not opts.all_step:
        main(opts)
    else:
        base_iou = []
        novel_iou = []
        np.random.seed(1234)
        seed_list = np.random.randint(0, 99999, size=(opts.num_runs,))

        orig_step = opts.step
        for i in range(opts.num_runs):
            print('\nStarting run {}: \n'.format(i))
            for j in range(orig_step, 6):
                print('\nRun {} Step {}: '.format(i, j))
                opts.step = j
                opts.random_seed = seed_list[i]
                new_base, new_novel = main(opts)
                if j == 5:
                    base_iou.append(new_base)
                    novel_iou.append(new_novel)

        print("test base iou: ")
        print(base_iou)
        print("test novel iou: ")
        print(novel_iou)

        base_iou = np.array(base_iou)
        novel_iou = np.array(novel_iou)
        total_mean_iou = np.add(base_iou, novel_iou) / 2
        max_mean_iou_index = np.where(total_mean_iou == np.amax(total_mean_iou))[0][0]
        print("Results of {} runs in a non-few-shot setting".format(opts.num_runs))
        print("max Base IoU: {:.10f} max Novel IoU: {:.10f}".format(base_iou[max_mean_iou_index],
                                                                  novel_iou[max_mean_iou_index]))

        print("mean Base IoU: {:.10f} mean Novel IoU: {:.10f}".format(np.mean(base_iou),
                                                                    np.mean(novel_iou)))
