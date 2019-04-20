def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if epoch <=5:
    	lr = start_lr
    elif epoch < 15 :
    	lr =start_lr*0.1
    # elif epoch <30:
    # 	lr = start_lr*0.01
    else:
    	lr = start_lr*0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr