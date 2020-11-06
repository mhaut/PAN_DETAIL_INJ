import argparse
import models
import myaux
import numpy as np
import scipy.io as sio
import torch


def train(trainloader, model, criterion, optimizer, use_cuda):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(testloader, model, criterion, use_cuda):
    model.eval()
    tloss = 0.0
    for i, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        outputs = model(inputs)
        tloss += criterion(outputs, targets).item()
    return tloss / (i+1)


def predict(testloader, model, use_cuda):
    model.eval()
    predicted = []
    for i, (inputs, targets) in enumerate(testloader):
        if use_cuda: inputs = inputs.cuda()
        inputs = torch.autograd.Variable(inputs, volatile=True)
        outputs = model(inputs)
        [predicted.append(a) for a in model(inputs).data.cpu().numpy()] 
    return np.array(predicted)


def main():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument('--dataset', default='ik', type=str, help='dataset (options: ik)')
    parser.add_argument('--model', default='DiCNN1', type=str, help='dataset (options: ik)')
    parser.add_argument('--tr_bsize', default=64, type=int, help='mini-batch train size (default: 64)')
    parser.add_argument('--te_bsize', default=1000, type=int, help='mini-batch test size (default: 16)')    
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=250, type=int, help='number of total epochs to run')


    use_cuda = True
    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}


    train_loader, val_loader, te_loader, input_size = myaux.load_data(args.dataset, (args.tr_bsize, args.te_bsize))
    

    if args.model == "DiCNN1":   model = models.DiCNN1(input_size)
    elif args.model == "DiCNN2": model = models.DiCNN2(input_size-1)
    if use_cuda: model = model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50,100,150,200], gamma=0.5)
    best_err = float("inf")
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, use_cuda)
        tr_err  = test(train_loader, model, criterion, use_cuda)
        val_err = test(val_loader, model, criterion, use_cuda)

        scheduler.step()
        print("EPOCH", epoch, "TR_LOSS", tr_err, "VAL LOSS", val_err)
        # save model
        if val_err < best_err:
            state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'err': val_err,
                    'best_err': best_err,
                    'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, "/tmp/best_model.pth.tar")
            best_err = val_err

    checkpoint = torch.load("/tmp/best_model.pth.tar")
    best_err = checkpoint['best_err']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    test_loss = test(te_loader, model, criterion, use_cuda)
    print("FINAL MSE LOSS", test_loss)
    output = predict(te_loader, model, use_cuda)
    output = np.transpose(output, [2, 3, 1, 0])
    sio.savemat("outputTORCH.mat", {"output": output})

if __name__ == '__main__':
    main()
