import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# Define the digit classification CNN model
class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        self.layer1 = nn.Conv2d(1, 32, 3, 1)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.drop_a = nn.Dropout(0.25)
        self.drop_b = nn.Dropout(0.5)
        self.dense1 = nn.Linear(9216, 128)
        self.dense2 = nn.Linear(128, 10)

    def forward(self, input_img):
        input_img = self.layer1(input_img)
        input_img = F.relu(input_img)
        input_img = self.layer2(input_img)
        input_img = F.relu(input_img)
        input_img = F.max_pool2d(input_img, 2)
        input_img = self.drop_a(input_img)
        input_img = torch.flatten(input_img, 1)
        input_img = self.dense1(input_img)
        input_img = F.relu(input_img)
        input_img = self.drop_b(input_img)
        input_img = self.dense2(input_img)
        return F.log_softmax(input_img, dim=1)

# Perform training for one epoch
def run_training(params, net, dev, loader, opt, current_epoch):
    print("Training started!\n")
    net.train()
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        preds = net(x)
        loss_val = F.nll_loss(preds, y)
        loss_val.backward()
        opt.step()
        if batch_idx % params.log_interval == 0:
            print(f'Train Epoch: {current_epoch} [{batch_idx * len(x)}/{len(loader.dataset)} '
                  f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss_val.item():.6f}')
            if params.dry_run:
                break
    print("Training ended!\n")

# Optional testing function
def run_evaluation(net, dev, eval_loader):
    net.eval()
    total_loss = 0
    correct_preds = 0
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(dev), y.to(dev)
            outputs = net(x)
            total_loss += F.nll_loss(outputs, y, reduction='sum').item()
            guess = outputs.argmax(dim=1, keepdim=True)
            correct_preds += guess.eq(y.view_as(guess)).sum().item()
    total_loss /= len(eval_loader.dataset)
    print(f'\nTest set: Average loss: {total_loss:.4f}, '
          f'Accuracy: {correct_preds}/{len(eval_loader.dataset)} '
          f'({100. * correct_preds / len(eval_loader.dataset):.0f}%)\n')

# Program entry point
def main():
    parser = argparse.ArgumentParser(description='Digit Classifier - PyTorch MNIST')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-accel', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true')
    args = parser.parse_args()

    # Setup device
    use_cuda = not args.no_accel and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if use_cuda else "cpu")

    # Loader setup
    train_config = {'batch_size': args.batch_size}
    eval_config = {'batch_size': args.test_batch_size}
    if use_cuda:
        loader_opts = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_config.update(loader_opts)
        eval_config.update(loader_opts)

    # Transform input image
    preprocess_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=preprocess_mnist)
    train_data_loader = torch.utils.data.DataLoader(train_data, **train_config)

    # Initialize model and optimizer
    digit_classifier = DigitNet().to(dev)
    optimizer = optim.Adadelta(digit_classifier.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Main training loop
    for ep in range(1, args.epochs + 1):
        run_training(args, digit_classifier, dev, train_data_loader, optimizer, ep)
        # run_evaluation(digit_classifier, dev, test_loader)  # Optional
        lr_scheduler.step()

    # Save trained model
    if args.save_model:
        print("Done with training!")
        print("The model is saved at /mnt/ac11950_model.pt")
        torch.save(digit_classifier.state_dict(), "/mnt/ac11950_model.pt")

# Execute main
if __name__ == '__main__':
    main()
