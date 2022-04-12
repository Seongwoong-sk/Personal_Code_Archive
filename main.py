# Main은 깔끔하게 가급적 짜는 걸로.

import argparse # Terminal에서 실행할 수 있게
import training
import datasets
from tsne import tsne

if __name__ == "__main__":

    # 터미널에서 해당되는 변수들 컨트롤 가능
    parser = argparse.ArgumentParser(description='CIFAR10 image classification') # argparse 선언
    # Control 하고자 하는 변수들
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=101, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--l2', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--model_name', default='resnet18', type=str, help='model name')
    parser.add_argument('--pretrained', default=None, type=str, help='model path')
    parser.add_argument('--train', default='train', type=str, help='train and eval')
    args = parser.parse_args()
    print(args)

    if args.train == 'train':
        # 데이터 불러오기
        trainloader, testloader = datasets.dataloader(args.batch_size, 'train')
        print('Completed loading your datasets.')
        
        # 모델 불러오기 및 학습하기
        learning = training.SupervisedLearning(trainloader, testloader, args.model_name, args.pretrained)
        learning.train(args.epoch, args.lr, args.l2)
        
    else:
        trainloader, testloader = datasets.dataloader(args.batch_size, 'eval')
        learning = training.SupervisedLearning(trainloader, testloader, args.model_name, args.pretrained)
        print('Completed loading your datasets.')
        
        train_acc = learning.eval(trainloader)
        test_acc = learning.eval(testloader)
        print(f' Train Accuracy: {train_acc}, Test Accuraccy: {test_acc}')


        # t-SNE graph
        tsne(testloader, args.model_name, args.pretrained)

