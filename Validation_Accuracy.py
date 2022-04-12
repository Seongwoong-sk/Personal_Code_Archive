def eval(self, dataloader):

        correct = 0
        total = 0
        self.model.eval()
        
        with torch.no_grad():
            for data in dataloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

        return acc
