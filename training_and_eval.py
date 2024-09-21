from helper_functions import *
from model_classes import *

def test_accuracy(model, testloader, device):
    denom = 0
    num = 0
    for i, data in enumerate(testloader):
        xs, ys = data
        xs = xs.to(device)
        ys = ys.to(device)
        output = torch.argmax(model(xs)).item()
        label = ys.item()
        if (output == label):
            num += 1
        denom += 1
    return num / denom

def train_vae(dataloader, device, num_epochs=50, batch_size=128, learning_rate=3e-4):
    print("TRAINING VAE")
    losses = []
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    train_loss = 0.0
    start_time_total = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, labels)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            losses.append(loss.item() / batch_size)
        end_time = time.time()
        train_loss /= len(dataloader.dataset)
        print(f"Epoch: {epoch+1}/{num_epochs}, Epoch Train Loss: {train_loss:.4f}")
        print("Epoch Time:", end_time - start_time, "seconds")
        print("")
    end_time_total = time.time()
    print("Total VAE training time: ", end_time_total - start_time_total, "seconds")
    print("")
    return model, losses

def sample(model, num_samples, labels, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z, labels)
        return samples

def save_synth_data(vae_model, device, filename='generated_MNIST.pt'):
    imgs_tot = []
    labels_tot = []
    for i in range(10):
        labels = [i] * 6000
        imgs = sample(vae_model, 6000, torch.Tensor(labels).long().to(device), device)
        imgs_tot.append(imgs)
        labels_tot.append(torch.Tensor(labels).long())
    imgs_tot = [item for sublist in imgs_tot for item in sublist]
    labels_tot = [item for sublist in labels_tot for item in sublist]

    imgs_tot = torch.stack(imgs_tot)
    labels_tot = torch.tensor(labels_tot, dtype=torch.long)
    torch.save((imgs_tot, labels_tot), filename)

class GeneratedMNISTDataset(Dataset):
    def __init__(self, filepath='generated_MNIST.pt'):
        self.data, self.targets = torch.load(filepath)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def prep_real_data(trainset):
    trainloader = get_trainloader(trainset, batch_size=1)
    data = []
    labels = []

    for batch_data, batch_labels in trainloader:
        data.append(batch_data)
        labels.append(batch_labels)

    data = torch.cat(data, dim=0)
    labels = torch.cat(labels, dim=0)

    return data, labels
    
class OriginalMNISTDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
def train_cnn(trainset, testloader, genset, mnist_ratio, device, generated_ratio=0.0, batch_size=16, num_epochs=5, learning_rate=1e-3):
    model = LeNet().to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    losses = []

    mnist_size = len(trainset)
    generated_size = len(genset)
    mnist_split_size = round(mnist_size * mnist_ratio)
    generated_split_size = round(generated_size * generated_ratio)

    mnist_indices = torch.randperm(mnist_size)[:mnist_split_size]
    generated_indices = torch.randperm(generated_size)[:generated_split_size]

    mnist_subset = torch.utils.data.Subset(trainset, mnist_indices)
    generated_subset = torch.utils.data.Subset(genset, generated_indices)

    mnist_subset.dataset.data = mnist_subset.dataset.data.cpu()
    mnist_subset.dataset.targets = mnist_subset.dataset.targets.cpu()
    generated_subset.dataset.data = generated_subset.dataset.data.cpu()
    generated_subset.dataset.targets = generated_subset.dataset.targets.cpu()

    combined_dataset = ConcatDataset([mnist_subset, generated_subset])
    combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    #print("TRAINING CNN")
    for epoch in range(num_epochs):

        for batch_data, batch_targets in combined_dataloader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item() / batch_size)
    
    model.eval()
    return model, losses, test_accuracy(model, testloader, device)

def train_pure_cnns(trainset, testloader, genset, mnist_ratios, device, nets=10, batch_size=16, num_epochs=5, learning_rate=1e-3):
    test_accuracies = []
    test_devs = []
    for mnist_ratio in mnist_ratios:
        start_time = time.time()
        print("Ratio:", mnist_ratio)
        test_accuracies_sub = []
        for i in range(nets):
            print("Net:", i+1, "/", nets)
            generated_ratio = 0.0
            _, _, test_accuracy = train_cnn(trainset, testloader, genset, mnist_ratio, device, generated_ratio=generated_ratio, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)
            test_accuracies_sub.append(test_accuracy)
        test_accuracies.append(np.mean(test_accuracies_sub))
        test_devs.append(np.std(test_accuracies_sub, ddof=1))
        end_time = time.time()
        print("Ratio Time:", end_time - start_time, "seconds")
        print("")
    return test_accuracies, test_devs

def train_synth_cnns(trainset, testloader, genset, gen_ratios, device, nets=10, batch_size=16, num_epochs=5, learning_rate=1e-3):
    test_accuracies = []
    test_devs = []
    for generated_ratio in gen_ratios:
        start_time = time.time()
        print("Ratio:", generated_ratio)
        test_accuracies_sub = []
        for i in range(nets):
            print("Net:", i+1, "/", nets)
            mnist_ratio = 0.0
            _, _, test_accuracy = train_cnn(trainset, testloader, genset, mnist_ratio, device, generated_ratio=generated_ratio, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)
            test_accuracies_sub.append(test_accuracy)
        test_accuracies.append(np.mean(test_accuracies_sub))
        test_devs.append(np.std(test_accuracies_sub, ddof=1))
        end_time = time.time()
        print("Ratio Time:", end_time - start_time, "seconds")
        print("")
    return test_accuracies, test_devs

def train_mixed_cnns(trainset, testloader, genset, mnist_ratios, device, nets=10, batch_size=16, num_epochs=5, learning_rate=1e-3):
    test_accuracies = []
    test_devs = []
    for mnist_ratio in mnist_ratios:
        start_time = time.time()
        print("Ratio:", mnist_ratio)
        test_accuracies_sub = []
        for i in range(nets):
            print("Net:", i+1, "/", nets)
            generated_ratio = 1.0 - mnist_ratio
            _, _, test_accuracy = train_cnn(trainset, testloader, genset, mnist_ratio, device, generated_ratio=generated_ratio, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)
            test_accuracies_sub.append(test_accuracy)
        test_accuracies.append(np.mean(test_accuracies_sub))
        test_devs.append(np.std(test_accuracies_sub, ddof=1))
        end_time = time.time()
        print("Ratio Time:", end_time - start_time, "seconds")
        print("")
    return test_accuracies, test_devs

def train_extended_cnns(trainset, testloader, genset, gen_ratios, device, nets=10, batch_size=16, num_epochs=5, learning_rate=1e-3):
    test_accuracies = []
    test_devs = []
    for generated_ratio in gen_ratios:
        start_time = time.time()
        print("Ratio:", generated_ratio)
        test_accuracies_sub = []
        for i in range(nets):
            print("Net:", i+1, "/", nets)
            mnist_ratio = 1.0
            _, _, test_accuracy = train_cnn(trainset, testloader, genset, mnist_ratio, device, generated_ratio=generated_ratio, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)
            test_accuracies_sub.append(test_accuracy)
        test_accuracies.append(np.mean(test_accuracies_sub))
        test_devs.append(np.std(test_accuracies_sub, ddof=1))
        end_time = time.time()
        print("Ratio Time:", end_time - start_time, "seconds")
        print("")
    return test_accuracies, test_devs

def train_cnn_vae(mnist_ratio_vae, trainset, testloader, genset, mnist_ratio, device, generated_ratio=0.0, batch_size=16, num_epochs=5, learning_rate=1e-3,
                  vae_epochs=50, vae_batch_size=128, vae_lr=3e-4):
    model = LeNet().to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    losses = []

    mnist_size = len(trainset)
    mnist_split_size = round(mnist_size * mnist_ratio_vae)
    mnist_indices = torch.randperm(mnist_size)[:mnist_split_size]
    mnist_subset = torch.utils.data.Subset(trainset, mnist_indices)
    mnist_subset.dataset.data = mnist_subset.dataset.data.cpu()
    mnist_subset.dataset.targets = mnist_subset.dataset.targets.cpu()
    dataloader = DataLoader(mnist_subset, batch_size=vae_batch_size, shuffle=True)
    vae, _ = train_vae(dataloader, device, num_epochs=vae_epochs, batch_size=vae_batch_size, learning_rate=vae_lr)
    filename = 'generated_MNIST_inner.pt'
    save_synth_data(vae, device, filename=filename)
    genset = GeneratedMNISTDataset(filepath=filename)

    generated_size = len(genset)
    generated_split_size = round(generated_size * generated_ratio)
    generated_indices = torch.randperm(generated_size)[:generated_split_size]
    generated_subset = torch.utils.data.Subset(genset, generated_indices)
    generated_subset.dataset.data = generated_subset.dataset.data.cpu()
    generated_subset.dataset.targets = generated_subset.dataset.targets.cpu()

    mnist_size = len(trainset)
    mnist_split_size = round(mnist_size * mnist_ratio)
    mnist_indices = torch.randperm(mnist_size)[:mnist_split_size]
    mnist_subset = torch.utils.data.Subset(trainset, mnist_indices)
    mnist_subset.dataset.data = mnist_subset.dataset.data.cpu()
    mnist_subset.dataset.targets = mnist_subset.dataset.targets.cpu()

    combined_dataset = ConcatDataset([mnist_subset, generated_subset])
    combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    #print("TRAINING CNN")
    for epoch in range(num_epochs):

        for batch_data, batch_targets in combined_dataloader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item() / batch_size)
    
    model.eval()
    return model, losses, test_accuracy(model, testloader, device)

def train_synth_cnns_vae(trainset, testloader, genset, mnist_ratios, device, nets=10, batch_size=16, num_epochs=5, learning_rate=1e-3,
                         vae_epochs=50, vae_batch_size=128, vae_lr=3e-4):
    test_accuracies = []
    test_devs = []
    for mnist_ratio_vae in mnist_ratios:
        start_time = time.time()
        print("Ratio:", mnist_ratio_vae)
        test_accuracies_sub = []
        for i in range(nets):
            print("Net:", i+1, "/", nets)
            generated_ratio = 1.0
            mnist_ratio = 0.0
            _, _, test_accuracy = train_cnn_vae(mnist_ratio_vae, trainset, testloader, genset, mnist_ratio, device, generated_ratio=generated_ratio, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                                                vae_epochs=vae_epochs, vae_batch_size=vae_batch_size, vae_lr=vae_lr)
            test_accuracies_sub.append(test_accuracy)
        test_accuracies.append(np.mean(test_accuracies_sub))
        test_devs.append(np.std(test_accuracies_sub, ddof=1))
        end_time = time.time()
        print("Ratio Time:", end_time - start_time, "seconds")
        print("")
    return test_accuracies, test_devs

def train_mixed_cnns_vae(trainset, testloader, genset, mnist_ratios, device, nets=10, batch_size=16, num_epochs=5, learning_rate=1e-3,
                         vae_epochs=50, vae_batch_size=128, vae_lr=3e-4):
    test_accuracies = []
    test_devs = []
    for mnist_ratio_vae in mnist_ratios:
        start_time = time.time()
        print("Ratio:", mnist_ratio_vae)
        test_accuracies_sub = []
        for i in range(nets):
            print("Net:", i+1, "/", nets)
            generated_ratio = 1.0 - mnist_ratio_vae
            mnist_ratio = mnist_ratio_vae
            _, _, test_accuracy = train_cnn_vae(mnist_ratio_vae, trainset, testloader, genset, mnist_ratio, device, generated_ratio=generated_ratio, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                                                vae_epochs=vae_epochs, vae_batch_size=vae_batch_size, vae_lr=vae_lr)
            test_accuracies_sub.append(test_accuracy)
        test_accuracies.append(np.mean(test_accuracies_sub))
        test_devs.append(np.std(test_accuracies_sub, ddof=1))
        end_time = time.time()
        print("Ratio Time:", end_time - start_time, "seconds")
        print("")
    return test_accuracies, test_devs

