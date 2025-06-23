import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Przygotowanie danych
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=1000, shuffle=False)

# 2. Definicja modelu CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

model = Net()
optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss()

# 3. Funkcja ewaluacji na zbiorze testowym
def evaluate():
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    accuracy = correct / total
    return accuracy, all_preds, all_targets

# 4. Trening modelu i zbieranie statystyk
num_epochs = 10
train_losses = []
test_accuracies = []
conv1_weight_means = []
fc1_weight_means = []
fc2_weight_means = []

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    conv1_weight_means.append(model.conv1.weight.data.abs().mean().item())
    fc1_weight_means.append(model.fc1.weight.data.abs().mean().item())
    fc2_weight_means.append(model.fc2.weight.data.abs().mean().item())

    acc, _, _ = evaluate()
    test_accuracies.append(acc)
    print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Test Accuracy={acc:.4f}")

# 5. Wykresy strat i dokładności
plt.figure()
plt.plot(train_losses, label='Strata treningowa (NLL)')
plt.xlabel('Epoka')
plt.ylabel('Wartość straty')
plt.title('Wartość funkcji straty podczas uczenia')
plt.legend()
plt.grid(True)
plt.savefig('train_loss.png')
plt.show()

plt.figure()
plt.plot([1 - acc for acc in test_accuracies], label='Błąd klasyfikacji (1 - Accuracy)')
plt.xlabel('Epoka')
plt.ylabel('Błąd klasyfikacji')
plt.title('Zmiana błędu klasyfikacji w czasie')
plt.legend()
plt.grid(True)
plt.savefig('test_classification_error.png')
plt.show()

# Wykres skuteczności klasyfikacji (accuracy)
plt.figure()
plt.plot(test_accuracies, label='Test Accuracy', color='green')
plt.xlabel('Epoka')
plt.ylabel('Skuteczność klasyfikacji')
plt.title('Skuteczność klasyfikacji w czasie (Accuracy)')
plt.ylim(0.0, 1.05)
plt.grid(True)
plt.legend()
plt.savefig('classification_accuracy.png')
plt.show()

# 6. Wykresy średnich wartości wag
plt.figure()
plt.plot(conv1_weight_means, label='Warstwa Conv1')
plt.xlabel('Epoka')
plt.ylabel('Średnia wartość bezwzględna wag')
plt.title('Zmiany wag w czasie - warstwa Conv1')
plt.legend()
plt.grid(True)
plt.savefig('weights_conv1.png')
plt.show()

plt.figure()
plt.plot(fc1_weight_means, label='Warstwa FC1')
plt.xlabel('Epoka')
plt.ylabel('Średnia wartość bezwzględna wag')
plt.title('Zmiany wag w czasie - warstwa FC1')
plt.legend()
plt.grid(True)
plt.savefig('weights_fc1.png')
plt.show()

plt.figure()
plt.plot(fc2_weight_means, label='Warstwa FC2')
plt.xlabel('Epoka')
plt.ylabel('Średnia wartość bezwzględna wag')
plt.title('Zmiany wag w czasie - warstwa FC2')
plt.legend()
plt.grid(True)
plt.savefig('weights_fc2.png')
plt.show()

# 7. Macierz konfuzji
_, preds, targets = evaluate()
cm = confusion_matrix(targets, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(figsize=(8,8))
disp.plot(cmap=plt.cm.Blues)
plt.title('Macierz konfuzji - zbiór testowy')
plt.savefig('confusion_matrix.png')
plt.show()

# 8. Przykładowe predykcje
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
data = example_data[:5]
targets = example_targets[:5]

model.eval()
with torch.no_grad():
    output = model(data)
    probs = torch.exp(output)

for i in range(5):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(data[i][0], cmap='gray')
    plt.title(f"Obraz wejściowy\nRzeczywista: {targets[i].item()} | Przewidziana: {probs[i].argmax().item()}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.bar(np.arange(10), probs[i].cpu().numpy(), color='skyblue')
    plt.xticks(np.arange(10))
    plt.title("Rozkład prawdopodobieństw dla klas")
    plt.xlabel("Cyfra")
    plt.ylabel("Prawdopodobieństwo")
    plt.tight_layout()
    plt.show()
