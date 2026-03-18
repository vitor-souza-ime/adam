import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------------------
# 1. Reprodutibilidade
# -------------------------------
torch.manual_seed(0)

# -------------------------------
# 2. Dados (não linear + ruído)
# -------------------------------
x = torch.unsqueeze(torch.linspace(-2, 2, 200), dim=1)
y = torch.sin(3 * x) + 0.5 * torch.randn(x.size())

# -------------------------------
# 3. Modelo
# -------------------------------
def create_model():
    return nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

model_sgd = create_model()
model_adam = create_model()

# -------------------------------
# 4. Otimizadores
# -------------------------------
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.01)

loss_fn = nn.MSELoss()

losses_sgd = []
losses_adam = []

# -------------------------------
# 5. Treinamento
# -------------------------------
for epoch in range(200):

    # SGD
    optimizer_sgd.zero_grad()
    pred_sgd = model_sgd(x)
    loss_sgd = loss_fn(pred_sgd, y)
    loss_sgd.backward()
    optimizer_sgd.step()
    losses_sgd.append(loss_sgd.item())

    # Adam
    optimizer_adam.zero_grad()
    pred_adam = model_adam(x)
    loss_adam = loss_fn(pred_adam, y)
    loss_adam.backward()
    optimizer_adam.step()
    losses_adam.append(loss_adam.item())

# -------------------------------
# 6. Impressão COMPLETA (todas épocas)
# -------------------------------
print("\n=== TODOS OS VALORES ===")
print("Epoch |   SGD       |   Adam")
print("---------------------------------")

for i in range(len(losses_sgd)):
    print(f"{i:5d} | {losses_sgd[i]:.6f} | {losses_adam[i]:.6f}")

# -------------------------------
# 7. Impressão REDUZIDA
# -------------------------------
print("\n=== VALORES (de 10 em 10 épocas) ===")
print("Epoch |   SGD       |   Adam")
print("---------------------------------")

for i in range(0, len(losses_sgd), 10):
    print(f"{i:5d} | {losses_sgd[i]:.6f} | {losses_adam[i]:.6f}")

# -------------------------------
# 8. Últimas épocas
# -------------------------------
print("\n=== ÚLTIMAS ÉPOCAS ===")
print("Epoch |   SGD       |   Adam")
print("---------------------------------")

for i in range(len(losses_sgd)-10, len(losses_sgd)):
    print(f"{i:5d} | {losses_sgd[i]:.6f} | {losses_adam[i]:.6f}")

# -------------------------------
# 9. Comparação final
# -------------------------------
print("\n=== COMPARAÇÃO FINAL ===")
print(f"SGD final:  {losses_sgd[-1]:.6f}")
print(f"Adam final: {losses_adam[-1]:.6f}")

if losses_adam[-1] < losses_sgd[-1]:
    print("Adam teve melhor desempenho ✅")
else:
    print("SGD teve melhor desempenho ❗")

# -------------------------------
# 10. Formato CSV
# -------------------------------
print("\n=== CSV ===")
print("Epoch,SGD,Adam")

for i in range(len(losses_sgd)):
    print(f"{i},{losses_sgd[i]:.6f},{losses_adam[i]:.6f}")

# -------------------------------
# 11. Gráfico
# -------------------------------
plt.plot(losses_sgd, label="SGD")
plt.plot(losses_adam, label="Adam")
plt.xlabel("Epoch")
plt.ylabel("Loss (Erro)")
plt.title("Comparação: SGD vs Adam")
plt.legend()
plt.show()
