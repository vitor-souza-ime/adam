# 📊 Comparação entre Otimizadores: SGD vs Adam em Redes Neurais

Este projeto apresenta uma análise prática do comportamento de dois algoritmos de otimização amplamente utilizados no treinamento de redes neurais artificiais:

- **SGD (Stochastic Gradient Descent)**
- **Adam (Adaptive Moment Estimation)**

O objetivo é demonstrar, de forma experimental e visual, as diferenças de desempenho entre esses métodos no processo de minimização da função de erro.

---

## 🎯 Objetivo

Investigar como diferentes estratégias de otimização influenciam:

- A velocidade de convergência
- A estabilidade do treinamento
- O erro final alcançado pelo modelo

---

## 🧠 Fundamentação

O treinamento de redes neurais consiste na minimização de uma função de erro (loss). Para isso, algoritmos de otimização ajustam iterativamente os pesos do modelo com base nos gradientes calculados via backpropagation.

- **SGD** utiliza uma taxa de aprendizado fixa
- **Adam** adapta dinamicamente o passo de atualização com base no histórico dos gradientes

---

## 🧪 Metodologia

O experimento consiste em:

- Gerar dados sintéticos não lineares com ruído
- Treinar uma rede neural feedforward
- Comparar o desempenho dos otimizadores ao longo de 200 épocas

### 📌 Função alvo:

```

y = sin(3x) + ruído

````

### 🧱 Arquitetura da rede:

- Entrada: 1 neurônio
- Camadas ocultas: 2 (32 neurônios cada)
- Função de ativação: ReLU
- Saída: 1 neurônio

---

## ⚙️ Tecnologias Utilizadas

- Python
- PyTorch
- Matplotlib

---

## ▶️ Execução

### 1. Instale as dependências:

```bash
pip install torch matplotlib
````

### 2. Execute o script:

```bash
python main.py
```

---

## 📊 Saídas do Programa

O código fornece:

### ✔️ 1. Impressão completa dos erros por época

Tabela com valores de loss para SGD e Adam

### ✔️ 2. Impressão reduzida

Valores amostrados a cada 10 épocas

### ✔️ 3. Últimas épocas

Foco no comportamento final do treinamento

### ✔️ 4. Comparação final

Resumo direto do desempenho

### ✔️ 5. Exportação em formato CSV

Compatível com Excel e ferramentas científicas

### ✔️ 6. Visualização gráfica

Curvas de erro ao longo das épocas

---

## 📈 Resultados Esperados

* O **Adam apresenta convergência mais rápida**
* O **Adam atinge menor erro final**
* O **SGD apresenta comportamento mais lento e menos eficiente**

---

## 🧠 Interpretação

O experimento evidencia que:

> O Adam é um otimizador adaptativo capaz de ajustar dinamicamente a taxa de aprendizado, resultando em maior eficiência em problemas não lineares e com ruído.

---

## 📌 Conclusão

Os resultados demonstram que algoritmos adaptativos como o Adam são mais adequados para cenários complexos, enquanto o SGD pode ser suficiente para problemas mais simples.

---

## 🚀 Possíveis Extensões

* Comparação com outros otimizadores (RMSProp, Adagrad)
* Testes com redes mais profundas
* Aplicação em datasets reais
* Análise estatística dos resultados

---

## 📄 Licença

Este projeto é destinado a fins acadêmicos e educacionais.


