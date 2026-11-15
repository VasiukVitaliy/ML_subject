import numpy as np


class MLP:
    def __init__(self, layers, loss, optimizer, l1=0.0, l2=0.0,
                 dropout_probs=None, weight_clip=None, grad_clip=None):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

        self.l1 = l1
        self.l2 = l2
        self.weight_clip = weight_clip
        self.grad_clip = grad_clip

        n_hidden = len(self.layers) - 1
        if dropout_probs is None:
            dropout_probs = [0.0] * n_hidden
        if len(dropout_probs) != n_hidden:
            raise ValueError("Кількість dropout-ймовірностей має дорівнювати кількості прихованих шарів")
        for p in dropout_probs:
            if not 0 <= p < 1:
                raise ValueError("Ймовірність dropout має бути в діапазоні [0, 1)")
        self.dropout_probs = list(dropout_probs)

        self.dropout_masks = [None] * len(self.layers)
        self.history = None


    def regularize(self, w):
        reg = 0.0
        if self.l1:
            reg += self.l1 * np.sum(np.abs(w))
        if self.l2:
            reg += (self.l2 / 2) * np.sum(w ** 2)
        return reg

    def global_regularize_error(self):
        return sum(self.regularize(layer.W) for layer in self.layers)

    def regularize_grad(self, w):
        grad = np.zeros_like(w)
        if self.l1:
            grad += self.l1 * np.sign(w)
        if self.l2:
            grad += self.l2 * w
        return grad


    def forward(self, x, training=False):
        self.dropout_masks = [None] * len(self.layers)
        last = len(self.layers) - 1

        for i, layer in enumerate(self.layers):
            x = layer.forward(x)

            if i < last and training and self.dropout_probs[i] > 0:
                p = self.dropout_probs[i]
                mask = (np.random.rand(*x.shape) >= p) / (1.0 - p)  # inverted dropout
                self.dropout_masks[i] = mask
                x = x * mask

        return x

    def backward(self, P, y_true):
        dP = self.loss.backward(P, y_true)

        gradients = [None] * len(self.layers)
        da = dP
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            da, grad = layer.backward(da)
            grad["W"] = grad["W"] + self.regularize_grad(layer.W)
            gradients[i] = grad

            if i > 0 and self.dropout_masks[i - 1] is not None:
                da = da * self.dropout_masks[i - 1]

        return gradients


    def clip_gradients(self, gradients):
        if self.grad_clip is None:
            return
        total = np.sqrt(sum(np.sum(g["W"] ** 2) + np.sum(g["b"] ** 2) for g in gradients))
        if total > self.grad_clip:
            scale = self.grad_clip / (total + 1e-12)
            for g in gradients:
                g["W"] *= scale
                g["b"] *= scale

    def update(self, gradients):
        self.clip_gradients(gradients)

        for layer, grad in zip(self.layers, gradients):
            self.optimizer.step(layer.parameters(), grad)

        if self.weight_clip is not None:
            for layer in self.layers:
                np.clip(layer.W, -self.weight_clip, self.weight_clip, out=layer.W)


    def fit(self, X, y, epochs=100, batch_size=32, X_val=None, y_val=None):
        self.history = {
            "train_ce": [], "train_reg": [], "train_acc": [],
            "val_ce": [], "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            indexes = np.random.permutation(len(X))

            ce_sum = 0.0
            reg_sum = 0.0
            batches = 0

            for start in range(0, len(X), batch_size):
                idx = indexes[start:start + batch_size]
                X_batch, y_batch = X[idx], y[idx]

                P = self.forward(X_batch, training=True)
                ce = self.loss.forward(P, y_batch)
                reg = self.global_regularize_error()

                gradients = self.backward(P, y_batch)
                self.update(gradients)

                ce_sum += ce
                reg_sum += reg
                batches += 1

            train_ce = ce_sum / batches
            train_reg = reg_sum / batches

            self.history["train_ce"].append(train_ce)
            self.history["train_reg"].append(train_reg)

            # train accuracy на всій тренувальній вибірці в inference режимі (без dropout),
            # щоб її можна було чесно порівнювати з val accuracy
            train_acc = np.mean(self.predict(X) == y)
            self.history["train_acc"].append(train_acc)

            msg = (f"Epoch {epoch}/{epochs}, CE: {train_ce:.6f}, "
                   f"Reg: {train_reg:.6f}, Train acc: {train_acc:.4f}")

            if X_val is not None:
                P_val = self.predict_proba(X_val)
                val_ce = self.loss.forward(P_val, y_val)
                val_acc = np.mean(np.argmax(P_val, axis=1) == y_val)

                self.history["val_ce"].append(val_ce)
                self.history["val_acc"].append(val_acc)
                msg += f", Val CE: {val_ce:.6f}, Val acc: {val_acc:.4f}"

            print(msg)

    def predict_proba(self, X):
        return self.forward(X, training=False)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


    def dead_neurons(self, X):
        """Частка мертвих нейронів (%) для кожного прихованого шару, inference режим."""
        self.forward(X, training=False)
        result = []
        for layer in self.layers[:-1]:
            dead = np.sum(np.max(layer.z, axis=0) <= 0)
            result.append(100.0 * dead / layer.z.shape[1])
        return result

    def sparsity(self, threshold=0.01):
        """Частка |W| < threshold (%) по кожному шару і загалом. Лише W, без b."""
        per_layer = [100.0 * np.mean(np.abs(layer.W) < threshold) for layer in self.layers]
        overall = 100.0 * np.mean(np.abs(self.weights_flat()) < threshold)
        return per_layer, overall

    def weights_flat(self):
        """Усі W одним масивом (без b) для гістограм."""
        return np.concatenate([layer.W.ravel() for layer in self.layers])

    def count_at_bounds(self):
        if self.weight_clip is None:
            return 0
        return int(np.sum(np.abs(self.weights_flat()) >= self.weight_clip))

    # ---------- збереження і відновлення ----------

    def save_weights(self, path):
        data = {}
        for i, layer in enumerate(self.layers):
            data[f"W_{i}"] = layer.W
            data[f"b_{i}"] = layer.b

        data["num_layers"] = np.array(len(self.layers))
        data["layer_sizes"] = np.array(
            [self.layers[0].W.shape[0]] + [layer.W.shape[1] for layer in self.layers]
        )
        data["l1"] = np.array(self.l1)
        data["l2"] = np.array(self.l2)
        data["dropout_probs"] = np.array(self.dropout_probs)
        data["weight_clip"] = np.array(np.nan if self.weight_clip is None else self.weight_clip)
        data["grad_clip"] = np.array(np.nan if self.grad_clip is None else self.grad_clip)

        np.savez(path, **data)

    def load_weights(self, path):
        """Завантажує ваги і конфігурацію в мережу з тією самою архітектурою."""
        data = np.load(path)

        num_layers = int(data["num_layers"])
        if num_layers != len(self.layers):
            raise ValueError(f"Файл має {num_layers} шарів, мережа — {len(self.layers)}")

        for i, layer in enumerate(self.layers):
            W = data[f"W_{i}"]
            b = data[f"b_{i}"]
            if W.shape != layer.W.shape or b.shape != layer.b.shape:
                raise ValueError(f"Невідповідність форм у шарі {i}")
            layer.W = W.copy()
            layer.b = b.copy()

        self.l1 = float(data["l1"])
        self.l2 = float(data["l2"])
        self.dropout_probs = list(data["dropout_probs"])
        wc = float(data["weight_clip"])
        self.weight_clip = None if np.isnan(wc) else wc
        gc = float(data["grad_clip"])
        self.grad_clip = None if np.isnan(gc) else gc