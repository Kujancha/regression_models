import numpy as np




class LogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_biased = np.c_[np.ones((len(X), 1)), self.X]
        self.n_samples, self.n_features = self.X_biased.shape
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
        
    
    def fit(self, alpha = 0.1, max_ite = 100):
        
        theta = np.zeros(self.n_features)
        
        for i in range(max_ite):
            z = self.X_biased @ theta
            y_pred = self.sigmoid(z) - self.y
            gradient = self.X_biased.T @ y_pred
            gradient  /= self.n_samples
            theta = theta - alpha * gradient
            
        self.theta = theta
        return theta
    
    def predict_probability(self, test_set):
        test_set_biased = np.c_[np.ones((len(test_set), 1)), test_set]
        return self.sigmoid(test_set_biased @ self.theta)
    
    def predict(self,X, threshold = 0.5):
        return (self.predict_probability(X) > threshold).astype(int)

    def confusion_matrix(self, y_true, y_pred, class_names=None):
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        # Default class names if not provided lol
        if class_names is None:
            class_names = ['Class 0', 'Class 1']
        
      
        total = len(y_true)
        accuracy = (TP + TN) / total if total > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("╔══════════════════════════════════════════════╗")
        print("║             CONFUSION MATRIX                 ║")
        print("╠══════════════════════════════════════════════╣")
        print("║              Predicted Labels                ║")
        print("╠══════════════════════════╦══════════╦════════╣")
        print(f"║                          ║ {class_names[0]:<8} ║ {class_names[1]:<6} ║")
        print("║                          ║   (0)    ║  (1)   ║")
        print("╠══════════════════════════╬══════════╬════════╣")
        print("║         Actual           ║          ║        ║")
        print("║         Labels           ║          ║        ║")
        print("╠══════════════════════════╬══════════╬════════╣")
        print(f"║     {class_names[0]:<15} ║    {TN:<4}  ║  {FP:<4}  ║")
        print("╠══════════════════════════╬══════════╬════════╣")
        print(f"║     {class_names[1]:<15} ║    {FN:<4}  ║  {TP:<4}  ║")
        print("╚══════════════════════════╩══════════╩════════╝")
        print()
        print("╔══════════════════════════════════════╗")
        print("║           PERFORMANCE METRICS        ║")
        print("╠══════════════════════════════════════╣")
        print(f"║ Accuracy:  {accuracy:>7.2%} ({TP+TN:>2}/{total:>2})        ║")
        print(f"║ Precision: {precision:>7.2%}                    ║")
        print(f"║ Recall:    {recall:>7.2%}                    ║")
        print(f"║ F1-Score:  {f1:>7.2%}                    ║")
        print("╚══════════════════════════════════════╝")
        
        return np.array([[TN, FP],
                        [FN, TP]])

