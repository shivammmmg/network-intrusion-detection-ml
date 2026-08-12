export interface ModelDisplayInfo {
  id: string;
  name: string;
  category: string;
}

export const MODEL_DISPLAY: ModelDisplayInfo[] = [
  { id: "logistic_regression", name: "Logistic Regression", category: "Linear model" },
  { id: "neural_network", name: "Neural Network", category: "Multilayer perceptron" },
  { id: "random_forest", name: "Random Forest", category: "Tree ensemble" },
  { id: "xgboost", name: "XGBoost", category: "Gradient-boosted trees" }
];
